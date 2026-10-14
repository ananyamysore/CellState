"""Machine-readable outputs, figures, and run manifest export."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns

logger = logging.getLogger(__name__)


def _write_qc_summary(frame: pd.DataFrame, path: Path) -> None:
    """Summarize per-cell QC distributions by sample/condition where available."""
    metric_columns = [
        column
        for column in ("n_genes_by_counts", "total_counts", "pct_counts_mt")
        if column in frame and pd.api.types.is_numeric_dtype(frame[column])
    ]
    group_columns = [
        column
        for column in ("dataset_id", "sample_id", "donor_id", "condition", "batch")
        if column in frame
    ]
    groups = (
        frame.groupby(group_columns, observed=True, dropna=False)
        if group_columns
        else [("all", frame)]
    )
    rows = []
    for labels, part in groups:
        if not isinstance(labels, tuple):
            labels = (labels,)
        row = dict(zip(group_columns or ["group"], labels))
        row["n_cells"] = int(len(part))
        for column in metric_columns:
            values = part[column].dropna()
            row[f"{column}_mean"] = float(values.mean()) if len(values) else None
            row[f"{column}_median"] = float(values.median()) if len(values) else None
            row[f"{column}_q05"] = float(values.quantile(0.05)) if len(values) else None
            row[f"{column}_q95"] = float(values.quantile(0.95)) if len(values) else None
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def save_dataset_outputs(
    adata: ad.AnnData,
    out_dir: Path,
    *,
    qc_accounting: dict[str, Any],
    qc_before: pd.DataFrame,
    signature_table: pd.DataFrame,
    composition: pd.DataFrame,
    composition_differences: list[dict[str, Any]],
    condition_key: str,
    cluster_key: str,
    metadata_keys: tuple[str, ...] = (),
) -> list[str]:
    """Write cell-level QC, accounting, signature/composition tables, and UMAP figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    qc_before.to_csv(out_dir / "qc_metrics_before.csv", index=True)
    files.append(str(out_dir / "qc_metrics_before.csv"))
    before_summary = out_dir / "qc_summary_before.csv"
    _write_qc_summary(qc_before, before_summary)
    files.append(str(before_summary))
    (out_dir / "qc_filter_accounting.json").write_text(
        json.dumps(qc_accounting, indent=2), encoding="utf-8"
    )
    files.append(str(out_dir / "qc_filter_accounting.json"))
    signature_table.to_csv(out_dir / "signatures.csv", index=False)
    files.append(str(out_dir / "signatures.csv"))
    composition.to_csv(out_dir / "composition_by_sample.csv", index=False)
    files.append(str(out_dir / "composition_by_sample.csv"))
    pd.DataFrame(composition_differences).to_csv(
        out_dir / "composition_differences.csv", index=False
    )
    files.append(str(out_dir / "composition_differences.csv"))
    assignment_cols = [k for k in (*metadata_keys, cluster_key) if k in adata.obs]
    assignments = adata.obs[assignment_cols].copy()
    if "X_umap" in adata.obsm:
        assignments["umap_1"] = adata.obsm["X_umap"][:, 0]
        assignments["umap_2"] = adata.obsm["X_umap"][:, 1]
    if f"{cluster_key}_combat" in adata.obs:
        assignments[f"{cluster_key}_combat"] = adata.obs[f"{cluster_key}_combat"]
    assignments.to_csv(out_dir / "cell_assignments.csv", index=True)
    files.append(str(out_dir / "cell_assignments.csv"))
    obs_cols = [c for c in ("n_genes_by_counts", "total_counts", "pct_counts_mt") if c in adata.obs]
    after_columns = [column for column in (*metadata_keys, *obs_cols) if column in adata.obs]
    qc_after = adata.obs[after_columns].copy()
    qc_after.to_csv(out_dir / "qc_metrics_after.csv", index=True)
    files.append(str(out_dir / "qc_metrics_after.csv"))
    after_summary = out_dir / "qc_summary_after.csv"
    _write_qc_summary(qc_after, after_summary)
    files.append(str(after_summary))
    metric_columns = [
        column
        for column in ("n_genes_by_counts", "total_counts", "pct_counts_mt")
        if column in qc_before and column in qc_after
    ]
    if metric_columns:
        fig, axes = plt.subplots(1, len(metric_columns), figsize=(4 * len(metric_columns), 3))
        axes = [axes] if len(metric_columns) == 1 else list(axes)
        for ax, metric in zip(axes, metric_columns):
            before_values = qc_before[metric].rename("value").to_frame().assign(stage="before")
            after_values = qc_after[metric].rename("value").to_frame().assign(stage="after")
            long = pd.concat([before_values, after_values], ignore_index=True)
            sns.histplot(
                data=long,
                x="value",
                hue="stage",
                stat="density",
                common_norm=False,
                element="step",
                ax=ax,
            )
            ax.set_title(metric)
        fig.tight_layout()
        fig.savefig(out_dir / "qc_distributions.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        files.append(str(out_dir / "qc_distributions.png"))
    if "X_umap" in adata.obsm:
        keys = [k for k in (cluster_key, condition_key) if k in adata.obs]
        if keys:
            try:
                sc.pl.umap(adata, color=keys, show=False, frameon=False)
                fig = plt.gcf()
                fig.savefig(out_dir / "umap.png", dpi=160, bbox_inches="tight")
                plt.close(fig)
                files.append(str(out_dir / "umap.png"))
            except Exception as exc:
                logger.warning("UMAP figure export failed: %s", exc)
        if "X_umap_combat" in adata.obsm:
            original = adata.obsm["X_umap"].copy()
            adata.obsm["X_umap"] = adata.obsm["X_umap_combat"].copy()
            try:
                sc.pl.umap(adata, color=keys, show=False, frameon=False)
                fig = plt.gcf()
                fig.savefig(out_dir / "umap_combat.png", dpi=160, bbox_inches="tight")
                plt.close(fig)
                files.append(str(out_dir / "umap_combat.png"))
            except Exception as exc:
                logger.warning("ComBat UMAP figure export failed: %s", exc)
            finally:
                adata.obsm["X_umap"] = original
    return files


def write_json(path: Path, value: Any) -> None:
    """Serialize with deterministic key order and reject non-standard NaNs."""
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str), encoding="utf-8"
    )
