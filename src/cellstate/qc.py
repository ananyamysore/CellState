"""Quality-control summaries and auditable cell filtering."""

from __future__ import annotations

import logging
from typing import Any

import anndata as ad
import numpy as np
import scanpy as sc

from .config import QCConfig

logger = logging.getLogger(__name__)


def calculate_qc(
    adata: ad.AnnData, *, mito_prefixes: tuple[str, ...] = ("MT-", "mt-")
) -> ad.AnnData:
    """Annotate counts, detected genes, and mitochondrial percentage when identifiable."""
    out = adata.copy()
    if "counts" not in out.layers:
        raise ValueError("raw counts layer is required for QC")
    out.X = out.layers["counts"].copy()
    sc.pp.calculate_qc_metrics(out, qc_vars=[], percent_top=None, log1p=False, inplace=True)
    mito = np.zeros(out.n_vars, dtype=bool)
    if "mt" in out.var:
        mito = np.asarray(out.var["mt"], dtype=bool)
    else:
        names = out.var_names.astype(str)
        for prefix in mito_prefixes:
            mito |= np.char.startswith(names, prefix)
    if mito.any():
        counts = out.layers["counts"]
        mt = np.asarray(counts[:, mito].sum(axis=1)).ravel()
        total = np.asarray(counts.sum(axis=1)).ravel()
        out.obs["pct_counts_mt"] = np.divide(
            100 * mt, total, out=np.zeros_like(mt, dtype=float), where=total > 0
        )
        out.uns.setdefault("cellstate", {})["mitochondrial_gene_count"] = int(mito.sum())
    else:
        out.uns.setdefault("cellstate", {})["warnings"] = list(
            out.uns.get("cellstate", {}).get("warnings", [])
        ) + [
            "Mitochondrial percentage unavailable: no mitochondrial gene annotation or configured prefix matched."
        ]
    return out


def filter_qc(adata: ad.AnnData, config: QCConfig) -> tuple[ad.AnnData, dict[str, Any]]:
    """Apply filters in sequence and return per-rule removal accounting.

    Optional Scrublet runs only when installed and at least 50 cells are present.
    It is a screening signal rather than a definitive doublet label.
    """
    out = calculate_qc(adata, mito_prefixes=config.mito_prefixes)
    initial = out.n_obs
    accounting = []
    keep = np.ones(initial, dtype=bool)
    rules = [
        ("min_genes", out.obs["n_genes_by_counts"].to_numpy() >= config.min_genes),
        (
            "max_genes",
            np.ones(initial, bool)
            if config.max_genes is None
            else out.obs["n_genes_by_counts"].to_numpy() <= config.max_genes,
        ),
        ("min_counts", out.obs["total_counts"].to_numpy() >= config.min_counts),
        (
            "max_counts",
            np.ones(initial, bool)
            if config.max_counts is None
            else out.obs["total_counts"].to_numpy() <= config.max_counts,
        ),
    ]
    if config.max_mito_percent is not None and "pct_counts_mt" in out.obs:
        rules.append(
            ("max_mito_percent", out.obs["pct_counts_mt"].to_numpy() <= config.max_mito_percent)
        )
    for name, rule in rules:
        before = int(keep.sum())
        keep &= rule
        accounting.append(
            {
                "filter": name,
                "before": before,
                "after": int(keep.sum()),
                "removed": before - int(keep.sum()),
            }
        )
    if config.doublet_method == "scrublet":
        try:
            import scrublet as scr
        except ImportError as exc:
            raise ImportError("install cellstate[doublets] to use Scrublet") from exc
        if int(keep.sum()) < 50:
            raise ValueError("Scrublet requires at least 50 QC-passing cells")
        scores, pred = scr.Scrublet(out.layers["counts"][keep]).scrub_doublets(verbose=False)
        out.obs["doublet_score"] = np.nan
        out.obs["predicted_doublet"] = False
        out.obs.loc[keep, "doublet_score"] = scores
        out.obs.loc[keep, "predicted_doublet"] = pred
        before = int(keep.sum())
        keep &= ~out.obs["predicted_doublet"].to_numpy()
        accounting.append(
            {
                "filter": "scrublet_predicted_doublet",
                "before": before,
                "after": int(keep.sum()),
                "removed": before - int(keep.sum()),
                "interpretation": "optional screening only",
            }
        )
    if not keep.any():
        raise ValueError("QC filters removed every cell")
    filtered = out[keep].copy()
    return filtered, {
        "initial_cells": initial,
        "retained_cells": int(keep.sum()),
        "filters": accounting,
    }
