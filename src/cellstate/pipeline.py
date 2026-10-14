"""End-to-end local multi-dataset workflow with reproducibility manifest."""

from __future__ import annotations

import logging
import platform
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy
import scipy
import sklearn

from .clustering import cluster_cells
from .config import CellStateConfig
from .embedding import compute_embedding
from .integration import (
    combat_representation,
    compare_signature_tables,
    diagnose_condition_batch_confounding,
)
from .io import adata_checksum, load_dataset, sha256_path, validate_adata, validate_metadata
from .metrics import cluster_stability, composition_differences, composition_table
from .preprocessing import preprocess
from .qc import calculate_qc, filter_qc
from .reporting import save_dataset_outputs, write_json
from .signatures import pseudobulk_signatures

logger = logging.getLogger(__name__)


def _qc_frame(adata: ad.AnnData, metadata_keys: tuple[str, ...] = ()) -> pd.DataFrame:
    cols = [
        k
        for k in ("n_genes_by_counts", "total_counts", "pct_counts_mt", *metadata_keys)
        if k in adata.obs
    ]
    return adata.obs[cols].copy()


def run_pipeline(
    inputs: list[str | Path | ad.AnnData],
    config: CellStateConfig,
    *,
    run_combat: bool = False,
    run_sensitivity: bool = False,
    sensitivity_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run QC, preprocessing, embeddings, clustering, pseudobulk signatures, and exports.

    Inputs remain local and are never downloaded. Cluster identities are analyzed
    separately per dataset; cross-dataset comparisons use only supplied matching
    cell-type annotations and matching condition contrasts.
    """
    if not inputs:
        raise ValueError("at least one input dataset is required")
    started_unix = time.time()
    timer = time.perf_counter()
    root = Path(config.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    reports = []
    sensitivity_inputs = []
    all_warnings = []
    output_files = []
    sigs = []
    ids = []
    for ordinal, source in enumerate(inputs):
        source_path = None
        if isinstance(source, (str, Path)):
            source_path = Path(source).expanduser().resolve()
            data = load_dataset(source_path, dataset_key=config.dataset_key)
        elif isinstance(source, ad.AnnData):
            data = validate_adata(source, dataset_key=config.dataset_key)
        else:
            raise TypeError("inputs must be local paths or AnnData objects")
        dataset_values = data.obs[config.dataset_key].astype(str).unique()
        if len(dataset_values) != 1:
            raise ValueError("each input must represent exactly one dataset_id")
        dataset_id = str(dataset_values[0])
        if dataset_id in ids:
            dataset_id = f"{dataset_id}_{ordinal + 1}"
            data.obs[config.dataset_key] = dataset_id
        ids.append(dataset_id)
        sensitivity_inputs.append(data.copy())
        data = calculate_qc(data, mito_prefixes=config.qc.mito_prefixes)
        qc_before = _qc_frame(
            data,
            (
                config.dataset_key,
                config.sample_key,
                config.donor_key,
                config.condition_key,
                config.batch_key,
            ),
        )
        filtered, account = filter_qc(data, config.qc)
        batch_key = config.batch_key if config.batch_key in filtered.obs else None
        pre = preprocess(filtered, config.preprocessing, batch_key=batch_key)
        embedded = compute_embedding(pre, config.embedding, seed=config.seed)
        clustered = cluster_cells(embedded, config.clustering, seed=config.seed)
        signature_table = pd.DataFrame()
        signature_info = {
            "status": "inconclusive",
            "reason": "condition or annotated cell type metadata unavailable",
        }
        if config.condition_key in clustered.obs and config.cell_type_key in clustered.obs:
            try:
                signature_table, signature_info = pseudobulk_signatures(
                    filtered,
                    condition_key=config.condition_key,
                    cell_type_key=config.cell_type_key,
                    sample_key=config.sample_key,
                    donor_key=config.donor_key,
                )
            except ValueError as exc:
                signature_info = {"status": "inconclusive", "reason": str(exc)}
            if not signature_table.empty:
                signature_table["condition_a"] = signature_info.get("condition_a")
                signature_table["condition_b"] = signature_info.get("condition_b")
        composition = pd.DataFrame()
        composition_status = {"status": "inconclusive", "reason": "required metadata missing"}
        composition_keys = (config.sample_key, config.condition_key, config.cell_type_key)
        composition_valid = (
            all(k in filtered.obs for k in composition_keys)
            and not filtered.obs[list(composition_keys)].isna().any().any()
        )
        if composition_valid:
            composition_status = {"status": "descriptive"}
            composition = composition_table(
                filtered.obs,
                sample_key=config.sample_key,
                condition_key=config.condition_key,
                cell_type_key=config.cell_type_key,
            )
        composition_diff = []
        if composition_valid:
            composition_diff = composition_differences(
                filtered.obs,
                sample_key=config.sample_key,
                donor_key=config.donor_key,
                condition_key=config.condition_key,
                cell_type_key=config.cell_type_key,
            )
        confounding = diagnose_condition_batch_confounding(
            clustered.obs, condition_key=config.condition_key, batch_key=config.batch_key
        )
        combat_info = {"status": "not_requested"}
        if run_combat:
            clustered, combat_info = combat_representation(
                clustered, batch_key=config.batch_key, condition_key=config.condition_key
            )
        if run_combat and combat_info.get("status") == "ok":
            combat_key = f"{config.clustering.key}_combat"
            combat_config = replace(config.clustering, key=combat_key)
            corrected = compute_embedding(
                clustered, config.embedding, seed=config.seed, use_rep="X_combat_pca"
            )
            corrected = cluster_cells(corrected, combat_config, seed=config.seed)
            clustered.obs[combat_key] = corrected.obs[combat_key].astype(str)
            clustered.obsm["X_combat_pca"] = corrected.obsm["X_combat_pca"].copy()
            clustered.obsm["X_umap_combat"] = corrected.obsm["X_umap"].copy()
            combat_info["comparison"] = {
                "original_vs_combat_cluster_ari": cluster_stability(
                    clustered.obs[config.clustering.key], clustered.obs[combat_key]
                ),
                "original_cluster_count": int(clustered.obs[config.clustering.key].nunique()),
                "combat_cluster_count": int(clustered.obs[combat_key].nunique()),
                "interpretation": "Partition change is descriptive and cannot establish that technical variation was removed or biological signal preserved.",
            }
            combat_file = root / "datasets" / dataset_id / "combat_comparison.json"
            write_json(combat_file, combat_info)
            output_files.append(str(combat_file))
        out_dir = root / "datasets" / dataset_id
        files = save_dataset_outputs(
            clustered,
            out_dir,
            qc_accounting=account,
            qc_before=qc_before,
            signature_table=signature_table,
            composition=composition,
            composition_differences=composition_diff,
            condition_key=config.condition_key,
            cluster_key=config.clustering.key,
            metadata_keys=(
                config.dataset_key,
                config.sample_key,
                config.donor_key,
                config.condition_key,
                config.batch_key,
                config.cell_type_key,
            ),
        )
        output_files.extend(files)
        sigs.append(signature_table)
        provenance = data.uns.get("cellstate", {}).get("provenance", {})
        if source_path and not provenance:
            provenance = {"local_path": str(source_path), "sha256": sha256_path(source_path)}
        if not provenance:
            provenance = {
                "source": "in_memory",
                "count_matrix_sha256": adata_checksum(data),
                "synthetic": bool(data.uns.get("cellstate", {}).get("synthetic", False)),
            }
        metadata_diagnostics = validate_metadata(
            data,
            [
                config.dataset_key,
                config.sample_key,
                config.donor_key,
                config.condition_key,
                config.cell_type_key,
                config.batch_key,
            ],
        )
        warnings_for_dataset = list(clustered.uns.get("cellstate", {}).get("warnings", []))
        if confounding.get("confounded"):
            warnings_for_dataset.append(str(confounding.get("reason")))
        if dataset_id == "unspecified":
            warnings_for_dataset.append(
                "dataset_id was not supplied; input was labeled unspecified."
            )
        if metadata_diagnostics["missing_columns"]:
            warnings_for_dataset.append(
                f"Missing metadata columns: {metadata_diagnostics['missing_columns']}"
            )
        if metadata_diagnostics["rows_with_missing_values"]:
            warnings_for_dataset.append(
                "Missing values in metadata: "
                + ", ".join(metadata_diagnostics["rows_with_missing_values"])
            )
        if signature_info.get("status") == "inconclusive":
            warnings_for_dataset.append(str(signature_info.get("reason")))
        all_warnings.extend(f"{dataset_id}: {w}" for w in warnings_for_dataset)
        reports.append(
            {
                "dataset_id": dataset_id,
                "input_path": str(source_path) if source_path else None,
                "input_provenance": provenance,
                "n_cells_before_qc": int(data.n_obs),
                "n_cells_after_qc": int(clustered.n_obs),
                "n_genes_after_hvg": int(clustered.n_vars),
                "qc_accounting": account,
                "preprocessing_history": clustered.uns.get("cellstate", {}).get(
                    "preprocessing_history", []
                ),
                "signature_analysis": signature_info,
                "condition_batch_diagnostic": confounding,
                "composition_analysis": composition_status,
                "metadata_diagnostics": metadata_diagnostics,
                "composition_differences": composition_diff,
                "combat": combat_info,
                "cluster_counts": clustered.obs[config.clustering.key]
                .value_counts()
                .sort_index()
                .astype(int)
                .to_dict(),
                "warnings": warnings_for_dataset,
                "outputs": files,
            }
        )
    cross = compare_signature_tables(sigs, ids, cell_type_key="cell_type") if len(sigs) > 1 else []
    for pair in cross:
        if pair.get("status") == "inconclusive":
            all_warnings.append(f"Cross-dataset comparison inconclusive: {pair.get('reason')}")
    sensitivity = {"status": "not_requested"}
    if run_sensitivity:
        from .validation import run_sensitivity as _run_sensitivity

        options = sensitivity_options or {
            "min_genes_values": list(config.sensitivity.min_genes),
            "hvg_values": list(config.sensitivity.n_top_genes),
            "resolutions": list(config.sensitivity.resolutions),
            "seeds": list(config.sensitivity.seeds),
            "max_runs": config.sensitivity.max_runs,
        }
        sensitivity = {
            "dataset_id": ids[0],
            **_run_sensitivity(sensitivity_inputs[0], config, **options),
        }
        sp = root / "sensitivity.json"
        write_json(sp, sensitivity)
        output_files.append(str(sp))
    manifest = {
        "cellstate_version": "0.1.0",
        "started_unix_seconds": started_unix,
        "elapsed_seconds": time.perf_counter() - timer,
        "software_versions": {
            "python": platform.python_version(),
            "scanpy": scanpy.__version__,
            "anndata": ad.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "configuration": config.to_dict(),
        "random_seed": config.seed,
        "dataset_runs": reports,
        "cross_dataset_signature_comparisons": cross,
        "sensitivity": sensitivity,
        "warnings": all_warnings,
        "output_files": output_files,
        "scientific_scope": "Exploratory reproducibility analysis; no causal claims or real-world performance claims.",
        "status": "complete_with_warnings" if all_warnings else "complete",
    }
    manifest_path = root / "run_manifest.json"
    write_json(manifest_path, manifest)
    results_path = root / "results.json"
    write_json(
        results_path,
        {
            "datasets": reports,
            "cross_dataset_signature_comparisons": cross,
            "sensitivity": sensitivity,
            "warnings": all_warnings,
        },
    )
    manifest["output_files"] += [str(manifest_path), str(results_path)]
    # Rewrite manifest to include its companion summary path.
    write_json(manifest_path, manifest)
    logger.info(
        "CellState run completed for %d dataset(s) in %.2fs",
        len(inputs),
        manifest["elapsed_seconds"],
    )
    return manifest
