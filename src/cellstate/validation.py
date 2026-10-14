"""Cluster/signature stability metrics and bounded parameter-sensitivity runs."""

from __future__ import annotations

import time
import warnings
from dataclasses import replace
from itertools import product
from typing import Any

import anndata as ad

from .clustering import cluster_cells
from .config import CellStateConfig
from .embedding import compute_embedding
from .metrics import cluster_stability
from .preprocessing import preprocess
from .qc import filter_qc


def compare_clusterings(
    left: ad.AnnData, right: ad.AnnData, *, key: str = "leiden"
) -> dict[str, Any]:
    """ARI on shared cell identifiers; changed filtering is handled by intersection."""
    if key not in left.obs or key not in right.obs:
        return {"status": "inconclusive", "reason": f"cluster key {key!r} missing"}
    shared = left.obs_names.intersection(right.obs_names)
    if len(shared) < 2:
        return {
            "status": "inconclusive",
            "reason": "fewer than two shared cells",
            "shared_cells": len(shared),
        }
    score = cluster_stability(left.obs.loc[shared, key], right.obs.loc[shared, key])
    return {
        "status": "ok",
        "metric": "adjusted_rand_index",
        "score": score,
        "shared_cells": len(shared),
        "limitations": "ARI measures partition agreement, not biological validity; only shared cells are scored.",
    }


def run_sensitivity(
    adata: ad.AnnData,
    config: CellStateConfig,
    *,
    min_genes_values: list[int] | None = None,
    hvg_values: list[int] | None = None,
    resolutions: list[float] | None = None,
    seeds: list[int] | None = None,
    combat_options: list[bool] | None = None,
    max_runs: int = 12,
) -> dict[str, Any]:
    """Run a small Cartesian sensitivity grid and compare clusters on shared cells."""
    grid = list(
        product(
            min_genes_values or [config.qc.min_genes],
            hvg_values or [config.preprocessing.n_top_genes],
            resolutions or [config.clustering.resolution],
            seeds or [config.seed],
            [False] if combat_options is None else combat_options,
        )
    )
    if len(grid) > max_runs:
        raise ValueError(f"sensitivity grid has {len(grid)} runs; max_runs={max_runs}")
    results = []
    outputs = []
    output_run_ids = []
    for run_id, (min_genes, n_hvg, res, seed, use_combat) in enumerate(grid):
        started = time.perf_counter()
        run_config = replace(
            config,
            qc=replace(config.qc, min_genes=min_genes),
            preprocessing=replace(config.preprocessing, n_top_genes=n_hvg),
            clustering=replace(config.clustering, resolution=res),
            seed=seed,
        )
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                filtered, account = filter_qc(adata, run_config.qc)
                pre = preprocess(
                    filtered,
                    run_config.preprocessing,
                    batch_key=run_config.batch_key
                    if run_config.batch_key in filtered.obs
                    else None,
                )
                embedded = compute_embedding(pre, run_config.embedding, seed=seed)
                integration = {"status": "not_requested"}
                if use_combat:
                    from .integration import combat_representation

                    embedded, integration = combat_representation(
                        embedded, batch_key=config.batch_key, condition_key=config.condition_key
                    )
                    if integration.get("status") != "ok":
                        results.append(
                            {
                                "config": {
                                    "min_genes": min_genes,
                                    "n_top_genes": n_hvg,
                                    "resolution": res,
                                    "seed": seed,
                                    "combat": use_combat,
                                },
                                "status": "inconclusive",
                                "integration": integration,
                                "elapsed_seconds": time.perf_counter() - started,
                            }
                        )
                        continue
                    embedded = compute_embedding(
                        embedded, run_config.embedding, seed=seed, use_rep="X_combat_pca"
                    )
                clustered = cluster_cells(embedded, run_config.clustering, seed=seed)
            outputs.append(clustered)
            output_run_ids.append(run_id)
            results.append(
                {
                    "config": {
                        "min_genes": min_genes,
                        "n_top_genes": n_hvg,
                        "resolution": res,
                        "seed": seed,
                        "combat": use_combat,
                    },
                    "n_cells": int(clustered.n_obs),
                    "n_genes": int(clustered.n_vars),
                    "qc": account,
                    "warnings": [str(w.message) for w in captured],
                    "elapsed_seconds": time.perf_counter() - started,
                    "status": "ok",
                    "integration": integration,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "config": {
                        "min_genes": min_genes,
                        "n_top_genes": n_hvg,
                        "resolution": res,
                        "seed": seed,
                        "combat": use_combat,
                    },
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
    pairwise = []
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            pairwise.append(
                {
                    "run_a": output_run_ids[i],
                    "run_b": output_run_ids[j],
                    **compare_clusterings(outputs[i], outputs[j], key=config.clustering.key),
                }
            )
    return {
        "runs": results,
        "pairwise_cluster_stability": pairwise,
        "limitations": "Sensitivity scores measure algorithmic stability under supplied settings; they are not evidence of cell-type validity.",
    }
