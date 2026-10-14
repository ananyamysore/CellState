"""Graph-based clustering with explicit algorithms and random seeds."""

from __future__ import annotations

import anndata as ad
import scanpy as sc

from .config import ClusterConfig


def cluster_cells(
    adata: ad.AnnData, config: ClusterConfig, *, seed: int = 0, algorithm: str = "leiden"
) -> ad.AnnData:
    """Run Leiden or Louvain on the Scanpy neighbor graph."""
    if "neighbors" not in adata.uns:
        raise ValueError("neighbor graph missing; call compute_embedding first")
    if algorithm not in {"leiden", "louvain"}:
        raise ValueError("algorithm must be leiden or louvain")
    out = adata.copy()
    if algorithm == "leiden":
        sc.tl.leiden(
            out,
            resolution=config.resolution,
            key_added=config.key,
            random_state=seed,
            flavor="leidenalg",
            n_iterations=2,
            directed=False,
        )
    else:
        sc.tl.louvain(out, resolution=config.resolution, key_added=config.key, random_state=seed)
    out.uns.setdefault("cellstate", {})["clustering"] = {
        "algorithm": algorithm,
        "resolution": config.resolution,
        "key": config.key,
        "seed": seed,
    }
    return out
