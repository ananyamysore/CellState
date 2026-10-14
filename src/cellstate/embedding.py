"""Reproducible PCA, neighborhood graph, and UMAP utilities."""

from __future__ import annotations

import anndata as ad
import scanpy as sc

from .config import EmbeddingConfig


def compute_embedding(
    adata: ad.AnnData, config: EmbeddingConfig, *, seed: int = 0, use_rep: str | None = None
) -> ad.AnnData:
    """Compute PCA, neighbors, and UMAP while recording parameters and seed."""
    if adata.n_obs < 3:
        raise ValueError("at least three cells are required for embedding")
    out = adata.copy()
    n_pcs = min(config.n_pcs, max(2, min(out.n_obs - 1, out.n_vars - 1)))
    if use_rep is None:
        sc.pp.pca(out, n_comps=n_pcs, random_state=seed)
    sc.pp.neighbors(
        out,
        n_neighbors=min(config.n_neighbors, out.n_obs - 1),
        n_pcs=n_pcs if use_rep is None else None,
        use_rep=use_rep,
        metric=config.metric,
        random_state=seed,
    )
    sc.tl.umap(out, min_dist=config.min_dist, random_state=seed)
    out.uns.setdefault("cellstate", {})["embedding"] = {
        "method": "PCA/neighbors/UMAP",
        "n_pcs": n_pcs,
        "n_neighbors": min(config.n_neighbors, out.n_obs - 1),
        "metric": config.metric,
        "min_dist": config.min_dist,
        "seed": seed,
        "use_rep": use_rep,
    }
    return out
