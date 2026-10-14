"""Raw-count preservation and standard Scanpy preprocessing."""

from __future__ import annotations

import anndata as ad
import scanpy as sc

from .config import PreprocessConfig


def preprocess(
    adata: ad.AnnData, config: PreprocessConfig, *, batch_key: str | None = None
) -> ad.AnnData:
    """Normalize, log1p, select HVGs, and optionally scale while retaining counts.

    The original matrix is copied to ``layers['counts']`` before transformation.
    Normalized log-expression remains in ``X``; a full-feature ``raw`` snapshot is
    kept before feature filtering. ``seurat_v3`` estimates HVGs from raw counts.
    """
    if "counts" not in adata.layers:
        raise ValueError("raw counts layer is required")
    out = adata.copy()
    out.X = out.layers["counts"].copy()
    sc.pp.normalize_total(out, target_sum=config.target_sum)
    sc.pp.log1p(out)
    out.raw = out.copy()
    layer = "counts" if config.flavor == "seurat_v3" else None
    flavor = config.flavor
    if flavor == "seurat_v3" and out.n_vars < config.n_top_genes:
        n_top = out.n_vars
    else:
        n_top = min(config.n_top_genes, out.n_vars)
    sc.pp.highly_variable_genes(
        out, n_top_genes=n_top, flavor=flavor, batch_key=batch_key, layer=layer, subset=True
    )
    if config.scale:
        sc.pp.scale(out, zero_center=True, max_value=config.max_value)
    history = {
        "method": "normalize_total+log1p+highly_variable_genes",
        "target_sum": config.target_sum,
        "hvg_flavor": flavor,
        "n_top_genes": n_top,
        "batch_key": batch_key,
        "scaled": config.scale,
    }
    out.uns.setdefault("cellstate", {}).setdefault("preprocessing_history", []).append(history)
    provenance = out.uns.get("cellstate", {}).get("provenance")
    if provenance is not None:
        provenance.setdefault("preprocessing_history", []).append(history)
    return out
