"""AnnData loading, validation, and local provenance tracking."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse

logger = logging.getLogger(__name__)


def sha256_path(path: Path) -> str:
    """Hash a file in blocks; for directories hash relative filenames and contents."""
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for f in files:
        if path.is_dir():
            digest.update(str(f.relative_to(path)).encode())
        with f.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def validate_adata(
    adata: ad.AnnData,
    *,
    dataset_id: str | None = None,
    dataset_key: str = "dataset_id",
    require_counts: bool = True,
) -> ad.AnnData:
    """Validate nonempty AnnData, unique features, finite nonnegative expression, and IDs.

    The matrix in ``layers['counts']`` is preferred as raw counts; otherwise ``X``
    must be nonnegative and is copied into that layer on returned data.
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("expected an AnnData object")
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("AnnData must have at least one cell and one gene")
    if not adata.var_names.is_unique:
        raise ValueError("gene/feature identifiers must be unique")
    if any(not str(name).strip() for name in adata.var_names):
        raise ValueError("gene/feature identifiers cannot be empty")
    key = "counts" if "counts" in adata.layers else None
    matrix = adata.layers[key] if key else adata.X
    if matrix is None and require_counts:
        raise ValueError("count matrix is missing")
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(values).all():
        raise ValueError("expression matrix contains NaN or infinite values")
    if (values < 0).any():
        raise ValueError("expression matrix must be nonnegative")
    if require_counts and not np.allclose(values, np.rint(values), rtol=0, atol=1e-6):
        raise ValueError(
            "X does not look like raw counts and layers['counts'] is absent; provide an integer raw-count matrix"
        )
    out = adata.copy()
    if "counts" not in out.layers:
        out.layers["counts"] = out.X.copy()
    if dataset_id is not None:
        out.obs[dataset_key] = str(dataset_id)
    elif dataset_key not in out.obs:
        out.obs[dataset_key] = "unspecified"
    if out.obs[dataset_key].isna().any():
        raise ValueError(f"{dataset_key} contains missing values")
    out.uns.setdefault("cellstate", {})["validation"] = {
        "n_cells": int(out.n_obs),
        "n_genes": int(out.n_vars),
        "counts_layer": "counts",
    }
    return out


def adata_checksum(adata: ad.AnnData) -> str:
    """Stable content hash of count matrix and ordered cell/feature identifiers."""
    import hashlib

    matrix = adata.layers["counts"] if "counts" in adata.layers else adata.X
    digest = hashlib.sha256()
    digest.update(str(matrix.shape).encode())
    if sparse.issparse(matrix):
        csr = matrix.tocsr(copy=True)
        csr.sort_indices()
        for arr in (csr.indptr, csr.indices, csr.data):
            digest.update(np.ascontiguousarray(arr).view(np.uint8))
    else:
        digest.update(np.ascontiguousarray(matrix).view(np.uint8))
    for ids in (adata.obs_names, adata.var_names):
        for item in ids.astype(str):
            digest.update(item.encode())
            digest.update(b"\0")
    return digest.hexdigest()


def load_dataset(
    path: str | Path,
    *,
    dataset_id: str | None = None,
    dataset_key: str = "dataset_id",
    source_url: str | None = None,
    license_terms: str | None = None,
) -> ad.AnnData:
    """Load local .h5ad or 10x directory/.h5 data without network access."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    if p.is_dir():
        if not (p / "matrix.mtx").exists() and not (p / "matrix.mtx.gz").exists():
            raise ValueError("directory must contain a 10x Matrix Market matrix")
        obj = sc.read_10x_mtx(p, var_names="gene_symbols", make_unique=True)
    elif p.suffix.lower() == ".h5ad":
        obj = ad.read_h5ad(p)
    elif p.suffix.lower() == ".h5":
        obj = sc.read_10x_h5(p)
    else:
        raise ValueError("supported inputs: .h5ad, 10x .h5, or 10x Matrix Market directory")
    if dataset_id is not None:
        ident = str(dataset_id)
    elif dataset_key in obj.obs:
        values = obj.obs[dataset_key].dropna().astype(str).unique()
        if len(values) > 1:
            raise ValueError(f"each input file must contain exactly one {dataset_key}")
        ident = values[0] if len(values) == 1 else p.stem
    else:
        ident = p.stem
    obj = validate_adata(obj, dataset_id=ident, dataset_key=dataset_key)
    obj.uns["cellstate"]["provenance"] = {
        "local_path": str(p),
        "sha256": sha256_path(p),
        "dataset_id": ident,
        "source_url": source_url,
        "retrieved_at": None,
        "license_or_terms": license_terms,
        "preprocessing_history": [],
    }
    logger.info("Loaded %s: %d cells x %d genes", p, obj.n_obs, obj.n_vars)
    return obj


def validate_metadata(adata: ad.AnnData, keys: Iterable[str]) -> dict[str, list[str]]:
    """Return missing metadata diagnostics, including columns and per-column nulls."""
    missing = []
    nulls = {}
    for key in keys:
        if key not in adata.obs:
            missing.append(key)
        elif adata.obs[key].isna().any():
            nulls[key] = adata.obs_names[adata.obs[key].isna()].astype(str).tolist()
    return {"missing_columns": missing, "rows_with_missing_values": nulls}
