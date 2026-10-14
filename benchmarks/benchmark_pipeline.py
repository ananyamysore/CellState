"""Runtime harness on a generated, explicitly synthetic AnnData dataset."""

from __future__ import annotations

import platform
import time

import anndata
import numpy
import scanpy
import scipy
import sklearn

from cellstate.config import (
    CellStateConfig,
    ClusterConfig,
    EmbeddingConfig,
    PreprocessConfig,
    QCConfig,
)
from cellstate.pipeline import run_pipeline
from cellstate.synthetic import generate_synthetic_adata

if __name__ == "__main__":
    data = generate_synthetic_adata(cells_per_sample=100, n_genes=1000, n_donors=8, seed=23)
    cfg = CellStateConfig(
        seed=23,
        output_dir="benchmark-output",
        qc=QCConfig(min_genes=0, min_counts=0, max_mito_percent=100),
        preprocessing=PreprocessConfig(n_top_genes=500),
        embedding=EmbeddingConfig(n_pcs=30, n_neighbors=15),
        clustering=ClusterConfig(resolution=0.6),
    )
    start = time.perf_counter()
    result = run_pipeline([data], cfg)
    elapsed = time.perf_counter() - start
    print(
        {
            "elapsed_seconds": elapsed,
            "synthetic_shape": [data.n_obs, data.n_vars],
            "seed": 23,
            "python": platform.python_version(),
            "scanpy": scanpy.__version__,
            "anndata": anndata.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "output_status": result["status"],
        }
    )
