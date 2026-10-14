"""Small deterministic synthetic AnnData for tests and examples (not biological evidence)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def generate_synthetic_adata(
    *,
    cells_per_sample: int = 24,
    n_genes: int = 120,
    n_donors: int = 4,
    cell_types: tuple[str, ...] = ("TypeA", "TypeB"),
    paired: bool = True,
    confounded_batch: bool = False,
    seed: int = 0,
) -> ad.AnnData:
    """Generate known cell-type and condition signals plus controllable batch effects.

    Each cell type has marker genes; treated cells have another known marker set.
    ``confounded_batch=True`` makes batch perfectly identify condition to exercise
    the pipeline's non-identifiability warning. This simulation is deliberately
    simple and must not be interpreted as a real biological dataset.
    """
    if (
        cells_per_sample < 2
        or n_donors < 2
        or not cell_types
        or any(not isinstance(name, str) or not name.strip() for name in cell_types)
        or len(set(cell_types)) != len(cell_types)
    ):
        raise ValueError("need >=2 cells/sample, >=2 donors, and unique nonempty cell types")
    condition_start = 6 * len(cell_types) + 4
    batch_start = condition_start + 8
    if n_genes < batch_start + 10:
        raise ValueError(
            "n_genes is too small for disjoint cell-type, condition, batch, and mitochondrial features"
        )
    rng = np.random.default_rng(seed)
    genes = [f"gene_{i:04d}" for i in range(n_genes)]
    if n_genes > 40:
        genes[-2:] = ["MT-ND1", "MT-CO1"]
    conditions = ["control", "treated"]
    rows = []
    mats = []
    donor_conditions = (
        [(f"donor_{d}", cond) for d in range(n_donors) for cond in conditions]
        if paired
        else [(f"donor_{d}_{cond}", cond) for cond in conditions for d in range(n_donors)]
    )
    for donor, cond in donor_conditions:
        cond_i = conditions.index(cond)
        batch = f"batch_{cond_i}" if confounded_batch else f"batch_{int(donor.split('_')[1]) % 2}"
        sample = f"{donor}_{cond}"
        for cell_i in range(cells_per_sample):
            ctype = cell_types[cell_i % len(cell_types)]
            cidx = cell_types.index(ctype)
            rate = np.full(n_genes, 0.25, dtype=float)
            marker_start = cidx * 6
            rate[marker_start : marker_start + 6] = 3.5
            if cond == "treated":
                rate[condition_start : condition_start + 8] *= 3.0
            if batch == "batch_1":
                rate[batch_start : batch_start + 8] *= 3.0
            counts = rng.poisson(rate)
            if counts.sum() == 0:
                counts[0] = 1
            mats.append(counts.astype(np.int32))
            rows.append(
                {
                    "dataset_id": "synthetic",
                    "sample_id": sample,
                    "donor_id": donor,
                    "condition": cond,
                    "batch": batch,
                    "cell_type": ctype,
                    "synthetic_cell_type": ctype,
                }
            )
    obs = pd.DataFrame(rows, index=[f"cell_{i:05d}" for i in range(len(rows))])
    var = pd.DataFrame(index=genes)
    var["mt"] = var.index.str.startswith("MT-")
    matrix = sparse.csr_matrix(np.stack(mats))
    result = ad.AnnData(X=matrix.copy(), obs=obs, var=var)
    result.layers["counts"] = matrix.copy()
    result.uns["cellstate"] = {
        "synthetic": True,
        "seed": seed,
        "known_cell_type_markers": {
            name: [genes[i] for i in range(k * 6, k * 6 + 6)] for k, name in enumerate(cell_types)
        },
        "known_condition_markers": genes[condition_start : condition_start + 8],
        "confounded_batch": confounded_batch,
        "warning": "Synthetic observations are for software checks and are not biological findings.",
    }
    return result
