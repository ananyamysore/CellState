"""Write a deterministic synthetic .h5ad and ground-truth metadata table."""

from pathlib import Path

from cellstate.synthetic import generate_synthetic_adata

out = Path("example-data")
out.mkdir(exist_ok=True)
adata = generate_synthetic_adata(cells_per_sample=24, n_genes=120, n_donors=4, seed=123)
adata.write_h5ad(out / "synthetic_control_treated.h5ad")
adata.obs.to_csv(out / "synthetic_cell_metadata.csv")
print(f"Wrote synthetic-only data to {out}")
