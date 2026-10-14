"""Generate and analyze synthetic data without downloading any files."""

from dataclasses import replace
from pathlib import Path

from cellstate.config import load_config
from cellstate.pipeline import run_pipeline
from cellstate.synthetic import generate_synthetic_adata

Path("example-results").mkdir(exist_ok=True)
adata = generate_synthetic_adata(cells_per_sample=24, n_genes=120, n_donors=4, seed=123)
config = load_config(Path(__file__).parents[1] / "configs" / "synthetic.yml")
config = replace(config, output_dir="example-results")
result = run_pipeline([adata], config, run_combat=True, run_sensitivity=True)
print(f"{result['status']}: outputs in example-results")
