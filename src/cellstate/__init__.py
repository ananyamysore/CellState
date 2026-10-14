"""CellState: reproducible cross-dataset single-cell analysis."""

from .config import CellStateConfig, load_config
from .io import load_dataset, validate_adata
from .pipeline import run_pipeline
from .synthetic import generate_synthetic_adata

__version__ = "0.1.0"
__all__ = [
    "CellStateConfig",
    "generate_synthetic_adata",
    "load_config",
    "load_dataset",
    "run_pipeline",
    "validate_adata",
]
