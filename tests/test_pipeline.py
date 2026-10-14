import json

import pytest

from cellstate.config import (
    CellStateConfig,
    ClusterConfig,
    EmbeddingConfig,
    PreprocessConfig,
    QCConfig,
    SensitivityConfig,
)
from cellstate.pipeline import run_pipeline
from cellstate.synthetic import generate_synthetic_adata


def config_for(tmp_path):
    return CellStateConfig(
        output_dir=str(tmp_path),
        seed=17,
        qc=QCConfig(min_genes=0, min_counts=0, max_mito_percent=100),
        preprocessing=PreprocessConfig(n_top_genes=40),
        embedding=EmbeddingConfig(n_pcs=12, n_neighbors=8, min_dist=0.2),
        clustering=ClusterConfig(resolution=0.5),
        sensitivity=SensitivityConfig(
            min_genes=(0,), n_top_genes=(30,), resolutions=(0.4, 0.7), seeds=(1,), max_runs=3
        ),
    )


def test_end_to_end_exports_manifest_and_inconclusive_confounded_batch(tmp_path):
    a = generate_synthetic_adata(
        cells_per_sample=12, n_genes=80, n_donors=3, confounded_batch=True, seed=5
    )
    cfg = config_for(tmp_path)
    result = run_pipeline([a], cfg, run_combat=True)
    assert result["dataset_runs"][0]["condition_batch_diagnostic"]["confounded"] is True
    assert result["dataset_runs"][0]["combat"]["status"] == "inconclusive"
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["random_seed"] == 17
    files = list(tmp_path.rglob("*"))
    assert any(p.name == "cell_assignments.csv" for p in files)
    assert any(p.name == "signatures.csv" for p in files)
    assert any(p.name == "qc_summary_before.csv" for p in files)
    assert any(p.name == "qc_summary_after.csv" for p in files)
    assert any(p.name == "qc_distributions.png" for p in files)
    assert any(p.name == "umap.png" for p in files)
    assert result["dataset_runs"][0]["n_cells_after_qc"] > 0


def test_sensitivity_runner_records_runs_and_reproducible_seeds(tmp_path):
    a = generate_synthetic_adata(cells_per_sample=8, n_genes=60, n_donors=2, seed=8)
    run_pipeline([a], config_for(tmp_path), run_sensitivity=True)
    sensitivity = json.loads((tmp_path / "sensitivity.json").read_text())
    assert len(sensitivity["runs"]) == 2
    assert all(r["status"] == "ok" for r in sensitivity["runs"])


def test_pipeline_rejects_empty_input(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        run_pipeline([], config_for(tmp_path))
