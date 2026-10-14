import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cellstate.config import (
    PreprocessConfig,
    QCConfig,
    load_config,
)
from cellstate.integration import diagnose_condition_batch_confounding
from cellstate.io import load_dataset, validate_adata
from cellstate.metrics import cluster_stability, composition_differences, signature_agreement
from cellstate.preprocessing import preprocess
from cellstate.qc import calculate_qc, filter_qc
from cellstate.signatures import pseudobulk_signatures
from cellstate.synthetic import generate_synthetic_adata


def small_adata():
    x = sparse.csr_matrix([[1, 2, 0, 1], [0, 1, 3, 1], [4, 0, 1, 2], [1, 1, 1, 1]], dtype=np.int32)
    obs = pd.DataFrame(
        {
            "dataset_id": ["d"] * 4,
            "sample_id": ["s1", "s1", "s2", "s2"],
            "condition": ["a", "a", "b", "b"],
        },
        index=[f"c{i}" for i in range(4)],
    )
    var = pd.DataFrame({"mt": [False, False, True, False]}, index=["g1", "g2", "MT-A", "g4"])
    return ad.AnnData(X=x, obs=obs, var=var)


def test_validation_rejects_empty_noninteger_and_invalid_counts():
    with pytest.raises(ValueError):
        validate_adata(ad.AnnData(X=np.empty((0, 2))))
    bad = ad.AnnData(X=np.array([[1.5, 0.2]]))
    with pytest.raises(ValueError, match="raw counts"):
        validate_adata(bad)
    negative = ad.AnnData(X=np.array([[1, -1]]))
    with pytest.raises(ValueError, match="nonnegative"):
        validate_adata(negative)
    nan = ad.AnnData(X=np.array([[1, np.nan]]))
    with pytest.raises(ValueError, match="NaN"):
        validate_adata(nan)


def test_load_h5ad_records_provenance_and_feature_ids(tmp_path):
    a = validate_adata(small_adata())
    path = tmp_path / "sample.h5ad"
    a.write_h5ad(path)
    loaded = load_dataset(path)
    assert loaded.uns["cellstate"]["provenance"]["sha256"]
    assert loaded.obs["dataset_id"].unique().tolist() == ["d"]


def test_qc_calculation_and_filter_accounting():
    a = calculate_qc(validate_adata(small_adata()))
    assert "pct_counts_mt" in a.obs
    filtered, account = filter_qc(a, QCConfig(min_genes=1, min_counts=1, max_mito_percent=100))
    assert filtered.n_obs == 4 and account["initial_cells"] == 4
    assert all(item["before"] - item["after"] == item["removed"] for item in account["filters"])


def test_preprocessing_preserves_counts_and_full_gene_raw():
    a = validate_adata(small_adata())
    out = preprocess(a, PreprocessConfig(n_top_genes=3))
    np.testing.assert_array_equal(out.layers["counts"].toarray(), a.layers["counts"].toarray())
    assert out.n_vars == 3 and out.raw.n_vars == a.n_vars
    assert np.all(np.asarray(out.X.sum(axis=1)).ravel() > 0)


def test_pseudobulk_signature_detects_known_condition_signal():
    a = generate_synthetic_adata(cells_per_sample=32, n_genes=80, n_donors=6, seed=12)
    table, info = pseudobulk_signatures(a, min_replicates=3)
    assert info["status"] == "ok" and info["unit_level"] == "donor"
    marker = table[(table.cell_type == "TypeA") & (table.gene == "gene_0018")].iloc[0]
    assert marker.log2fc > 0 and marker.padj < 0.05


def test_metadata_confounded_batch_is_flagged_and_unlabeled_is_inconclusive():
    a = generate_synthetic_adata(cells_per_sample=8, n_genes=60, n_donors=3, confounded_batch=True)
    report = diagnose_condition_batch_confounding(a.obs)
    assert report["status"] == "warning" and report["confounded"] is True
    _result, info = pseudobulk_signatures(a)
    assert info["status"] == "ok"
    no_donor = a.copy()
    no_donor.obs["donor_id"] = None
    no_donor.obs["sample_id"] = None
    _, inconclusive = pseudobulk_signatures(no_donor)
    assert inconclusive["status"] == "inconclusive"


def test_reproducibility_metrics_and_composition_unit_counts():
    assert cluster_stability([0, 0, 1, 1], [1, 1, 0, 0]) == pytest.approx(1)
    agreed = signature_agreement(["a", "b", "c"], [3, -2, 1], ["a", "b", "d"], [2, -3, 1], top_n=2)
    assert agreed["top_gene_jaccard"] == pytest.approx(1.0)
    a = generate_synthetic_adata(cells_per_sample=8, n_genes=60, n_donors=4)
    diffs = composition_differences(
        a.obs,
        sample_key="sample_id",
        donor_key="donor_id",
        condition_key="condition",
        cell_type_key="cell_type",
    )
    assert len(diffs) == 2 and all(d["n_a"] == 4 for d in diffs)


def test_config_yaml_rejects_unknown_fields(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text("qc:\n  min_genes: 12\nsensitivity:\n  seeds: [2, 3]\n")
    cfg = load_config(path)
    assert cfg.qc.min_genes == 12 and cfg.sensitivity.seeds == (2, 3)
    path.write_text("mystery: true\n")
    with pytest.raises(ValueError, match="unknown configuration"):
        load_config(path)
