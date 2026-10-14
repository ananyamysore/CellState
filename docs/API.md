# API reference

- `CellStateConfig`, `QCConfig`, `PreprocessConfig`, `EmbeddingConfig`, `ClusterConfig`, and `SensitivityConfig` define validated runtime options. `load_config(path)` reads YAML.
- `load_dataset(path, dataset_id=None, dataset_key='dataset_id', source_url=None, license_terms=None)` reads local h5ad, 10x H5, or 10x Matrix Market and records provenance/checksum. `validate_adata(adata, ...)` validates and retains raw counts.
- `calculate_qc(adata, mito_prefixes=...)` returns annotated QC metrics. `filter_qc(adata, config)` returns filtered AnnData and sequential decision accounting.
- `preprocess(adata, config, batch_key=None)` normalizes, logs, selects HVGs, and optionally scales while preserving `layers['counts']` and `.raw`.
- `compute_embedding(adata, config, seed=0, use_rep=None)` builds PCA/neighbors/UMAP. `cluster_cells(adata, config, seed=0, algorithm='leiden')` applies graph clustering.
- `pseudobulk_signatures(adata, condition_key, cell_type_key, sample_key, donor_key, group_a, group_b, min_replicates)` returns a gene table and an explicit status/replicate summary.
- `diagnose_condition_batch_confounding(obs, condition_key, batch_key)` detects perfect one-to-one confounding. `combat_representation(adata, ...)` optionally stores corrected log-expression and PCA separately.
- `signature_agreement(...)`, `cluster_stability(...)`, and `composition_differences(...)` provide documented agreement summaries. `run_sensitivity(...)` runs a bounded parameter grid.
- `run_pipeline(inputs, config, run_combat=False, run_sensitivity=False)` executes local analysis and writes reports.
- `generate_synthetic_adata(...)` produces deterministic toy AnnData with known cell type/condition markers and a deliberately confounded batch option.

See Python docstrings and `docs/methods.md` for output definitions and interpretation.
