# Methods, assumptions, and limits

## Data and observations

CellState expects a cell-by-gene AnnData matrix of raw nonnegative integer counts. Prefer `layers['counts']`; otherwise the input `X` must be integer-like and will be copied. Transformed `X` without raw counts is rejected. Feature IDs must be unique. Each input corresponds to one dataset. When possible, include `dataset_id`, `sample_id`, `donor_id`, `condition`, `batch`, `tissue`, and an independently curated `cell_type`. Missing fields and null values are surfaced in the manifest.

## Replicate-aware analysis

Condition-level expression is aggregated at donor-condition-cell-type (or sample-condition-cell-type when donor IDs are absent). Paired donors shared across conditions use a paired t-test; otherwise Welch's t-test is performed on log2 CPM aggregates. BH controls the expected false discovery proportion over tested genes within the combined output table only after all rows are evaluated? The implementation currently adjusts separately within each cell type. It is a transparent exploratory approximation and does not model count mean/dispersion, covariates, paired sample structure beyond donor pairing, or random effects. For confirmatory inference use a validated pseudobulk count model and a design appropriate to the experiment.

Cell-type composition is computed per sample and compared at donor level if donor identifiers are present. Standard t-tests ignore the joint sum-to-one constraint; use a compositional model for confirmatory inference. Cells from the same sample are not independent replicates.

## Integration and confounding

Each dataset is QC-filtered, normalized, embedded, and clustered separately. Scanpy ComBat is optional and applied to normalized log-expression. Its corrected layer and PCA representation are separate from raw counts and normalized `X`. Perfect one-to-one condition/batch assignment is not identifiable and is flagged; ComBat is skipped. For nonperfect designs, correction can still remove biological signal. Assess correction with donor-held-out validation and signature/marker preservation; UMAP mixing alone is insufficient.

Cross-dataset signature comparison requires the same explicit cell-type annotation and condition contrast. No unsupervised cluster-number matching is attempted. Jaccard of top absolute effect sizes ignores ordering; Spearman compares signed effect-size ranks. These statistics are descriptive and depend on gene identifier harmonization, annotation quality, and comparable study designs.

## QC, embeddings, and stability

QC rules are sequential and cell losses are recorded by rule. Mitochondrial fraction is computed only when `var['mt']` or configured gene prefixes identify mitochondrial features. Scrublet is optional and its prediction is only a doublet-screening signal.

Library-size normalization and log1p are standard exploratory transforms. HVG selection, PCA, graph construction, Leiden/Louvain and UMAP parameters are recorded. UMAP is for visualization and has no inferential meaning. Adjusted Rand index compares cluster partitions on the cell-ID intersection and is invariant to label permutations, but does not validate biological identity. Sensitivity analysis measures dependence on selected parameters only; it does not capture all analytical uncertainty.

## Synthetic data and reproducibility

The generator uses Poisson counts, simple marker multipliers, and deterministic random seeds. It includes paired donors, two conditions, two cell types, technical batch signals, and an option where condition and batch are perfectly confounded. These simulated data are intentionally limited; they test software behavior and warnings, not biological accuracy. No real dataset is downloaded or bundled.

The run manifest records software versions, parameters, seed, source path/checksum for local files, or a count-matrix content hash for in-memory inputs. Checksums establish file identity, not data rights. Users should verify dataset terms and record accession, source URL, retrieval date, and license information when they curate public data.
