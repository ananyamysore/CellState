# CellState

CellState is a local, CPU-only Python package for reproducible single-cell RNA-seq analysis across conditions and independent datasets. It emphasizes sample-aware inference, data provenance, parameter sensitivity, and explicit limitations. It does not download data during installation or tests and does not include real biological datasets.

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[test,lint]'
```

Install optional Scrublet support with `python -m pip install '.[doublets]'`. The Seurat v3 HVG flavor additionally requires `python -m pip install '.[seurat-v3]'`.

## Quick start

Generate and analyze an offline synthetic dataset:

```bash
python examples/run_synthetic.py
```

Generate a portable synthetic `.h5ad` first:

```bash
python examples/generate_synthetic.py
cellstate run --config configs/synthetic.yml --input example-data/synthetic_control_treated.h5ad
```

Analyze local datasets with config:

```bash
cellstate run --config configs/real_data.yml --input data/study_a.h5ad --input data/study_b.h5ad --combat --sensitivity
```

The CLI accepts `.h5ad`, 10x `.h5`, and 10x Matrix Market directories. Inputs must contain raw integer counts in `layers['counts']` or in `X`; normalized/log expression in `X` without a counts layer is rejected. A single input must represent one dataset. Configurable observation keys track dataset, sample, donor, condition, batch, and cell-type annotations.

## Output

The run directory contains `run_manifest.json`, `results.json`, and a `datasets/<dataset_id>/` directory for every input. Dataset outputs include per-cell QC before/after tables, per-sample QC summaries and distributions, sequential filter accounting, sample-level cell-type proportions and condition comparisons, donor-aware signature tables, cell assignments, and UMAP figures. Optional ComBat runs save a separate corrected representation and report its cluster partition changes alongside the original result. Corrected expression does not replace raw counts or normalized expression.

The manifest records configuration, versions, seed, local provenance/checksum where available, cell/gene counts, QC decisions, warnings, output paths, and elapsed time. No large input files are copied into the output or repository.

## Scientific methods and interpretation

- **QC:** detected genes, total counts, and mitochondrial fraction when gene names or `var['mt']` identify mitochondrial features. Optional Scrublet is a screening tool, not ground truth.
- **Preprocessing:** raw counts are retained in a layer; `X` becomes library-size normalized and log1p transformed. HVGs are selected with Scanpy; the full log-expression feature set is preserved in `.raw` before feature filtering.
- **Embedding/clustering:** PCA, a k-nearest-neighbor graph, UMAP, and Leiden by default. Random seeds and algorithm parameters are recorded. Clusters are exploratory labels, not automatically cell types.
- **Differential signatures:** counts are aggregated by donor (preferred) or sample and cell type. Paired donors use a paired t-test; otherwise independent units use Welch's t-test on log2 CPM, followed by Benjamini–Hochberg correction. This transparent screening method is not a negative-binomial pseudobulk model and does not adjust for multiple covariates. Use a validated count-model design for confirmatory analysis.
- **Composition:** proportions are calculated within samples and compared at donor/sample level; cells are never treated as replicates. The basic t-tests do not model the joint compositional constraint and are descriptive screening only.
- **Cross-dataset agreement:** signature overlap and ranked effect-size Spearman agreement are calculated only for explicitly matching cell-type labels and matching condition contrasts. Matching labels do not prove biological equivalence.
- **Batch correction:** ComBat is optional and operates on normalized log-expression; corrected values and PCA coordinates are stored separately. Perfect one-to-one batch/condition confounding is flagged inconclusive and correction is refused. UMAP batch mixing alone is not a validation criterion.
- **Sensitivity:** configurable QC thresholds, HVG counts, clustering resolutions, and seeds are run on the same input; cluster comparisons use adjusted Rand index on shared cells. Stability is algorithmic reproducibility, not biological validity.

No causal inference is made. Observational expression data cannot by itself establish causation. If donor, condition, sample, tissue, or batch annotations are absent or confounded, the report says what cannot be identified. Synthetic data only verify software behavior under known toy conditions and are not evidence of performance on real studies.

## Configuration

See `configs/synthetic.yml` for a complete YAML example. QC thresholds should be selected from the study's assay, tissue, chemistry, and per-sample QC distributions. Unknown YAML keys are rejected. Sensitivity grids are capped by `max_runs` to avoid accidental excessive computation.

## Development commands

```bash
python -m pip install '.[test,lint]'
ruff check src tests examples benchmarks
pytest
python benchmarks/benchmark_pipeline.py
```

The benchmark prints machine/runtime versions and measured elapsed time for a generated synthetic workload. No benchmark result is precomputed or claimed here.

## Documentation

- [API reference](docs/API.md)
- [Methods, assumptions, and limitations](docs/methods.md)
- [Data and provenance guide](docs/data_and_provenance.md)

## License and citation

MIT licensed. See `CITATION.cff`. Cite the exact release/version, Scanpy/AnnData, and the statistical methods used in any resulting study.
