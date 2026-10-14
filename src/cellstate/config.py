"""Validated configuration objects and YAML loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class QCConfig:
    min_genes: int = 200
    max_genes: int | None = None
    min_counts: int = 500
    max_counts: int | None = None
    max_mito_percent: float | None = 20.0
    mito_prefixes: tuple[str, ...] = ("MT-", "mt-")
    doublet_method: str = "none"  # none or scrublet

    def __post_init__(self) -> None:
        if self.min_genes < 0 or self.min_counts < 0:
            raise ValueError("minimum QC thresholds must be nonnegative")
        if self.max_genes is not None and self.max_genes < self.min_genes:
            raise ValueError("max_genes must be >= min_genes")
        if self.max_counts is not None and self.max_counts < self.min_counts:
            raise ValueError("max_counts must be >= min_counts")
        if self.max_mito_percent is not None and not 0 <= self.max_mito_percent <= 100:
            raise ValueError("max_mito_percent must be in [0,100]")
        if self.doublet_method not in {"none", "scrublet"}:
            raise ValueError("doublet_method must be none or scrublet")


@dataclass(frozen=True)
class PreprocessConfig:
    target_sum: float = 10_000.0
    n_top_genes: int = 2_000
    flavor: str = "seurat"
    scale: bool = False
    max_value: float = 10.0

    def __post_init__(self) -> None:
        if self.target_sum <= 0 or self.n_top_genes < 1:
            raise ValueError("target_sum and n_top_genes must be positive")
        if self.flavor not in {"seurat", "cell_ranger", "seurat_v3"}:
            raise ValueError("unsupported HVG flavor")
        if self.scale and self.max_value <= 0:
            raise ValueError("max_value must be positive")


@dataclass(frozen=True)
class EmbeddingConfig:
    n_pcs: int = 30
    n_neighbors: int = 15
    min_dist: float = 0.3
    metric: str = "euclidean"

    def __post_init__(self) -> None:
        if self.n_pcs < 2 or self.n_neighbors < 2 or self.min_dist < 0:
            raise ValueError("invalid embedding parameters")


@dataclass(frozen=True)
class ClusterConfig:
    resolution: float = 0.8
    key: str = "leiden"

    def __post_init__(self) -> None:
        if self.resolution <= 0 or not self.key:
            raise ValueError("resolution and cluster key must be positive/nonempty")


@dataclass(frozen=True)
class SensitivityConfig:
    min_genes: tuple[int, ...] = (200, 300)
    n_top_genes: tuple[int, ...] = (1500, 2000)
    resolutions: tuple[float, ...] = (0.6, 0.8)
    seeds: tuple[int, ...] = (0,)
    combat_options: tuple[bool, ...] = (False,)
    max_runs: int = 12

    def __post_init__(self) -> None:
        if (
            not self.min_genes
            or not self.n_top_genes
            or not self.resolutions
            or not self.seeds
            or not self.combat_options
        ):
            raise ValueError("sensitivity parameter grids cannot be empty")
        if (
            min(self.min_genes) < 0
            or min(self.n_top_genes) < 1
            or min(self.resolutions) <= 0
            or min(self.seeds) < 0
            or self.max_runs < 1
        ):
            raise ValueError("invalid sensitivity grid values")


@dataclass(frozen=True)
class CellStateConfig:
    """Run options and metadata field names; defaults are intentionally explicit."""

    dataset_key: str = "dataset_id"
    sample_key: str = "sample_id"
    donor_key: str = "donor_id"
    condition_key: str = "condition"
    batch_key: str = "batch"
    cell_type_key: str = "cell_type"
    seed: int = 0
    qc: QCConfig = field(default_factory=QCConfig)
    preprocessing: PreprocessConfig = field(default_factory=PreprocessConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    clustering: ClusterConfig = field(default_factory=ClusterConfig)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    output_dir: str = "cellstate-results"

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        for name in (
            self.dataset_key,
            self.sample_key,
            self.donor_key,
            self.condition_key,
            self.batch_key,
            self.cell_type_key,
        ):
            if not name:
                raise ValueError("metadata keys must be nonempty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> CellStateConfig:
    """Load nested YAML into validated config dataclasses; reject unknown keys."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    known = {
        "dataset_key",
        "sample_key",
        "donor_key",
        "condition_key",
        "batch_key",
        "cell_type_key",
        "seed",
        "qc",
        "preprocessing",
        "embedding",
        "clustering",
        "sensitivity",
        "output_dir",
    }
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")

    def section(key: str, cls: type) -> Any:
        value = raw.get(key, {})
        if not isinstance(value, dict):
            raise ValueError(f"{key} must be a mapping")
        allowed = cls.__dataclass_fields__.keys()
        extra = set(value) - allowed
        if extra:
            raise ValueError(f"unknown {key} keys: {sorted(extra)}")
        if key == "qc" and "mito_prefixes" in value:
            value = {**value, "mito_prefixes": tuple(value["mito_prefixes"])}
        return cls(**value)

    sensitivity_raw = raw.get("sensitivity", {})
    if not isinstance(sensitivity_raw, dict):
        raise ValueError("sensitivity must be a mapping")
    extra = set(sensitivity_raw) - SensitivityConfig.__dataclass_fields__.keys()
    if extra:
        raise ValueError(f"unknown sensitivity keys: {sorted(extra)}")
    sensitivity_raw = {
        k: tuple(v) if isinstance(v, list) else v for k, v in sensitivity_raw.items()
    }
    return CellStateConfig(
        **{
            k: v
            for k, v in raw.items()
            if k not in {"qc", "preprocessing", "embedding", "clustering", "sensitivity"}
        },
        qc=section("qc", QCConfig),
        preprocessing=section("preprocessing", PreprocessConfig),
        embedding=section("embedding", EmbeddingConfig),
        clustering=section("clustering", ClusterConfig),
        sensitivity=SensitivityConfig(**sensitivity_raw),
    )
