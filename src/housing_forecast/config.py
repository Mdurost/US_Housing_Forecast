"""Typed configuration loaded from YAML with optional ``key.path=value`` overrides."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


@dataclass
class PathsConfig:
    raw_dir: Path = Path("data/raw")
    processed_data: Path = Path("data/processed/processed_data_pre_model.csv")
    model_checkpoint: Path = Path("models/tft_housing.ckpt")
    feature_spec: Path = Path("models/feature_spec.json")
    model_card: Path = Path("models/model_card.json")
    candidate_dir: Path = Path("models/candidate")
    archive_dir: Path = Path("models/archive")
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("lightning_logs")
    output_dir: Path = Path("outputs")


@dataclass
class IngestConfig:
    request_timeout_seconds: float = 120
    max_retries: int = 3
    economic_min_date: str = "1948-01-01"
    supply_min_date: str = "1960-01-01"
    mortgage_min_date: str = "2016-01-01"
    include_housing_starts: bool = False
    economic_month_fills: dict[str, str] = field(default_factory=dict)


@dataclass
class FeaturesConfig:
    target: str = "median_listing_price"
    min_date: str = "2018-01-01"
    corr_threshold: float = 0.90
    missing_indicator_threshold: float = 0.01
    identifier_columns: list[str] = field(
        default_factory=lambda: ["county_fips", "city", "county_name", "cbsa_code", "cbsa_title"]
    )
    excluded_unknown_reals: list[str] = field(
        default_factory=lambda: ["year", "time_idx", "quality_flag", "cbsa_code"]
    )


@dataclass
class DatasetConfig:
    max_encoder_length: int = 36
    min_encoder_length: int = 18
    max_prediction_length: int = 12
    batch_size: int = 128
    val_batch_size_multiplier: int = 10
    num_workers: int = 0


@dataclass
class ModelConfig:
    learning_rate: float = 0.0002337546936010406
    hidden_size: int = 70
    attention_head_size: int = 1
    dropout: float = 0.23357639741209468
    hidden_continuous_size: int = 48
    reduce_on_plateau_patience: int = 4
    log_interval: int = 10


@dataclass
class TrainerConfig:
    accelerator: str = "auto"
    max_epochs: int = 100
    limit_train_batches: float = 50
    gradient_clip_val: float = 0.15761934729549992
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4


@dataclass
class TuningConfig:
    study_name: str = "tft_housing_v2"
    storage: str = "sqlite:///data/studies/tft_optuna.db"
    trial_dir: Path = Path("data/studies/optuna_trials")
    n_trials: int = 100
    timeout_hours: float = 6
    max_epochs: int = 40
    pruner_min_resource: int = 5
    pruner_reduction_factor: int = 3
    lr_finder_min_lr: float = 1e-6
    lr_finder_max_lr: float = 10.0
    lr_range_factor: float = 5
    gradient_clip_val_range: tuple[float, float] = (0.01, 1.0)
    hidden_size_range: tuple[int, int] = (16, 320)
    hidden_continuous_size_range: tuple[int, int] = (16, 128)
    attention_head_size_range: tuple[int, int] = (2, 8)
    dropout_range: tuple[float, float] = (0.1, 0.4)


@dataclass
class ForecastConfig:
    quantiles_to_report: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])


@dataclass
class TrackingConfig:
    enabled: bool = True
    tracking_uri: str = "sqlite:///mlflow.db"
    artifact_location: Path = Path("mlruns")
    experiment_name: str = "us-housing-tft"
    registered_model_name: str = "us-housing-tft"
    champion_alias: str = "champion"


@dataclass
class PromotionConfig:
    metric: str = "MASE"
    baseline_model: str | None = "Seasonal naive (12-month)"
    min_improvement: float = 0.0


@dataclass
class Config:
    seed: int = 1
    paths: PathsConfig = field(default_factory=PathsConfig)
    sources: dict[str, str] = field(default_factory=dict)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    project_root: Path = field(default_factory=Path.cwd)

    def resolve(self, path: Path | str) -> Path:
        """Resolve a configured path against the project root."""
        path = Path(path)
        return path if path.is_absolute() else self.project_root / path

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(dataclasses.asdict(self))


_SECTIONS: dict[str, type] = {
    "paths": PathsConfig,
    "ingest": IngestConfig,
    "features": FeaturesConfig,
    "dataset": DatasetConfig,
    "model": ModelConfig,
    "trainer": TrainerConfig,
    "tuning": TuningConfig,
    "forecast": ForecastConfig,
    "tracking": TrackingConfig,
    "promotion": PromotionConfig,
}


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _build_section(cls: type, values: dict[str, Any]) -> Any:
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for name, value in values.items():
        default = getattr(cls(), name)
        if isinstance(default, Path):
            value = Path(value)
        elif isinstance(default, tuple):
            value = tuple(value)
        kwargs[name] = value
    return cls(**kwargs)


def apply_overrides(raw: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``section.key=value`` overrides; values are parsed as YAML scalars."""
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must look like key.path=value, got {override!r}")
        dotted, value = override.split("=", 1)
        keys = dotted.strip().split(".")
        node = raw
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = yaml.safe_load(value)
    return raw


def _find_project_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def load_config(path: Path | str | None = None, overrides: list[str] | None = None) -> Config:
    if path:
        config_path = Path(path)
    elif DEFAULT_CONFIG_PATH.is_file():
        config_path = DEFAULT_CONFIG_PATH
    else:
        config_path = Path.cwd() / "configs" / "default.yaml"
    with open(config_path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw = apply_overrides(raw, overrides)

    unknown = set(raw) - {"seed", "sources", "project_root", *_SECTIONS}
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")

    kwargs: dict[str, Any] = {
        name: _build_section(cls, raw.get(name) or {}) for name, cls in _SECTIONS.items()
    }
    kwargs["seed"] = int(raw.get("seed", 1))
    kwargs["sources"] = dict(raw.get("sources") or {})
    root = raw.get("project_root")
    kwargs["project_root"] = (
        Path(root) if root else _find_project_root(config_path.resolve().parent)
    )
    return Config(**kwargs)
