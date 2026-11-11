"""Hyperparameter search: Lightning LR finder followed by an Optuna study."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import torch
import torch.serialization
import yaml

from housing_forecast.config import Config
from housing_forecast.dataset import (
    build_training_dataset,
    build_validation_dataset,
    train_dataloader,
    val_dataloader,
)
from housing_forecast.features import fit_features
from housing_forecast.modeling import build_model, build_trainer, seed_everything

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _trusted_checkpoint_loading() -> Iterator[None]:
    """Let Lightning/pytorch-forecasting reload their own trial checkpoints.

    Their checkpoints pickle dataset parameters, which ``torch.load(weights_only=True)``
    rejects. Only used around tuning code that loads checkpoints it just wrote.
    """
    original_load = torch.load
    original_serialization_load = torch.serialization.load

    def _load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = _load
    torch.serialization.load = _load
    try:
        yield
    finally:
        torch.load = original_load
        torch.serialization.load = original_serialization_load


def _sqlite_storage(storage: str, config: Config) -> str:
    """Resolve a relative ``sqlite:///`` path against the project root."""
    prefix = "sqlite:///"
    if not storage.startswith(prefix):
        return storage
    path = Path(storage[len(prefix):])
    if not path.is_absolute():
        path = config.resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return prefix + path.as_posix()


def find_learning_rate(raw: pd.DataFrame, config: Config) -> float:
    from lightning.pytorch.tuner import Tuner

    seed_everything(config.seed)
    frame, spec = fit_features(raw, config)
    training = build_training_dataset(frame, spec, config)
    validation = build_validation_dataset(training, frame)
    trainer, _ = build_trainer(config)
    model = build_model(training, config)
    with _trusted_checkpoint_loading():
        result = Tuner(trainer).lr_find(
            model,
            train_dataloaders=train_dataloader(training, config),
            val_dataloaders=val_dataloader(validation, config),
            min_lr=config.tuning.lr_finder_min_lr,
            max_lr=config.tuning.lr_finder_max_lr,
        )
    suggestion = result.suggestion()
    if suggestion is None:
        raise RuntimeError("Learning-rate finder did not return a suggestion.")
    logger.info("Suggested learning rate: %g", suggestion)
    return float(suggestion)


def tune(raw: pd.DataFrame, config: Config, learning_rate: float | None = None) -> dict:
    """Run the Optuna study and return the best parameters (also written to outputs/tuning)."""
    import optuna
    from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
        optimize_hyperparameters,
    )

    t = config.tuning
    if learning_rate is None:
        learning_rate = find_learning_rate(raw, config)

    seed_everything(config.seed)
    frame, spec = fit_features(raw, config)
    training = build_training_dataset(frame, spec, config)
    validation = build_validation_dataset(training, frame)

    sampler = optuna.samplers.TPESampler(seed=config.seed, multivariate=True)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=t.pruner_min_resource,
        max_resource=t.max_epochs,
        reduction_factor=t.pruner_reduction_factor,
    )
    study = optuna.create_study(
        study_name=t.study_name,
        storage=_sqlite_storage(t.storage, config),
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    with _trusted_checkpoint_loading():
        study = optimize_hyperparameters(
            train_dataloader(training, config),
            val_dataloader(validation, config),
            model_path=str(config.resolve(t.trial_dir)),
            study=study,
            n_trials=t.n_trials,
            timeout=t.timeout_hours * 60 * 60,
            max_epochs=t.max_epochs,
            gradient_clip_val_range=tuple(t.gradient_clip_val_range),
            hidden_size_range=tuple(t.hidden_size_range),
            hidden_continuous_size_range=tuple(t.hidden_continuous_size_range),
            attention_head_size_range=tuple(t.attention_head_size_range),
            learning_rate_range=(learning_rate / t.lr_range_factor, learning_rate * t.lr_range_factor),
            dropout_range=tuple(t.dropout_range),
            trainer_kwargs={
                "limit_train_batches": 1.0,
                "deterministic": "warn",
                "enable_progress_bar": False,
            },
            log_dir=str(config.resolve(config.paths.log_dir)),
            reduce_on_plateau_patience=config.model.reduce_on_plateau_patience,
            use_learning_rate_finder=False,
            pruner=pruner,
            verbose=1,
        )

    best = {"best_value": float(study.best_value), "params": study.best_trial.params}
    output = config.resolve(config.paths.output_dir) / "tuning" / f"{t.study_name}_best_params.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(best, sort_keys=False), encoding="utf-8")
    logger.info("Best validation loss %.4f | params written to %s", best["best_value"], output)
    return best
