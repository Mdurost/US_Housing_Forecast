"""TimeSeriesDataSet / dataloader construction."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from torch.utils.data import DataLoader

from housing_forecast.config import Config
from housing_forecast.features import FeatureSpec


def build_training_dataset(df: pd.DataFrame, spec: FeatureSpec, config: Config) -> TimeSeriesDataSet:
    ds = config.dataset
    return TimeSeriesDataSet(
        df[df["time_idx"] <= spec.training_cutoff],
        time_idx="time_idx",
        target=spec.target,
        group_ids=["state"],
        min_encoder_length=ds.min_encoder_length,
        max_encoder_length=ds.max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=ds.max_prediction_length,
        static_categoricals=spec.static_categoricals,
        time_varying_known_categoricals=spec.time_varying_known_categoricals,
        time_varying_known_reals=spec.time_varying_known_reals,
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=spec.time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(groups=["state"], transformation="log1p"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )


def build_validation_dataset(
    parameters: TimeSeriesDataSet | dict[str, Any], df: pd.DataFrame
) -> TimeSeriesDataSet:
    """Hold-out dataset: the last ``max_prediction_length`` months of each state.

    ``parameters`` is either the training dataset or a model's saved ``dataset_parameters``.
    """
    if isinstance(parameters, TimeSeriesDataSet):
        return TimeSeriesDataSet.from_dataset(parameters, df, predict=True, stop_randomization=True)
    return TimeSeriesDataSet.from_parameters(parameters, df, predict=True, stop_randomization=True)


def train_dataloader(dataset: TimeSeriesDataSet, config: Config) -> DataLoader:
    return dataset.to_dataloader(
        train=True, batch_size=config.dataset.batch_size, num_workers=config.dataset.num_workers
    )


def val_dataloader(dataset: TimeSeriesDataSet, config: Config) -> DataLoader:
    return dataset.to_dataloader(
        train=False,
        batch_size=config.dataset.batch_size * config.dataset.val_batch_size_multiplier,
        num_workers=config.dataset.num_workers,
    )
