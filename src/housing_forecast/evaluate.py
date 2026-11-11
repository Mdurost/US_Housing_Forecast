"""Hold-out evaluation of the TFT against persistence and seasonal-naive baselines."""

from __future__ import annotations

import logging

import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer

from housing_forecast.config import Config
from housing_forecast.dataset import build_validation_dataset, val_dataloader
from housing_forecast.modeling import resolve_accelerator

logger = logging.getLogger(__name__)

SEASON_LENGTH = 12


def forecast_metrics(
    forecast: torch.Tensor,
    actual: torch.Tensor,
    scales: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, float]:
    """MASE (seasonal scale), MAE, RMSE, and MAPE (%) over valid horizon steps."""
    errors = forecast - actual
    absolute = errors.abs()
    return {
        "MASE": (absolute / scales)[valid_mask].mean().item(),
        "MAE": absolute[valid_mask].mean().item(),
        "RMSE": torch.sqrt(errors.square()[valid_mask].mean()).item(),
        "MAPE": (absolute[valid_mask] / actual[valid_mask].abs().clamp_min(1e-6)).mean().item() * 100,
    }


def naive_baselines(
    history: pd.Series,
    prediction_index: pd.DataFrame,
    decoder_lengths: torch.Tensor,
    like: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Persistence (last value) and 12-month seasonal-naive forecasts.

    ``history`` is the target indexed by (state, time_idx).
    """
    persistence = torch.full_like(like, torch.nan)
    seasonal = torch.full_like(like, torch.nan)
    for row, prediction_row in prediction_index.iterrows():
        state = str(prediction_row["state"])
        first = int(prediction_row["time_idx"])
        horizon = int(decoder_lengths[row].item())
        persistence[row, :horizon] = float(history.loc[(state, first - 1)])
        for step in range(horizon):
            seasonal[row, step] = float(history.loc[(state, first + step - SEASON_LENGTH)])
    return persistence, seasonal


def seasonal_scales(frame: pd.DataFrame, target: str, training_cutoff: int) -> pd.Series:
    """Per-state mean absolute 12-month difference, from the training partition only."""
    training = frame.loc[frame["time_idx"] <= training_cutoff, ["state", "time_idx", target]]
    training = training.sort_values(["state", "time_idx"])
    return training.groupby("state")[target].apply(lambda v: v.diff(SEASON_LENGTH).abs().mean())


def evaluate(model: TemporalFusionTransformer, frame: pd.DataFrame, target: str, config: Config) -> pd.DataFrame:
    validation = build_validation_dataset(model.dataset_parameters, frame)
    predictions = model.predict(
        val_dataloader(validation, config),
        return_y=True,
        return_index=True,
        return_decoder_lengths=True,
        trainer_kwargs={"accelerator": resolve_accelerator(config.trainer.accelerator), "logger": False},
    )

    # GroupNormalizer reverses log1p, so predictions and targets are in dollars.
    model_forecast = predictions.output.detach().cpu()
    actual = predictions.y[0].cpu()
    decoder_lengths = predictions.decoder_lengths.cpu()
    prediction_index = predictions.index.reset_index(drop=True)

    history = frame[["state", "time_idx", target]].set_index(["state", "time_idx"])[target].sort_index()
    persistence, seasonal = naive_baselines(history, prediction_index, decoder_lengths, model_forecast)

    training_cutoff = int(frame["time_idx"].max() - config.dataset.max_prediction_length)
    scale = seasonal_scales(frame, target, training_cutoff)
    scales = torch.tensor(
        prediction_index["state"].astype(str).map(scale).to_numpy(),
        dtype=model_forecast.dtype,
    ).unsqueeze(1)
    valid_mask = torch.arange(model_forecast.size(1)).unsqueeze(0) < decoder_lengths.unsqueeze(1)

    if torch.isnan(scales).any() or (scales <= 0).any():
        raise ValueError("Every validation state must have a positive seasonal MASE scale.")
    for name, baseline in [("Persistence", persistence), ("Seasonal-naive", seasonal)]:
        if torch.isnan(baseline[valid_mask]).any():
            raise ValueError(f"{name} baseline contains missing validation forecasts.")

    rows = [
        {"model": "TFT", **forecast_metrics(model_forecast, actual, scales, valid_mask)},
        {"model": "Persistence", **forecast_metrics(persistence, actual, scales, valid_mask)},
        {"model": "Seasonal naive (12-month)", **forecast_metrics(seasonal, actual, scales, valid_mask)},
    ]
    return pd.DataFrame(rows)
