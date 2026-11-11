"""Generate 12-month state and national forecasts from a trained model."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer

from housing_forecast.config import Config
from housing_forecast.features import FeatureSpec, holiday_month_flags, time_index
from housing_forecast.modeling import resolve_accelerator

logger = logging.getLogger(__name__)


@dataclass
class Forecast:
    state: pd.DataFrame
    national: pd.DataFrame
    last_observed_date: pd.Timestamp


def build_future_frame(
    frame: pd.DataFrame, spec: FeatureSpec, encoder_length: int, horizon: int
) -> pd.DataFrame:
    """Encoder history plus ``horizon`` future months per state.

    Unknown covariates are carried forward from each state's last observed month; calendar
    features (``month``, ``is_holiday``, ``time_idx``) are computed for the future dates.
    """
    start_date = pd.Timestamp(spec.start_date)
    last_idx = frame["time_idx"].max()
    encoder = frame[frame["time_idx"] > last_idx - encoder_length].copy()
    last_known = frame[frame["time_idx"] == last_idx]

    decoder_frames = []
    for step in range(1, horizon + 1):
        future = last_known.copy()
        future["date"] = future["date"] + pd.DateOffset(months=step)
        decoder_frames.append(future)
    decoder = pd.concat(decoder_frames, ignore_index=True)
    decoder["time_idx"] = time_index(decoder["date"], start_date)
    decoder["year"] = decoder["date"].dt.year
    decoder["month"] = decoder["date"].dt.month.astype(str)
    decoder["is_holiday"] = holiday_month_flags(decoder["date"]).to_numpy()
    return pd.concat([encoder, decoder], ignore_index=True)


def _quantile_column(q: float) -> str:
    return "median_forecast" if np.isclose(q, 0.5) else f"q{round(q * 100):02d}"


def forecast(
    model: TemporalFusionTransformer, frame: pd.DataFrame, spec: FeatureSpec, config: Config
) -> Forecast:
    horizon = config.dataset.max_prediction_length
    data = build_future_frame(frame, spec, config.dataset.max_encoder_length, horizon)
    raw = model.predict(
        data,
        mode="raw",
        return_index=True,
        trainer_kwargs={"accelerator": resolve_accelerator(config.trainer.accelerator), "logger": False},
    )
    predictions = raw.output.prediction.cpu().numpy()  # (n_states, horizon, n_quantiles)
    if raw.index is None or "state" not in raw.index.columns:
        raise ValueError("Prediction index does not include state labels.")
    states = raw.index["state"].astype(str).to_numpy()
    if len(states) != predictions.shape[0]:
        raise ValueError(f"State count ({len(states)}) != prediction rows ({predictions.shape[0]}).")

    model_quantiles = list(model.loss.quantiles)
    last_date = frame["date"].max()
    future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=horizon, freq="MS")

    state_rows = {
        "state": np.repeat(states, horizon),
        "date": np.tile(future_dates, len(states)),
    }
    for q in config.forecast.quantiles_to_report:
        q_idx = int(np.argmin([abs(mq - q) for mq in model_quantiles]))
        state_rows[_quantile_column(q)] = predictions[:, :, q_idx].reshape(-1)
    state_df = pd.DataFrame(state_rows)

    # Unweighted mean of state forecasts, matching the notebook's national view.
    value_columns = [c for c in state_df.columns if c not in ("state", "date")]
    national_df = state_df.groupby("date", as_index=False)[value_columns].mean()
    logger.info(
        "Forecast %d states | %s - %s",
        len(states),
        future_dates[0].strftime("%b %Y"),
        future_dates[-1].strftime("%b %Y"),
    )
    return Forecast(state=state_df, national=national_df, last_observed_date=last_date)
