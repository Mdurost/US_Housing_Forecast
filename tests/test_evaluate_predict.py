import pandas as pd
import pytest
import torch

from housing_forecast.evaluate import forecast_metrics, naive_baselines
from housing_forecast.features import fit_features
from housing_forecast.predict import build_future_frame


def test_forecast_metrics_respects_mask():
    forecast = torch.tensor([[110.0, 999.0]])
    actual = torch.tensor([[100.0, 100.0]])
    scales = torch.tensor([[5.0]])
    mask = torch.tensor([[True, False]])
    metrics = forecast_metrics(forecast, actual, scales, mask)
    assert metrics == pytest.approx({"MASE": 2.0, "MAE": 10.0, "RMSE": 10.0, "MAPE": 10.0})


def test_naive_baselines():
    history = pd.Series(
        range(30), index=pd.MultiIndex.from_product([["ca"], range(30)], names=["state", "time_idx"]), dtype=float
    )
    index = pd.DataFrame({"state": ["ca"], "time_idx": [24]})
    persistence, seasonal = naive_baselines(history, index, torch.tensor([3]), torch.zeros(1, 3))
    assert persistence.tolist() == [[23.0, 23.0, 23.0]]
    assert seasonal.tolist() == [[12.0, 13.0, 14.0]]


def test_build_future_frame_extends_each_state(raw_panel, config):
    frame, spec = fit_features(raw_panel, config)
    data = build_future_frame(frame, spec, encoder_length=12, horizon=3)
    last = frame["time_idx"].max()
    future = data[data["time_idx"] > last]
    assert len(future) == frame["state"].nunique() * 3
    assert sorted(future["time_idx"].unique()) == [last + 1, last + 2, last + 3]
    first_future = future.sort_values("time_idx").iloc[0]
    assert first_future["month"] == str((frame["date"].max() + pd.DateOffset(months=1)).month)
