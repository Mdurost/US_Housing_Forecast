from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import pytest

from housing_forecast.config import load_config

STATES = ["ca", "tx", "fl", "ny"]
COUNTIES_PER_STATE = 3
N_MONTHS = 60


@pytest.fixture
def raw_panel() -> pd.DataFrame:
    """Synthetic county x month panel shaped like data/processed/processed_data_pre_model.csv."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2017-07-01", periods=N_MONTHS, freq="MS")
    rows = []
    for s_idx, state in enumerate(STATES):
        for c in range(COUNTIES_PER_STATE):
            fips = f"{s_idx + 1:02d}{c + 1:03d}"
            base = 250_000 + 50_000 * s_idx + 10_000 * c
            for t, date in enumerate(dates):
                season = 1 + 0.03 * np.sin(2 * np.pi * date.month / 12)
                price = base * (1 + 0.004 * t) * season + rng.normal(0, 2_000)
                listings = 500 + 20 * c + rng.normal(0, 25)
                rows.append(
                    {
                        "date": date,
                        "county_fips": fips,
                        "median_listing_price": price,
                        "median_listing_price_mm": 0.01,
                        "active_listing_count": listings,
                        "total_listing_count": listings * 1.2 + rng.normal(0, 1),
                        "median_days_on_market": 40 + rng.normal(0, 5),
                        # Missing for early months so imputation indicators are produced.
                        "hotness_score": np.nan if t < 12 else 50 + rng.normal(0, 5),
                        "quality_flag": 0,
                        "cbsa_code": 10000 + s_idx,
                        "cbsa_title": f"Metro {state}",
                        "city": f"County {c}",
                        "state": state,
                        "U.S. 30 year FRM": 4 + 0.02 * t,
                        "unemployment_rate": 4 + rng.normal(0, 0.2),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def config(tmp_path: Path):
    """Small, fast configuration writing every artifact under ``tmp_path``."""
    overrides = [
        f"project_root={tmp_path.as_posix()}",
        "dataset.max_encoder_length=12",
        "dataset.min_encoder_length=6",
        "dataset.max_prediction_length=3",
        "dataset.batch_size=16",
        "model.hidden_size=8",
        "model.hidden_continuous_size=4",
        "trainer.accelerator=cpu",
        "trainer.max_epochs=1",
        "trainer.limit_train_batches=2",
        "tracking.enabled=false",
    ]
    return load_config(None, overrides)
