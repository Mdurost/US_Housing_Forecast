import pandas as pd
import pandas.testing as pdt

from housing_forecast.features import (
    FeatureSpec,
    fit_features,
    holiday_month_flags,
    prepare_county_frame,
    select_correlated_columns,
    transform_features,
)


def test_holiday_flags_mark_months_with_federal_holidays():
    flags = holiday_month_flags(pd.Series(pd.to_datetime(["2024-01-01", "2024-03-01", "2024-08-01", "2024-12-01"])))
    assert flags.tolist() == ["1", "0", "0", "1"]


def test_prepare_county_frame_cleans_columns(raw_panel, config):
    out = prepare_county_frame(raw_panel, config)
    assert "median_listing_price_mm" not in out
    assert "U_S_30_year_FRM" in out
    assert not {"county_fips", "city", "cbsa_code", "cbsa_title"} & set(out.columns)
    assert out["date"].min() == pd.Timestamp(config.features.min_date)
    assert set(out["is_holiday"]) == {"0", "1"}


def test_select_correlated_columns_drops_one_of_each_pair(raw_panel, config):
    county = prepare_county_frame(raw_panel, config)
    dropped = select_correlated_columns(county, 0.9, config.features.target)
    assert len({"active_listing_count", "total_listing_count"} & set(dropped)) == 1
    assert config.features.target not in dropped


def test_fit_features_imputes_without_touching_target(raw_panel, config):
    frame, spec = fit_features(raw_panel, config)
    assert not frame[spec.time_varying_unknown_reals].isna().any().any()
    assert spec.time_varying_unknown_reals[0] == config.features.target
    assert "hotness_score_was_missing" in spec.missing_indicator_columns
    assert frame["time_idx"].min() == 0
    assert spec.training_cutoff == frame["time_idx"].max() - config.dataset.max_prediction_length

    county = prepare_county_frame(raw_panel, config)
    expected_target = county.groupby(["state", "date"])[config.features.target].mean()
    actual_target = frame.set_index(["state", "date"])[config.features.target]
    pdt.assert_series_equal(actual_target.sort_index(), expected_target.sort_index(), check_names=False)


def test_spec_round_trip_and_transform_reproduces_fit(raw_panel, config, tmp_path):
    frame, spec = fit_features(raw_panel, config)
    spec.save(tmp_path / "spec.json")
    loaded = FeatureSpec.load(tmp_path / "spec.json")
    assert loaded == spec
    pdt.assert_frame_equal(transform_features(raw_panel, loaded, config), frame)


def test_transform_uses_training_state_on_new_data(raw_panel, config):
    _, spec = fit_features(raw_panel, config)
    newer = raw_panel.copy()
    newer.loc[newer["date"] >= "2021-01-01", "hotness_score"] = float("nan")
    frame = transform_features(newer, spec, config)
    assert frame["time_idx"].min() == 0
    assert not frame[spec.time_varying_unknown_reals].isna().any().any()
