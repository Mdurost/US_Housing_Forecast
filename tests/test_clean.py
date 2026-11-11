import numpy as np
import pandas as pd
import pytest

from housing_forecast.clean import (
    clean_census_population_long,
    clean_economic_indicators,
    clean_mortgage_rates,
    clean_supply_indicators,
    drop_lag_columns,
    fill_missing_months,
    standardize_panel,
)


def test_drop_lag_columns_only_removes_mm_yy_segments():
    df = pd.DataFrame(columns=["price", "price_mm", "price_yy", "median_dom_mm_day", "summary", "hammock"])
    out, dropped = drop_lag_columns(df)
    assert dropped == ["price_mm", "price_yy", "median_dom_mm_day"]
    assert list(out.columns) == ["price", "summary", "hammock"]


def test_fill_missing_months_only_fills_absent_targets():
    indicators = pd.DataFrame(
        {"date": pd.to_datetime(["2025-08-01", "2025-09-01", "2025-11-01"]), "cpi": [1.0, 2.0, 4.0]}
    )
    out = fill_missing_months(indicators, {"2025-10-01": "2025-09-01", "2025-11-01": "2025-08-01"})
    assert out["date"].is_monotonic_increasing
    assert out.set_index("date").loc["2025-10-01", "cpi"] == 2.0
    assert out.set_index("date").loc["2025-11-01", "cpi"] == 4.0


def test_economic_indicators_are_monthly_and_filtered():
    unemployment = pd.DataFrame({"DATE": ["1947-01-01", "1948-01-01", "1948-02-01"], "UNRATE": [1, 3.4, 3.8]})
    cpi = pd.DataFrame({"DATE": ["1948-01-01", "1948-02-01"], "CPIAUCSL": [23.7, "."]})
    out = clean_economic_indicators(unemployment, cpi, "1948-01-01")
    assert out["date"].min() == pd.Timestamp("1948-01-01")
    assert out.columns.tolist() == ["date", "unemployment_rate", "cpi"]


def test_supply_indicators_optionally_include_housing_starts():
    permits = pd.DataFrame({"DATE": ["1960-01-01", "1960-02-01"], "PERMIT": [1000, 1100]})
    starts = pd.DataFrame({"DATE": ["1960-01-01", "1960-02-01"], "HOUST": [1200, 1300]})
    assert clean_supply_indicators(permits, None, "1960-01-01").columns.tolist() == ["date", "building_permits"]
    assert "housing_starts" in clean_supply_indicators(permits, starts, "1960-01-01")


def test_mortgage_rates_weekly_to_monthly():
    sheet = pd.DataFrame(
        {
            "Unnamed: 0": ["Week", "2016-01-07", "2016-01-14", "2015-12-31", None],
            "Unnamed: 1": ["U.S. 30 yr FRM", 4.0, 3.0, 9.0, None],
            "Unnamed: 2": ["U.S. 15 yr FRM", 3.0, 2.0, 8.0, None],
        }
    )
    out = clean_mortgage_rates(sheet, "2016-01-01")
    assert out.to_dict("records") == [{"Month": pd.Timestamp("2016-01-01"), "U.S. 30 year FRM": 3.5}]


def test_census_wide_to_long_keeps_counties_only():
    wide = pd.DataFrame(
        {
            "SUMLEV": [40, 50],
            "STATE": [6, 6],
            "COUNTY": [0, 37],
            "STNAME": ["California", "California"],
            "CTYNAME": ["California", "Los Angeles County"],
            "POPESTIMATE2020": [39_000_000, 10_000_000],
            "POPESTIMATE2021": [39_100_000, 9_900_000],
            "BIRTHS2021": [400_000, 100_000],
        }
    )
    out = clean_census_population_long(wide, "2020s")
    assert out["county_fips"].unique().tolist() == ["06037"]
    assert out["year"].tolist() == [2020, 2021]
    assert out["population"].tolist() == [10_000_000, 9_900_000]


def test_standardize_panel_nulls_negatives_and_rejects_duplicates():
    panel = pd.DataFrame(
        {
            "date": ["2024-01-15", "2024-02-01"],
            "county_fips": ["1001", "01001"],
            "state": ["al", "al"],
            "median_listing_price": [-5, 200_000],
            "active_listing_count": [10, -1],
        }
    )
    out = standardize_panel(panel)
    assert out["county_fips"].tolist() == ["01001", "01001"]
    assert out["date"].tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")]
    assert np.isnan(out.loc[0, "median_listing_price"])
    assert np.isnan(out.loc[1, "active_listing_count"])

    with pytest.raises(ValueError, match="duplicate county-month"):
        standardize_panel(pd.concat([panel, panel]))
