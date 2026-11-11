"""Clean and merge raw sources into the county x month panel used for modeling.

Ported from ``data_sourcing_and_cleaning.ipynb``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from housing_forecast.config import Config

logger = logging.getLogger(__name__)

LAG_COLUMN_PATTERN = r"(?:^|_)(?:mm|yy)(?:_|$)"
TEXT_COLUMNS = ["county_fips", "county_name", "city", "state", "cbsa_title"]
REQUIRED_KEYS = ["county_fips", "date"]
NONNEGATIVE_COLUMNS = {
    "median_listing_price",
    "median_listing_price_per_square_foot",
    "median_square_feet",
    "average_listing_price",
    "U.S. 30 year FRM",
    "unemployment_rate",
    "cpi",
    "building_permits",
    "housing_starts",
    "population",
}
CENSUS_METRICS = {
    "POPESTIMATE": "population",
    "NPOPCHG": "population_change",
    "BIRTHS": "births",
    "DEATHS": "deaths",
    "NATURALINC": "natural_increase",
    "NATURALCHG": "natural_increase",
    "INTERNATIONALMIG": "international_migration",
    "DOMESTICMIG": "domestic_migration",
    "NETMIG": "net_migration",
    "RINTERNATIONALMIG": "international_migration_rate",
    "RDOMESTICMIG": "domestic_migration_rate",
    "RNETMIG": "net_migration_rate",
}
CENSUS_DROP_COLUMNS = ["county_name", "state_name", "census_series", "population_change"]


def drop_lag_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop month-over-month / year-over-year lag features (``*_mm_*``, ``*_yy_*``)."""
    lag_columns = df.columns[
        df.columns.str.contains(LAG_COLUMN_PATTERN, case=False, regex=True)
    ].tolist()
    return df.drop(columns=lag_columns), lag_columns


def dedupe_realtor_hotness(realtor: pd.DataFrame, hotness: pd.DataFrame) -> pd.DataFrame:
    merge_keys = {"month_date_yyyymm", "county_fips"}
    duplicate_columns = (set(realtor.columns) & set(hotness.columns)) - merge_keys
    return hotness.drop(columns=list(duplicate_columns))


def _clean_fred_series(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = ["date", name]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame.dropna().copy()


def _merge_monthly(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(left, right, on="date", how="outer")
    merged["date"] = merged["date"].dt.to_period("M").dt.to_timestamp()
    return merged.groupby("date").mean().reset_index()


def clean_economic_indicators(
    unemployment: pd.DataFrame,
    cpi: pd.DataFrame,
    min_date: str,
    month_fills: dict[str, str] | None = None,
) -> pd.DataFrame:
    indicators = _merge_monthly(
        _clean_fred_series(unemployment, "unemployment_rate"),
        _clean_fred_series(cpi, "cpi"),
    )
    indicators = indicators[indicators["date"] >= min_date]
    return fill_missing_months(indicators, month_fills or {})


def fill_missing_months(indicators: pd.DataFrame, month_fills: dict[str, str]) -> pd.DataFrame:
    """Copy values from a source month into a missing target month (causal forward fill)."""
    indicators = indicators.copy()
    for target, source in month_fills.items():
        target_date, source_date = pd.Timestamp(target), pd.Timestamp(source)
        if indicators["date"].eq(target_date).any():
            continue
        fill_rows = indicators.loc[indicators["date"].eq(source_date)].copy()
        if fill_rows.empty:
            logger.warning("Cannot fill %s: source month %s is missing", target, source)
            continue
        fill_rows["date"] = target_date
        indicators = pd.concat([indicators, fill_rows], ignore_index=True)
    return indicators.sort_values("date").reset_index(drop=True)


def clean_supply_indicators(
    building_permits: pd.DataFrame,
    housing_starts: pd.DataFrame | None,
    min_date: str,
) -> pd.DataFrame:
    permits = _clean_fred_series(building_permits, "building_permits")
    if housing_starts is None:
        supply = permits
    else:
        supply = _merge_monthly(permits, _clean_fred_series(housing_starts, "housing_starts"))
    return supply[supply["date"] >= min_date].reset_index(drop=True)


def clean_mortgage_rates(mortgage_rates: pd.DataFrame, min_date: str) -> pd.DataFrame:
    """Convert Freddie Mac weekly PMMS spreadsheet to monthly 30-year fixed rates."""
    frame = mortgage_rates.copy(deep=True)
    frame = frame[frame["Unnamed: 0"].notnull()]
    frame.columns = frame.iloc[0].tolist()
    frame = frame[1:].reset_index(drop=True)

    first_columns = ["Week", "U.S. 30 year FRM"]
    frame.columns = first_columns + [str(c) for c in frame.columns[len(first_columns):]]
    week = pd.to_datetime(frame["Week"], errors="coerce")
    frame = frame.loc[week.notna(), first_columns].copy()
    frame["Week"] = week[week.notna()]
    frame["U.S. 30 year FRM"] = pd.to_numeric(frame["U.S. 30 year FRM"], errors="coerce")
    frame = frame[frame["Week"] >= pd.Timestamp(min_date)]

    frame["Month"] = frame["Week"].dt.to_period("M").dt.to_timestamp()
    return frame.drop(columns=["Week"]).groupby("Month").mean().reset_index()


def clean_census_population_long(dataframe: pd.DataFrame, census_series: str) -> pd.DataFrame:
    """Reshape a wide Census county population estimate file into county x year rows."""
    dataframe = dataframe.copy()
    dataframe.columns = dataframe.columns.str.strip().str.upper()

    # Census files also contain state totals; keep county records only.
    if "SUMLEV" in dataframe.columns:
        dataframe = dataframe.loc[pd.to_numeric(dataframe["SUMLEV"], errors="coerce").eq(50)].copy()

    dataframe["county_fips"] = (
        pd.to_numeric(dataframe["STATE"], errors="coerce").astype("Int64").astype("string").str.zfill(2)
        + pd.to_numeric(dataframe["COUNTY"], errors="coerce").astype("Int64").astype("string").str.zfill(3)
    )
    dataframe["county_name"] = dataframe["CTYNAME"].astype("string").str.strip()
    dataframe["state_name"] = dataframe["STNAME"].astype("string").str.strip()

    metric_columns = [
        column
        for column in dataframe.columns
        if any(
            column.startswith(prefix) and column.removeprefix(prefix).isdigit()
            for prefix in CENSUS_METRICS
        )
    ]
    census_long = dataframe.melt(
        id_vars=["county_fips", "county_name", "state_name"],
        value_vars=metric_columns,
        var_name="metric_year",
        value_name="value",
    )
    census_long["year"] = census_long["metric_year"].str.extract(r"(\d{4})$", expand=False).astype("Int64")
    census_long["metric"] = (
        census_long["metric_year"].str.replace(r"\d{4}$", "", regex=True).map(CENSUS_METRICS)
    )
    census_long["value"] = pd.to_numeric(census_long["value"], errors="coerce")

    census_long = (
        census_long.dropna(subset=["county_fips", "year", "metric"])
        .pivot_table(
            index=["county_fips", "county_name", "state_name", "year"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    census_long["census_series"] = census_series
    return census_long.sort_values(["county_fips", "year"]).reset_index(drop=True)


def combine_census(census_2010s: pd.DataFrame, census_2020s: pd.DataFrame) -> pd.DataFrame:
    long_2010s = clean_census_population_long(census_2010s, census_series="2010s")
    long_2020s = clean_census_population_long(census_2020s, census_series="2020s")
    combined = (
        pd.concat(
            [long_2010s.loc[long_2010s["year"] < 2020], long_2020s.loc[long_2020s["year"] >= 2020]],
            ignore_index=True,
        )
        .sort_values(["county_fips", "year"])
        .reset_index(drop=True)
    )
    duplicates = combined.duplicated(["county_fips", "year"], keep=False)
    if duplicates.any():
        raise ValueError(f"Found {duplicates.sum()} duplicate county-year rows in census data")
    return combined


def merge_realtor(realtor: pd.DataFrame, hotness: pd.DataFrame) -> pd.DataFrame:
    hotness = dedupe_realtor_hotness(realtor, hotness)
    merged = pd.merge(realtor, hotness, how="outer", on=["month_date_yyyymm", "county_fips"])
    merged[["city", "state"]] = merged["county_name"].str.rsplit(", ", n=1, expand=True)
    merged["month_date_yyyymm"] = pd.to_datetime(merged["month_date_yyyymm"].astype(str), format="%Y%m")
    merged["county_fips"] = merged["county_fips"].astype(str).str.zfill(5)
    return merged


def standardize_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Type coercion, validity checks, and de-duplication for the county x month panel."""
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    panel["county_fips"] = (
        pd.to_numeric(panel["county_fips"], errors="coerce").astype("Int64").astype("string").str.zfill(5)
    )
    for column in set(TEXT_COLUMNS) & set(panel.columns):
        panel[column] = panel[column].astype("string").str.strip()

    numeric_columns = panel.columns.difference(TEXT_COLUMNS + ["date"])
    panel[numeric_columns] = panel[numeric_columns].apply(pd.to_numeric, errors="coerce")
    panel[numeric_columns] = panel[numeric_columns].replace([np.inf, -np.inf], np.nan)

    panel = panel.dropna(subset=REQUIRED_KEYS)
    panel = panel.loc[panel["county_fips"].str.fullmatch(r"\d{5}", na=False)].copy()

    for column in panel.columns:
        if column.endswith("_count") or column in NONNEGATIVE_COLUMNS:
            panel.loc[panel[column] < 0, column] = np.nan

    duplicates = panel.duplicated(REQUIRED_KEYS, keep=False)
    if duplicates.any():
        raise ValueError(f"Found {duplicates.sum()} rows with duplicate county-month keys")
    return panel.sort_values(REQUIRED_KEYS).reset_index(drop=True)


def build_panel(sources: dict[str, pd.DataFrame], config: Config) -> pd.DataFrame:
    """Build the modeling panel from raw source frames keyed by ``config.sources`` names."""
    ingest = config.ingest
    realtor_merge = merge_realtor(sources["realtor_core"], sources["realtor_hotness"])
    mortgage = clean_mortgage_rates(sources["mortgage_rates"], ingest.mortgage_min_date)
    economic = clean_economic_indicators(
        sources["fred_unemployment"],
        sources["fred_cpi"],
        ingest.economic_min_date,
        ingest.economic_month_fills,
    )
    supply = clean_supply_indicators(
        sources["fred_building_permits"],
        sources.get("fred_housing_starts") if ingest.include_housing_starts else None,
        ingest.supply_min_date,
    )
    census = combine_census(sources["census_population_2010s"], sources["census_population_2020s"])
    census = census.drop(columns=CENSUS_DROP_COLUMNS, errors="ignore")

    panel = realtor_merge.rename(columns={"month_date_yyyymm": "date"})
    panel = panel.merge(mortgage.rename(columns={"Month": "date"}), on="date", how="left")
    panel = panel.merge(economic, on="date", how="left", validate="many_to_one")
    panel = panel.merge(supply, on="date", how="left", validate="many_to_one")
    panel["year"] = panel["date"].dt.year
    panel = panel.merge(census, on=["county_fips", "year"], how="left", validate="many_to_one")
    panel = panel.drop(columns=["year", "county_name"], errors="ignore")

    panel = standardize_panel(panel)
    panel, lag_columns = drop_lag_columns(panel)
    logger.info("Dropped %d lag features", len(lag_columns))
    panel = panel.dropna(subset=[config.features.target])
    panel["year"] = panel["date"].dt.year
    panel["month"] = panel["date"].dt.month
    logger.info("Panel built: %d rows x %d columns", *panel.shape)
    return panel
