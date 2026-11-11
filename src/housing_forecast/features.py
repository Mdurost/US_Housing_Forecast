"""Feature engineering for the state-level TFT, split into fit (training) and transform (inference).

Ported from the "Data cleaning and model prep" section of ``TFT.ipynb``. Everything that is
learned from data (dropped correlated columns, imputation medians, missingness indicators,
``time_idx`` origin) is captured in a :class:`FeatureSpec` so inference reproduces training.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from housing_forecast.clean import drop_lag_columns
from housing_forecast.config import Config

logger = logging.getLogger(__name__)

GROUP_COLUMNS = ["state", "year", "month", "is_holiday"]
STATIC_CATEGORICALS = ["state"]
KNOWN_CATEGORICALS = ["month", "is_holiday"]
KNOWN_REALS = ["time_idx"]
INDICATOR_SUFFIX = "_was_missing"


@dataclass
class FeatureSpec:
    target: str
    start_date: str
    training_cutoff: int
    dropped_correlated_columns: list[str]
    impute_columns: list[str]
    impute_medians: dict[str, float]
    missing_indicator_columns: list[str]
    time_varying_unknown_reals: list[str]
    static_categoricals: list[str] = field(default_factory=lambda: list(STATIC_CATEGORICALS))
    time_varying_known_categoricals: list[str] = field(default_factory=lambda: list(KNOWN_CATEGORICALS))
    time_varying_known_reals: list[str] = field(default_factory=lambda: list(KNOWN_REALS))
    last_observed_date: str | None = None

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> FeatureSpec:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def sanitize_column_names(columns: pd.Index) -> pd.Index:
    return (
        pd.Index(columns.astype(str))
        .str.replace(r"[^0-9A-Za-z_]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )


def holiday_month_flags(dates: pd.Series) -> pd.Series:
    """Return "1" for months containing a US federal holiday, else "0"."""
    months = pd.to_datetime(dates).dt.to_period("M")
    years = sorted(months.dt.year.unique().tolist())
    holiday_months = {pd.Timestamp(day).to_period("M") for day in holidays.US(years=years)}
    return months.isin(holiday_months).astype("int8").astype(str)


def prepare_county_frame(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Stateless county-level preparation: filtering, calendar features, column hygiene."""
    target = config.features.target
    df = df.sort_values(["state", "date"])
    df = df.dropna(subset=[target]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df, _ = drop_lag_columns(df)
    df = df[df["date"] >= pd.Timestamp(config.features.min_date)].copy()

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df = df.drop(columns=[c for c in df.columns if c == "is_holiday" or c.startswith("is_holiday_")])
    df["is_holiday"] = holiday_month_flags(df["date"]).to_numpy()

    df = df.loc[:, ~df.columns.duplicated()]
    df["month"] = df["month"].astype(str)
    df.columns = sanitize_column_names(df.columns)
    return df.drop(columns=config.features.identifier_columns, errors="ignore")


def select_correlated_columns(df: pd.DataFrame, threshold: float, target: str) -> list[str]:
    """Pick one member of each numeric feature pair with |corr| > threshold to drop.

    The member with the higher mean absolute correlation to all other features is dropped.
    """
    protected = {"state", "date", "year", "month", "is_holiday", "time_idx", target}
    numeric_columns = [c for c in df.select_dtypes(include=[np.number]).columns if c not in protected]
    corr_matrix = df[numeric_columns].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    average_abs_corr = corr_matrix.mean()

    to_drop: set[str] = set()
    for column in upper.columns:
        for other in upper.index[upper[column] > threshold].tolist():
            if column in to_drop or other in to_drop:
                continue
            to_drop.add(column if average_abs_corr[column] >= average_abs_corr[other] else other)
    return sorted(to_drop)


def aggregate_to_state(df: pd.DataFrame) -> pd.DataFrame:
    agg = {
        column: "mean" if pd.api.types.is_numeric_dtype(df[column]) else "max"
        for column in df.columns
        if column not in GROUP_COLUMNS
    }
    return df.groupby(GROUP_COLUMNS).agg(agg).reset_index()


def time_index(dates: pd.Series, start_date: pd.Timestamp) -> pd.Series:
    """Months elapsed since ``start_date``."""
    dates = pd.to_datetime(dates)
    return (dates.dt.year * 12 + dates.dt.month) - (start_date.year * 12 + start_date.month)


def add_time_index(df: pd.DataFrame, start_date: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.Timestamp]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    if df.empty:
        raise ValueError("DataFrame is empty after date conversion.")
    if start_date is None:
        start_date = df["date"].min()
    df["time_idx"] = time_index(df["date"], start_date)
    for column in ["state", "month", "is_holiday"]:
        df[column] = df[column].astype(str)
    return df, start_date


def candidate_unknown_reals(df: pd.DataFrame, config: Config) -> list[str]:
    target = config.features.target
    excluded = set(config.features.excluded_unknown_reals) | {target}
    numeric = df.select_dtypes(include=[np.number]).columns
    return [target, *[c for c in numeric if c not in excluded and not c.endswith(INDICATOR_SUFFIX)]]


def apply_imputation(
    df: pd.DataFrame,
    impute_columns: list[str],
    medians: dict[str, float],
    indicator_columns: list[str],
    target: str,
) -> pd.DataFrame:
    """Causal imputation: indicators, within-state forward fill, then training medians."""
    df = df.sort_values(["state", "time_idx"]).reset_index(drop=True)
    target_before = df[target].copy()

    missing_columns = [c for c in impute_columns if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Input data is missing model covariates: {missing_columns}")

    missing_before = df[impute_columns].isna()
    for indicator in indicator_columns:
        source = indicator.removesuffix(INDICATOR_SUFFIX)
        df[indicator] = missing_before[source].astype("float32")

    df[impute_columns] = df.groupby("state", sort=False)[impute_columns].ffill()
    df[impute_columns] = df[impute_columns].fillna(pd.Series(medians, dtype="float64"))

    remaining = df[impute_columns].isna().sum()
    remaining = remaining[remaining > 0]
    if not remaining.empty:
        raise ValueError(f"Imputation left missing values: {remaining.to_dict()}")
    if not df[target].equals(target_before):
        raise AssertionError("Target values changed during covariate imputation.")
    return df


def fit_features(df: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, FeatureSpec]:
    """Fit feature state on training data and return the model-ready state x month frame."""
    feats = config.features
    target = feats.target

    county = prepare_county_frame(df, config)
    dropped = select_correlated_columns(county, feats.corr_threshold, target)
    logger.info("Dropping %d highly correlated features (|r| > %.2f)", len(dropped), feats.corr_threshold)
    state = aggregate_to_state(county.drop(columns=dropped))
    state, start_date = add_time_index(state)
    training_cutoff = int(state["time_idx"].max() - config.dataset.max_prediction_length)

    base_unknown_reals = candidate_unknown_reals(state, config)
    impute_columns = [c for c in base_unknown_reals if c != target]

    state = state.sort_values(["state", "time_idx"]).reset_index(drop=True)
    missing_fraction = state[impute_columns].isna().mean()
    indicator_columns = sorted(
        f"{c}{INDICATOR_SUFFIX}"
        for c in impute_columns
        if missing_fraction[c] >= feats.missing_indicator_threshold
    )

    # Medians come from forward-filled training rows only, so no validation data leaks in.
    filled = state.groupby("state", sort=False)[impute_columns].ffill()
    medians = filled.loc[state["time_idx"] <= training_cutoff].median(numeric_only=True)
    unavailable = medians[medians.isna()].index.tolist()
    if unavailable:
        raise ValueError("No observed training values are available for: " + ", ".join(unavailable))

    spec = FeatureSpec(
        target=target,
        start_date=start_date.strftime("%Y-%m-%d"),
        training_cutoff=training_cutoff,
        dropped_correlated_columns=dropped,
        impute_columns=impute_columns,
        impute_medians={k: float(v) for k, v in medians.items()},
        missing_indicator_columns=indicator_columns,
        time_varying_unknown_reals=base_unknown_reals + indicator_columns,
        last_observed_date=state["date"].max().strftime("%Y-%m-%d"),
    )
    state = apply_imputation(state, impute_columns, spec.impute_medians, indicator_columns, target)
    logger.info(
        "Features fit | %d states | %d unknown reals | %d indicators | training_cutoff=%d",
        state["state"].nunique(),
        len(spec.time_varying_unknown_reals),
        len(indicator_columns),
        training_cutoff,
    )
    return state, spec


def transform_features(df: pd.DataFrame, spec: FeatureSpec, config: Config) -> pd.DataFrame:
    """Apply a previously fitted :class:`FeatureSpec` to new data."""
    county = prepare_county_frame(df, config)
    state = aggregate_to_state(county.drop(columns=spec.dropped_correlated_columns, errors="ignore"))
    state, _ = add_time_index(state, pd.Timestamp(spec.start_date))
    return apply_imputation(
        state,
        spec.impute_columns,
        spec.impute_medians,
        spec.missing_indicator_columns,
        spec.target,
    )


def load_processed_data(config: Config) -> pd.DataFrame:
    path = config.resolve(config.paths.processed_data)
    logger.info("Loading %s", path)
    return pd.read_csv(path, parse_dates=["date"], low_memory=False)
