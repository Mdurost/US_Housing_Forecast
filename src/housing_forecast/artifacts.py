"""Load a trained model together with the feature state it was trained with."""

from __future__ import annotations

import logging

import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer

from housing_forecast.config import Config
from housing_forecast.features import FeatureSpec, fit_features, transform_features
from housing_forecast.modeling import load_model
from housing_forecast.provenance import ModelArtifacts

logger = logging.getLogger(__name__)


def check_model_compatibility(model: TemporalFusionTransformer, spec: FeatureSpec) -> None:
    expected = list(model.dataset_parameters["time_varying_unknown_reals"])
    actual = spec.time_varying_unknown_reals
    if set(expected) != set(actual):
        raise ValueError(
            "Feature spec does not match the model's training features. "
            f"Missing: {sorted(set(expected) - set(actual))}; "
            f"unexpected: {sorted(set(actual) - set(expected))}. Retrain the model."
        )


def load_model_and_frame(
    raw: pd.DataFrame, config: Config, artifacts: ModelArtifacts | None = None
) -> tuple[TemporalFusionTransformer, FeatureSpec, pd.DataFrame]:
    """Load a checkpoint (champion by default) and build the frame with its saved feature spec.

    Checkpoints created by the notebooks have no spec file; in that case the spec is
    refit on ``raw`` and validated against the features stored in the checkpoint.
    """
    artifacts = artifacts or ModelArtifacts.champion(config)
    if not artifacts.exists():
        raise FileNotFoundError(
            f"No model at {artifacts.checkpoint}. Run `housing-forecast train` then `promote`."
        )
    model = load_model(artifacts.checkpoint)
    if artifacts.feature_spec.is_file():
        spec = FeatureSpec.load(artifacts.feature_spec)
        frame = transform_features(raw, spec, config)
    else:
        logger.warning(
            "No feature spec at %s; refitting features on current data (legacy checkpoint).",
            artifacts.feature_spec,
        )
        frame, spec = fit_features(raw, config)
    check_model_compatibility(model, spec)
    return model, spec, frame
