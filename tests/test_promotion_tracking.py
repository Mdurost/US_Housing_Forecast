import pandas as pd
import pytest

from housing_forecast.promote import decide, model_metric
from housing_forecast.provenance import ModelArtifacts, git_info
from housing_forecast.tracking import Tracker, flatten, metric_key, resolve_tracking_uri


def test_decide_requires_beating_the_baseline():
    decision = decide(candidate=0.5, baseline=0.4, champion=None)
    assert not decision.promote
    assert "baseline" in decision.reason


def test_decide_promotes_first_model_that_beats_baseline():
    assert decide(candidate=0.3, baseline=0.4, champion=None).promote


def test_decide_compares_against_champion():
    assert decide(candidate=0.30, baseline=0.4, champion=0.35).promote
    assert not decide(candidate=0.36, baseline=0.4, champion=0.35).promote
    assert not decide(candidate=0.35, baseline=0.4, champion=0.35).promote


def test_decide_min_improvement():
    assert not decide(candidate=0.34, baseline=None, champion=0.35, min_improvement=0.05).promote
    assert decide(candidate=0.33, baseline=None, champion=0.35, min_improvement=0.05).promote


def test_model_metric_lookup():
    metrics = pd.DataFrame({"model": ["TFT", "Persistence"], "MASE": [0.3, 0.5]})
    assert model_metric(metrics, "Persistence", "MASE") == 0.5
    with pytest.raises(KeyError):
        model_metric(metrics, "Missing", "MASE")


def test_metric_key_and_flatten():
    assert metric_key("Seasonal naive (12-month)") == "seasonal_naive_12_month"
    assert flatten({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a.b": 1, "a.c.d": 2, "e": 3}


def test_tracking_uri_resolves_relative_paths(config, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    uri = resolve_tracking_uri(config)
    assert uri == "sqlite:///" + (config.project_root / "mlflow.db").as_posix()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.internal:5000")
    assert resolve_tracking_uri(config) == "http://mlflow.internal:5000"


def test_disabled_tracker_is_a_no_op(config):
    tracker = Tracker(config)
    assert tracker.start_run("x") is None
    assert tracker.lightning_logger(None) is None
    assert tracker.register_champion(None) is None
    tracker.log_metrics(None, {"a": 1.0})
    assert not (config.project_root / "mlflow.db").exists()


def test_attach_artifacts_skips_runs_from_another_store(config, tmp_path, monkeypatch):
    """A model card can reference a run in a different MLflow store (moved model, containers)."""
    import dataclasses

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    enabled = dataclasses.replace(config, tracking=dataclasses.replace(config.tracking, enabled=True))
    tracker = Tracker(enabled)
    artifact = tmp_path / "forecast.csv"
    artifact.write_text("a,b\n1,2\n")

    assert tracker.attach_artifacts("0123456789abcdef0123456789abcdef", [artifact], "forecasts") is False

    run_id = tracker.start_run("real-run")
    assert tracker.attach_artifacts(run_id, [artifact], "forecasts") is True
    assert [a.path for a in tracker.client.list_artifacts(run_id, "forecasts")] == ["forecasts/forecast.csv"]


def test_candidate_and_champion_paths_differ(config):
    champion, candidate = ModelArtifacts.champion(config), ModelArtifacts.candidate(config)
    assert champion.checkpoint.name == candidate.checkpoint.name
    assert champion.checkpoint.parent != candidate.checkpoint.parent


def test_git_info_handles_non_repo(tmp_path):
    info = git_info(tmp_path)
    assert set(info) == {"commit", "dirty"}
