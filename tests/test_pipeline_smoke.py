"""End-to-end smoke tests through the CLI on synthetic data (tiny model, CPU)."""

import json

import pandas as pd
import pytest

from housing_forecast.cli import main

pytestmark = pytest.mark.slow


@pytest.fixture
def cli_args(raw_panel, tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    processed = tmp_path / "data" / "processed.csv"
    processed.parent.mkdir(parents=True)
    raw_panel.to_csv(processed, index=False)
    overrides = {
        "project_root": tmp_path.as_posix(),
        "paths.processed_data": processed.as_posix(),
        "dataset.max_encoder_length": 12,
        "dataset.min_encoder_length": 6,
        "dataset.max_prediction_length": 3,
        "dataset.batch_size": 16,
        "model.hidden_size": 8,
        "model.hidden_continuous_size": 4,
        "trainer.accelerator": "cpu",
        "trainer.max_epochs": 1,
        "trainer.limit_train_batches": 2,
    }
    args = []
    for key, value in overrides.items():
        args += ["--set", f"{key}={value}"]
    return args


def _mlflow_client(tmp_path):
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri="sqlite:///" + (tmp_path / "mlflow.db").as_posix())


def test_gated_pipeline_with_mlflow_tracking(cli_args, raw_panel, tmp_path):
    # First run: no champion yet, baseline check disabled so the tiny model is promoted.
    first = [*cli_args, "--set", "promotion.baseline_model=null"]
    assert main([*first, "run", "--skip-ingest"]) == 0

    models = tmp_path / "models"
    for name in ("tft_housing.ckpt", "feature_spec.json", "model_card.json"):
        assert (models / name).is_file()
        assert (models / "candidate" / name).is_file()
    card = json.loads((models / "model_card.json").read_text())
    assert card["registered_version"] == "1"
    assert card["holdout"]["decision"]["promote"] is True
    assert card["data"]["sha256"]

    state = pd.read_csv(next((tmp_path / "outputs" / "forecasts").glob("state_forecast_*.csv")))
    assert len(state) == raw_panel["state"].nunique() * 3
    assert state[["q10", "median_forecast", "q90"]].notna().all().all()

    client = _mlflow_client(tmp_path)
    run = client.get_run(card["mlflow_run_id"])
    assert run.data.tags["promoted"] == "True"
    assert {"best_val_loss", "holdout_tft_MASE", "train_loss_epoch", "val_loss"} <= set(run.data.metrics)
    assert run.data.params["model.hidden_size"] == "8"
    artifacts = {a.path for a in client.list_artifacts(run.info.run_id)}
    assert {"model", "evaluation", "forecasts"} <= artifacts
    champion = client.get_model_version_by_alias("us-housing-tft", "champion")
    assert champion.run_id == run.info.run_id

    # Second candidate is compared with the champion; forcing promotion archives the old one.
    assert main([*cli_args, "train"]) == 0
    assert main([*cli_args, "promote"]) == 0
    report = json.loads(next((tmp_path / "outputs" / "evaluation").glob("promotion_*.json")).read_text())
    assert report["champion"] is not None
    assert report["champion_run_id"] == card["mlflow_run_id"]

    assert main([*cli_args, "promote", "--force"]) == 0
    assert any((models / "archive").iterdir())
    new_card = json.loads((models / "model_card.json").read_text())
    assert new_card["mlflow_run_id"] != card["mlflow_run_id"]
    assert str(client.get_model_version_by_alias("us-housing-tft", "champion").version) == "2"


def test_pipeline_without_tracking(cli_args, tmp_path):
    args = [*cli_args, "--set", "tracking.enabled=false", "--set", "promotion.baseline_model=null"]
    assert main([*args, "run", "--skip-ingest"]) == 0
    assert main([*args, "evaluate", "--candidate"]) == 0
    assert (tmp_path / "models" / "tft_housing.ckpt").is_file()
    assert json.loads((tmp_path / "models" / "model_card.json").read_text())["mlflow_run_id"] is None
    assert not (tmp_path / "mlflow.db").exists()
    assert next((tmp_path / "outputs" / "evaluation").glob("metrics_*_candidate.csv"))
