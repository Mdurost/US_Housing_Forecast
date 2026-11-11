"""Train a candidate TFT, tracking the run in MLflow and writing a model card."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from housing_forecast.config import Config
from housing_forecast.dataset import (
    build_training_dataset,
    build_validation_dataset,
    train_dataloader,
    val_dataloader,
)
from housing_forecast.features import FeatureSpec, fit_features
from housing_forecast.modeling import build_model, build_trainer, seed_everything
from housing_forecast.provenance import ModelArtifacts, data_summary, git_info, save_card, utc_now
from housing_forecast.tracking import MODEL_ARTIFACT_PATH, Tracker

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    artifacts: ModelArtifacts
    best_val_loss: float | None
    run_id: str | None
    spec: FeatureSpec
    frame: pd.DataFrame


def train(raw: pd.DataFrame, config: Config, data_path: Path | None = None) -> TrainResult:
    """Train on ``raw`` and write the candidate model to ``paths.candidate_dir``."""
    seed_everything(config.seed)
    artifacts = ModelArtifacts.candidate(config)
    tracker = Tracker(config)
    data = data_summary(data_path, raw)
    git = git_info(config.project_root)

    run_id = tracker.start_run(
        run_name=f"train-{data['end_date'][:7]}",
        tags={"stage": "train", "git_commit": git["commit"], "git_dirty": git["dirty"]},
    )
    try:
        tracker.log_params(run_id, {**config.to_dict(), "data": data})

        frame, spec = fit_features(raw, config)
        training = build_training_dataset(frame, spec, config)
        validation = build_validation_dataset(training, frame)
        tracker.log_params(
            run_id,
            {
                "features.n_unknown_reals": len(spec.time_varying_unknown_reals),
                "features.dropped_correlated": ",".join(spec.dropped_correlated_columns),
                "features.training_cutoff": spec.training_cutoff,
            },
        )

        model = build_model(training, config)
        mlflow_logger = tracker.lightning_logger(run_id)
        trainer, checkpoint = build_trainer(config, extra_loggers=[mlflow_logger] if mlflow_logger else None)
        trainer.fit(
            model,
            train_dataloaders=train_dataloader(training, config),
            val_dataloaders=val_dataloader(validation, config),
        )
        if not checkpoint.best_model_path:
            raise RuntimeError("Training finished without producing a checkpoint.")

        for path in artifacts.files():
            path.unlink(missing_ok=True)
        artifacts.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(checkpoint.best_model_path, artifacts.checkpoint)
        spec.save(artifacts.feature_spec)

        best = checkpoint.best_model_score
        best_val_loss = float(best) if best is not None else None
        save_card(
            {
                "created_at": utc_now(),
                "mlflow_run_id": run_id,
                "git": git,
                "data": data,
                "best_val_loss": best_val_loss,
                "epochs_trained": trainer.current_epoch,
                "n_parameters": int(model.size()),
                "config": config.to_dict(),
            },
            artifacts.model_card,
        )

        tracker.log_metrics(run_id, {"best_val_loss": best_val_loss, "epochs_trained": trainer.current_epoch})
        for path in artifacts.files():
            tracker.log_artifact(run_id, path, MODEL_ARTIFACT_PATH)
    except BaseException:
        tracker.end_run(run_id, status="FAILED")
        raise
    tracker.end_run(run_id)

    logger.info("Candidate model (val_loss=%s) saved to %s", best_val_loss, artifacts.checkpoint.parent)
    return TrainResult(artifacts, best_val_loss, run_id, spec, frame)
