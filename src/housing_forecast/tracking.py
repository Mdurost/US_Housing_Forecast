"""Thin MLflow wrapper. Every method is a no-op when tracking is disabled."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from housing_forecast.config import Config

logger = logging.getLogger(__name__)

MODEL_ARTIFACT_PATH = "model"


def flatten(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def metric_key(name: str) -> str:
    """MLflow-safe metric name, e.g. 'Seasonal naive (12-month)' -> 'seasonal_naive_12_month'."""
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_").replace("__", "_")


def resolve_tracking_uri(config: Config) -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI") or config.tracking.tracking_uri
    for prefix in ("sqlite:///", "file:///"):
        if uri.startswith(prefix):
            path = Path(uri[len(prefix):])
            if not path.is_absolute():
                path = config.resolve(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                return prefix + path.as_posix()
    return uri


class Tracker:
    def __init__(self, config: Config):
        self.config = config
        self.enabled = config.tracking.enabled
        self._client = None
        self.tracking_uri = resolve_tracking_uri(config) if self.enabled else None

    @property
    def client(self):
        if self._client is None:
            from mlflow.tracking import MlflowClient

            self._client = MlflowClient(tracking_uri=self.tracking_uri)
        return self._client

    def experiment_id(self) -> str:
        t = self.config.tracking
        experiment = self.client.get_experiment_by_name(t.experiment_name)
        if experiment is not None:
            return experiment.experiment_id
        artifact_location = self.config.resolve(t.artifact_location).as_uri()
        return self.client.create_experiment(t.experiment_name, artifact_location=artifact_location)

    def start_run(self, run_name: str, tags: dict[str, Any] | None = None) -> str | None:
        if not self.enabled:
            return None
        run = self.client.create_run(
            self.experiment_id(),
            run_name=run_name,
            tags={k: str(v) for k, v in (tags or {}).items()},
        )
        logger.info("MLflow run %s (%s) at %s", run.info.run_id, run_name, self.tracking_uri)
        return run.info.run_id

    def end_run(self, run_id: str | None, status: str = "FINISHED") -> None:
        if self.enabled and run_id:
            self.client.set_terminated(run_id, status=status)

    def log_params(self, run_id: str | None, params: dict[str, Any]) -> None:
        if not (self.enabled and run_id):
            return
        from mlflow.entities import Param

        items = [Param(k, str(v)[:6000]) for k, v in flatten(params).items()]
        for start in range(0, len(items), 100):
            self.client.log_batch(run_id, params=items[start : start + 100])

    def log_metrics(self, run_id: str | None, metrics: dict[str, float], step: int = 0) -> None:
        if not (self.enabled and run_id):
            return
        for key, value in metrics.items():
            if value is not None:
                self.client.log_metric(run_id, key, float(value), step=step)

    def set_tags(self, run_id: str | None, tags: dict[str, Any]) -> None:
        if not (self.enabled and run_id):
            return
        for key, value in tags.items():
            self.client.set_tag(run_id, key, str(value))

    def log_artifact(self, run_id: str | None, path: Path, artifact_path: str | None = None) -> None:
        if self.enabled and run_id and path.exists():
            self.client.log_artifact(run_id, str(path), artifact_path)

    def run_exists(self, run_id: str | None) -> bool:
        if not (self.enabled and run_id):
            return False
        from mlflow.exceptions import MlflowException

        try:
            self.client.get_run(run_id)
        except MlflowException:
            return False
        return True

    def attach_artifacts(self, run_id: str | None, paths: list[Path], artifact_path: str) -> bool:
        """Best-effort logging to an existing run (e.g. the champion's training run).

        The run may live in a different tracking store when a model is moved between
        machines or into containers; that must not fail the pipeline.
        """
        if not (self.enabled and run_id):
            return False
        if not self.run_exists(run_id):
            logger.warning(
                "MLflow run %s not found at %s; skipping artifact logging to it.", run_id, self.tracking_uri
            )
            return False
        for path in paths:
            self.log_artifact(run_id, path, artifact_path)
        return True

    def lightning_logger(self, run_id: str | None):
        """Lightning logger that streams per-epoch metrics into ``run_id``."""
        if not (self.enabled and run_id):
            return None
        from lightning.pytorch.loggers import MLFlowLogger

        return MLFlowLogger(
            experiment_name=self.config.tracking.experiment_name,
            tracking_uri=self.tracking_uri,
            run_id=run_id,
        )

    def register_champion(self, run_id: str | None, description: str = "") -> str | None:
        """Register the run's model artifacts as a new version and point the champion alias at it."""
        if not (self.enabled and run_id):
            return None
        from mlflow.exceptions import MlflowException

        t = self.config.tracking
        try:
            self.client.get_registered_model(t.registered_model_name)
        except MlflowException:
            self.client.create_registered_model(t.registered_model_name)
        source = f"{self.client.get_run(run_id).info.artifact_uri}/{MODEL_ARTIFACT_PATH}"
        version = self.client.create_model_version(
            t.registered_model_name, source=source, run_id=run_id, description=description
        )
        self.client.set_registered_model_alias(t.registered_model_name, t.champion_alias, version.version)
        logger.info("Registered %s version %s as '%s'", t.registered_model_name, version.version, t.champion_alias)
        return str(version.version)
