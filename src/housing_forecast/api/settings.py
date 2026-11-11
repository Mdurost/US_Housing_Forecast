"""Deployment settings for the API, read from environment variables (secrets never live in config)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ApiSettings:
    api_key: str | None = None
    airflow_url: str | None = None
    airflow_username: str | None = None
    airflow_password: str | None = None
    airflow_dag_id: str = "housing_forecast_monthly"
    airflow_timeout_seconds: float = 15.0

    @classmethod
    def from_env(cls) -> ApiSettings:
        return cls(
            api_key=os.environ.get("HOUSING_FORECAST_API_KEY") or None,
            airflow_url=os.environ.get("AIRFLOW_API_URL") or None,
            airflow_username=os.environ.get("AIRFLOW_USERNAME") or None,
            airflow_password=os.environ.get("AIRFLOW_PASSWORD") or None,
            airflow_dag_id=os.environ.get("AIRFLOW_DAG_ID", "housing_forecast_monthly"),
            airflow_timeout_seconds=float(os.environ.get("AIRFLOW_TIMEOUT_SECONDS", "15")),
        )

    @property
    def airflow_configured(self) -> bool:
        return bool(self.airflow_url and self.airflow_username and self.airflow_password)
