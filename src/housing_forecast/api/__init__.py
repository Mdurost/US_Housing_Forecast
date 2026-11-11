"""FastAPI service: serves batch forecasts and triggers the Airflow pipeline."""

from housing_forecast.api.app import create_app

__all__ = ["create_app"]
