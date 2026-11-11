"""FastAPI application exposing forecasts, model provenance, and Airflow pipeline control."""

# No `from __future__ import annotations`: FastAPI must resolve the dependency aliases
# defined inside create_app() when the routes are declared.
import logging
import secrets
from contextlib import asynccontextmanager
from typing import Annotated, Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Path, Query, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from housing_forecast import __version__
from housing_forecast.api.airflow_client import AirflowClient, AirflowError
from housing_forecast.api.settings import ApiSettings
from housing_forecast.api.store import ForecastStore, NotFound
from housing_forecast.config import Config, load_config

logger = logging.getLogger(__name__)

TagPath = Annotated[str, Path(pattern=r"^(latest|\d{4}-\d{2})$", description="YYYY-MM or 'latest'")]


class HealthResponse(BaseModel):
    status: str
    version: str
    champion_model: bool
    latest_forecast: str | None
    airflow: str


class ForecastIndex(BaseModel):
    latest: str | None
    available: list[str]


class ForecastResponse(BaseModel):
    forecast_month: str = Field(description="Last month of observed data the forecast starts from (YYYY-MM)")
    model: dict[str, Any] = Field(description="Champion model that produced the forecast")
    rows: list[dict[str, Any]]


class PipelineRunRequest(BaseModel):
    model_config = {"extra": "forbid"}

    skip_ingest: bool = Field(False, description="Reuse the existing processed dataset instead of downloading")
    force_promote: bool = Field(False, description="Promote the new model even if the gate rejects it")
    note: str | None = Field(None, max_length=500)


class PipelineRun(BaseModel):
    run_id: str
    state: str | None
    run_type: str | None = None
    run_after: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    conf: dict[str, Any] | None = None
    tasks: list[dict[str, Any]] | None = None


def _records(frame: pd.DataFrame, state: list[str] | None = None) -> list[dict[str, Any]]:
    if state:
        frame = frame[frame["state"].isin([s.lower() for s in state])]
    out = frame.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    numeric = out.select_dtypes("number").columns
    out[numeric] = out[numeric].round(2)
    return out.to_dict("records")


def _run(payload: dict[str, Any], tasks: list[dict[str, Any]] | None = None) -> PipelineRun:
    return PipelineRun(
        run_id=payload["dag_run_id"],
        state=payload.get("state"),
        run_type=payload.get("run_type"),
        run_after=payload.get("run_after"),
        start_date=payload.get("start_date"),
        end_date=payload.get("end_date"),
        conf=payload.get("conf"),
        tasks=[
            {k: t.get(k) for k in ("task_id", "state", "try_number", "start_date", "end_date", "duration")}
            for t in tasks
        ]
        if tasks is not None
        else None,
    )


def create_app(
    config: Config | None = None,
    settings: ApiSettings | None = None,
    airflow: AirflowClient | None = None,
) -> FastAPI:
    config = config or load_config()
    settings = settings or ApiSettings.from_env()
    store = ForecastStore(config)
    if airflow is None and settings.airflow_configured:
        airflow = AirflowClient(
            settings.airflow_url,
            settings.airflow_username,
            settings.airflow_password,
            settings.airflow_dag_id,
            timeout=settings.airflow_timeout_seconds,
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        if airflow is not None:
            airflow.close()

    app = FastAPI(
        title="US Housing Forecast API",
        version=__version__,
        description="12-month state and national median listing price forecasts (Temporal Fusion "
        "Transformer), model provenance, and control of the Airflow pipeline that produces them.",
        lifespan=lifespan,
    )
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def require_api_key(key: Annotated[str | None, Security(api_key_header)]) -> None:
        if not settings.api_key:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Pipeline control is disabled: no API key set.")
        if not key or not secrets.compare_digest(key, settings.api_key):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-API-Key header.")

    def require_airflow() -> AirflowClient:
        if airflow is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Airflow connection is not configured.")
        return airflow

    AirflowClientDep = Annotated[AirflowClient, Depends(require_airflow)]

    def tag_or_404(tag: str) -> str:
        try:
            return store.resolve_tag(tag)
        except NotFound as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc

    def model_info(tag: str) -> dict[str, Any]:
        metadata = store.forecast_metadata(tag)
        if metadata:
            return metadata.get("model", {})
        card = store.champion_card()
        return {k: card.get(k) for k in ("mlflow_run_id", "registered_version", "created_at", "promoted_at")}

    def airflow_error(exc: AirflowError) -> HTTPException:
        code = exc.status_code if exc.status_code in (404, 409, 503) else status.HTTP_502_BAD_GATEWAY
        return HTTPException(code, f"Airflow: {exc.detail}")

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    def health() -> HealthResponse:
        tags = store.tags()
        return HealthResponse(
            status="ok",
            version=__version__,
            champion_model=store.champion_exists(),
            latest_forecast=tags[-1] if tags else None,
            airflow="configured" if airflow is not None else "not_configured",
        )

    @app.get("/model", tags=["model"])
    def champion_model() -> dict[str, Any]:
        """Provenance of the production (champion) model: data, git commit, metrics, MLflow run."""
        card = store.champion_card()
        if not card:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No model card for the champion model.")
        return {k: v for k, v in card.items() if k != "config"}

    @app.get("/forecasts", response_model=ForecastIndex, tags=["forecasts"])
    def forecast_index() -> ForecastIndex:
        tags = store.tags()
        return ForecastIndex(latest=tags[-1] if tags else None, available=tags)

    @app.get("/forecasts/{tag}/national", response_model=ForecastResponse, tags=["forecasts"])
    def national_forecast(tag: TagPath) -> ForecastResponse:
        """National forecast (unweighted mean of state forecasts). ``tag`` is YYYY-MM or ``latest``."""
        tag = tag_or_404(tag)
        try:
            frame = store.national(tag)
        except NotFound as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
        return ForecastResponse(forecast_month=tag, model=model_info(tag), rows=_records(frame))

    @app.get("/forecasts/{tag}/states", response_model=ForecastResponse, tags=["forecasts"])
    def state_forecasts(
        tag: TagPath,
        state: Annotated[list[str] | None, Query(description="Filter by state code, e.g. ?state=ca&state=tx")] = None,
    ) -> ForecastResponse:
        tag = tag_or_404(tag)
        try:
            frame = store.states(tag)
        except NotFound as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
        rows = _records(frame, state)
        if state and not rows:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No forecasts for state(s) {state}.")
        return ForecastResponse(forecast_month=tag, model=model_info(tag), rows=rows)

    @app.get("/evaluation/{tag}", tags=["model"])
    def promotion_report(tag: TagPath) -> dict[str, Any]:
        """Hold-out metrics and the promotion decision recorded for the given data month."""
        tag = tag_or_404(tag)
        try:
            return store.promotion_report(tag)
        except NotFound as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc

    @app.post(
        "/pipeline/runs",
        response_model=PipelineRun,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["pipeline"],
        dependencies=[Depends(require_api_key)],
    )
    def trigger_pipeline(request: PipelineRunRequest, client: AirflowClientDep) -> PipelineRun:
        """Start an ingest -> train -> promote -> predict run in Airflow."""
        conf = {"skip_ingest": request.skip_ingest, "force_promote": request.force_promote}
        try:
            payload = client.trigger_run(conf, note=request.note)
        except AirflowError as exc:
            raise airflow_error(exc) from exc
        logger.info("Triggered Airflow run %s with %s", payload.get("dag_run_id"), conf)
        return _run(payload)

    @app.get(
        "/pipeline/runs",
        response_model=list[PipelineRun],
        tags=["pipeline"],
        dependencies=[Depends(require_api_key)],
    )
    def list_pipeline_runs(
        client: AirflowClientDep,
        limit: Annotated[int, Query(ge=1, le=100)] = 10,
    ) -> list[PipelineRun]:
        try:
            return [_run(run) for run in client.list_runs(limit)]
        except AirflowError as exc:
            raise airflow_error(exc) from exc

    @app.get(
        "/pipeline/runs/{run_id:path}",
        response_model=PipelineRun,
        tags=["pipeline"],
        dependencies=[Depends(require_api_key)],
    )
    def get_pipeline_run(run_id: str, client: AirflowClientDep) -> PipelineRun:
        """Run state plus per-task states (ingest, train, promote, predict)."""
        try:
            return _run(client.get_run(run_id), client.task_instances(run_id))
        except AirflowError as exc:
            raise airflow_error(exc) from exc

    return app
