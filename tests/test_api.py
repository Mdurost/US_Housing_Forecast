import json

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from housing_forecast.api import create_app
from housing_forecast.api.airflow_client import AirflowClient
from housing_forecast.api.settings import ApiSettings

API_KEY = "test-key"
DAG = "housing_forecast_monthly"
RUN_ID = "manual__2026-11-11T15:00:00+00:00"


@pytest.fixture
def outputs(config):
    forecasts = config.resolve(config.paths.output_dir) / "forecasts"
    forecasts.mkdir(parents=True)
    dates = pd.date_range("2026-08-01", periods=2, freq="MS")
    for tag in ("2026-06", "2026-07"):
        pd.DataFrame(
            {"state": ["ca", "ca", "tx", "tx"], "date": list(dates) * 2,
             "q10": [1.0, 2.0, 3.0, 4.0], "median_forecast": [1.5, 2.5, 3.5, 4.5], "q90": [2.0, 3.0, 4.0, 5.0]}
        ).to_csv(forecasts / f"state_forecast_{tag}.csv", index=False)
        pd.DataFrame(
            {"date": dates, "q10": [2.0, 3.0], "median_forecast": [2.5, 3.5], "q90": [3.0, 4.0]}
        ).to_csv(forecasts / f"national_forecast_{tag}.csv", index=False)
    (forecasts / "forecast_metadata_2026-07.json").write_text(
        json.dumps({"forecast_month": "2026-07", "model": {"registered_version": "3", "mlflow_run_id": "abc"}})
    )
    evaluation = config.resolve(config.paths.output_dir) / "evaluation"
    evaluation.mkdir(parents=True)
    (evaluation / "promotion_2026-07.json").write_text(json.dumps({"decision": {"promote": True}}))
    card = config.resolve(config.paths.model_card)
    card.parent.mkdir(parents=True)
    card.write_text(json.dumps({"mlflow_run_id": "abc", "registered_version": "3", "config": {"x": 1}}))
    return config


class FakeAirflow:
    """In-memory stand-in for the Airflow 3 REST API."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.token_calls = 0
        self.expire_next = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/auth/token":
            self.token_calls += 1
            body = json.loads(request.content)
            if body != {"username": "admin", "password": "pw"}:
                return httpx.Response(401, json={"detail": "bad credentials"})
            return httpx.Response(201, json={"access_token": f"token-{self.token_calls}"})
        if self.expire_next:
            self.expire_next = False
            return httpx.Response(401, json={"detail": "expired"})
        if not request.headers.get("Authorization", "").startswith("Bearer token-"):
            return httpx.Response(401, json={"detail": "unauthenticated"})
        run = {"dag_run_id": RUN_ID, "state": "queued", "run_type": "manual", "conf": {"skip_ingest": True}}
        if request.method == "POST" and path == f"/api/v2/dags/{DAG}/dagRuns":
            body = json.loads(request.content)
            assert body["logical_date"] is None
            return httpx.Response(200, json={**run, "conf": body["conf"]})
        if path == f"/api/v2/dags/{DAG}/dagRuns":
            return httpx.Response(200, json={"dag_runs": [run], "total_entries": 1})
        if path == f"/api/v2/dags/{DAG}/dagRuns/{RUN_ID}":
            return httpx.Response(200, json={**run, "state": "running"})
        if path == f"/api/v2/dags/{DAG}/dagRuns/{RUN_ID}/taskInstances":
            tasks = [{"task_id": t, "state": s, "try_number": 1} for t, s in [("ingest", "success"), ("train", "running")]]
            return httpx.Response(200, json={"task_instances": tasks, "total_entries": 2})
        return httpx.Response(404, json={"detail": f"DagRun not found: {path}"})


@pytest.fixture
def fake_airflow():
    return FakeAirflow()


@pytest.fixture
def client(outputs, fake_airflow):
    settings = ApiSettings(api_key=API_KEY)
    airflow = AirflowClient("http://airflow:8080", "admin", "pw", DAG, transport=httpx.MockTransport(fake_airflow))
    with TestClient(create_app(outputs, settings, airflow)) as test_client:
        yield test_client


def test_health(client):
    body = client.get("/health").json()
    assert body == {**body, "status": "ok", "champion_model": False, "latest_forecast": "2026-07", "airflow": "configured"}


def test_forecast_index_and_latest_national(client):
    assert client.get("/forecasts").json() == {"latest": "2026-07", "available": ["2026-06", "2026-07"]}
    body = client.get("/forecasts/latest/national").json()
    assert body["forecast_month"] == "2026-07"
    assert body["model"]["registered_version"] == "3"
    assert body["rows"][0] == {"date": "2026-08-01", "q10": 2.0, "median_forecast": 2.5, "q90": 3.0}


def test_state_forecasts_filter_and_errors(client):
    rows = client.get("/forecasts/2026-06/states", params={"state": ["TX"]}).json()["rows"]
    assert {r["state"] for r in rows} == {"tx"} and len(rows) == 2
    assert client.get("/forecasts/2026-06/states", params={"state": "zz"}).status_code == 404
    assert client.get("/forecasts/2025-01/national").status_code == 404
    assert client.get("/forecasts/..%2Fsecrets/national").status_code in (404, 422)
    assert client.get("/forecasts/2026-7/national").status_code == 422


def test_model_and_evaluation(client):
    model = client.get("/model").json()
    assert model["registered_version"] == "3" and "config" not in model
    assert client.get("/evaluation/latest").json()["decision"]["promote"] is True


def test_pipeline_endpoints_require_api_key(client):
    assert client.post("/pipeline/runs", json={}).status_code == 401
    assert client.get("/pipeline/runs", headers={"X-API-Key": "wrong"}).status_code == 401


def test_trigger_and_monitor_pipeline(client, fake_airflow):
    headers = {"X-API-Key": API_KEY}
    response = client.post("/pipeline/runs", json={"skip_ingest": True}, headers=headers)
    assert response.status_code == 202
    assert response.json()["run_id"] == RUN_ID
    assert response.json()["conf"] == {"skip_ingest": True, "force_promote": False}

    runs = client.get("/pipeline/runs", params={"limit": 5}, headers=headers).json()
    assert runs[0]["state"] == "queued"
    listed = next(r for r in fake_airflow.requests if r.method == "GET" and r.url.path.endswith("/dagRuns"))
    assert listed.url.params["order_by"] == "-run_after"

    detail = client.get(f"/pipeline/runs/{RUN_ID}", headers=headers).json()
    assert detail["state"] == "running"
    assert [t["task_id"] for t in detail["tasks"]] == ["ingest", "train"]
    assert fake_airflow.token_calls == 1


def test_trigger_rejects_invalid_payloads(client):
    headers = {"X-API-Key": API_KEY}
    assert client.post("/pipeline/runs", json={"skip_ingest": "rm -rf /"}, headers=headers).status_code == 422
    overrides = {"overrides": ["trainer.max_epochs=1"]}
    assert client.post("/pipeline/runs", json=overrides, headers=headers).status_code == 422


def test_expired_airflow_token_is_refreshed(client, fake_airflow):
    headers = {"X-API-Key": API_KEY}
    client.get("/pipeline/runs", headers=headers)
    fake_airflow.expire_next = True
    assert client.get("/pipeline/runs", headers=headers).status_code == 200
    assert fake_airflow.token_calls == 2


def test_unknown_run_maps_to_404(client):
    assert client.get("/pipeline/runs/nope", headers={"X-API-Key": API_KEY}).status_code == 404


def test_pipeline_control_disabled_without_key_or_airflow(outputs):
    with TestClient(create_app(outputs, ApiSettings())) as test_client:
        assert test_client.post("/pipeline/runs", json={}, headers={"X-API-Key": "x"}).status_code == 503
    with TestClient(create_app(outputs, ApiSettings(api_key=API_KEY))) as test_client:
        assert test_client.get("/health").json()["airflow"] == "not_configured"
        assert test_client.post("/pipeline/runs", json={}, headers={"X-API-Key": API_KEY}).status_code == 503
