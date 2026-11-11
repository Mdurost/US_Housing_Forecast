"""Minimal client for the Airflow 3 REST API (v2) used to trigger and monitor pipeline runs."""

from __future__ import annotations

import threading
from typing import Any
from urllib.parse import quote

import httpx


class AirflowError(RuntimeError):
    def __init__(self, status_code: int, detail: Any):
        super().__init__(f"Airflow API error {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class AirflowClient:
    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        dag_id: str,
        timeout: float = 15.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.dag_id = dag_id
        self._username = username
        self._password = password
        self._token: str | None = None
        self._lock = threading.Lock()
        self._http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, transport=transport)

    def close(self) -> None:
        self._http.close()

    def _fetch_token(self) -> str:
        try:
            response = self._http.post(
                "/auth/token", json={"username": self._username, "password": self._password}
            )
        except httpx.HTTPError as exc:
            raise AirflowError(503, f"Airflow unreachable: {exc}") from exc
        if response.status_code not in (200, 201):
            raise AirflowError(response.status_code, _detail(response))
        return response.json()["access_token"]

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        for attempt in range(2):
            with self._lock:
                if self._token is None:
                    self._token = self._fetch_token()
                token = self._token
            try:
                response = self._http.request(
                    method, path, headers={"Authorization": f"Bearer {token}"}, **kwargs
                )
            except httpx.HTTPError as exc:
                raise AirflowError(503, f"Airflow unreachable: {exc}") from exc
            if response.status_code == 401 and attempt == 0:
                with self._lock:
                    self._token = None
                continue
            if response.status_code >= 400:
                raise AirflowError(response.status_code, _detail(response))
            return response.json()
        raise AirflowError(401, "Authentication with Airflow failed")

    def _runs_path(self, run_id: str | None = None) -> str:
        path = f"/api/v2/dags/{quote(self.dag_id, safe='')}/dagRuns"
        return f"{path}/{quote(run_id, safe='')}" if run_id else path

    def trigger_run(self, conf: dict[str, Any], note: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"logical_date": None, "conf": conf}
        if note:
            body["note"] = note
        return self._request("POST", self._runs_path(), json=body)

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._request("GET", self._runs_path(run_id))

    def list_runs(self, limit: int = 10) -> list[dict[str, Any]]:
        data = self._request("GET", self._runs_path(), params={"limit": limit, "order_by": "-run_after"})
        return data.get("dag_runs", [])

    def task_instances(self, run_id: str) -> list[dict[str, Any]]:
        data = self._request("GET", f"{self._runs_path(run_id)}/taskInstances")
        return data.get("task_instances", [])


def _detail(response: httpx.Response) -> Any:
    try:
        return response.json().get("detail", response.text)
    except (ValueError, AttributeError):
        return response.text
