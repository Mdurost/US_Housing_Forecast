"""Monthly US housing forecast pipeline: ingest -> train -> promote -> predict.

Each step runs the ``housing-forecast`` CLI from an isolated virtualenv inside the Airflow image,
so the pipeline's pinned ML dependencies never conflict with Airflow's own.

Trigger parameters (UI "Trigger DAG w/ config", REST API, or the FastAPI ``POST /pipeline/runs``):
  * ``skip_ingest``   - reuse the existing processed dataset instead of downloading sources
  * ``force_promote`` - promote the new model even if the promotion gate rejects it
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Param

PROJECT_DIR = os.environ.get("HOUSING_FORECAST_PROJECT_DIR", "/app")
CLI = os.environ.get("HOUSING_FORECAST_CLI", "/opt/housing-forecast/bin/housing-forecast")
# Trusted, deployment-level CLI options (e.g. "--set trainer.max_epochs=50"); never taken from run conf.
CLI_ARGS = os.environ.get("HOUSING_FORECAST_CLI_ARGS", "").strip()
# 06:00 UTC on the 15th: Realtor.com and FRED publish the prior month's data earlier in the month.
SCHEDULE = os.environ.get("HOUSING_FORECAST_SCHEDULE", "0 6 15 * *")

BASE_COMMAND = f"{CLI} {CLI_ARGS}".strip()


def _step(task_id: str, bash_command: str, timeout: timedelta, retries: int = 1) -> BashOperator:
    return BashOperator(
        task_id=task_id,
        bash_command=bash_command,
        cwd=PROJECT_DIR,
        execution_timeout=timeout,
        retries=retries,
        env={"MPLBACKEND": "Agg"},
        append_env=True,
    )


with DAG(
    dag_id="housing_forecast_monthly",
    description="Monthly TFT housing forecast: ingest, train, gated promotion, batch predict",
    schedule=SCHEDULE,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "housing-forecast", "retry_delay": timedelta(minutes=10)},
    params={
        "skip_ingest": Param(False, type="boolean", description="Reuse the existing processed dataset"),
        "force_promote": Param(False, type="boolean", description="Promote even if the gate rejects it"),
    },
    tags=["housing-forecast", "ml"],
    doc_md=__doc__,
) as dag:
    ingest = _step(
        "ingest",
        "{% if params.skip_ingest %}echo 'skip_ingest=true: using the existing processed dataset'"
        "{% else %}" + f"{BASE_COMMAND} ingest" + "{% endif %}",
        timeout=timedelta(hours=1),
        retries=3,
    )
    # Training is long and deterministic given the data; a failure needs investigation, not a retry.
    train = _step("train", f"{BASE_COMMAND} train", timeout=timedelta(hours=8), retries=0)
    promote = _step(
        "promote",
        f"{BASE_COMMAND} promote" + "{% if params.force_promote %} --force{% endif %}",
        timeout=timedelta(hours=1),
    )
    predict = _step("predict", f"{BASE_COMMAND} predict", timeout=timedelta(minutes=30))

    ingest >> train >> promote >> predict
