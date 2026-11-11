#!/usr/bin/env bash
# Start Airflow standalone with a known admin login, so the FastAPI service can call the REST API.
set -euo pipefail

python - <<'PY'
import json, os
path = os.environ["AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_PASSWORDS_FILE"]
with open(path, "w") as handle:
    json.dump({os.environ["AIRFLOW_USERNAME"]: os.environ["AIRFLOW_PASSWORD"]}, handle)
PY

exec airflow standalone
