# syntax=docker/dockerfile:1
# Airflow image with the housing-forecast pipeline installed in an isolated virtualenv.
# Build context is the repository root (see docker-compose.yml).
FROM apache/airflow:3.4.0-python3.13

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV HOUSING_FORECAST_VENV=/opt/housing-forecast \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

USER root
RUN mkdir -p /opt/housing-forecast /app \
 && chown -R airflow:0 /opt/housing-forecast /app
USER airflow

# A separate venv keeps the pipeline's pinned dependencies independent of Airflow's constraints.
COPY --chown=airflow:0 requirements.txt /tmp/requirements.txt
RUN python -m venv "${HOUSING_FORECAST_VENV}" \
 && "${HOUSING_FORECAST_VENV}/bin/pip" install "torch==2.11.0" --index-url "${TORCH_INDEX_URL}" \
 && grep -vE '^(-e |pytest|ruff|ipykernel)' /tmp/requirements.txt > /tmp/requirements.pipeline.txt \
 && "${HOUSING_FORECAST_VENV}/bin/pip" install -r /tmp/requirements.pipeline.txt

# /app is the pipeline's project root: pyproject.toml + configs/ here, data/models/outputs/mlruns mounted.
COPY --chown=airflow:0 pyproject.toml README.md LICENSE /app/
COPY --chown=airflow:0 src /app/src
COPY --chown=airflow:0 configs /app/configs
RUN "${HOUSING_FORECAST_VENV}/bin/pip" install --no-deps /app \
 && mkdir -p /app/data /app/models /app/outputs /app/mlruns /app/lightning_logs /app/checkpoints

COPY --chown=airflow:0 airflow/dags /opt/airflow/dags
