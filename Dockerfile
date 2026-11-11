# syntax=docker/dockerfile:1
# Image for the housing-forecast pipeline (ingest / train / promote / predict / tune).
#   docker build -t housing-forecast:cpu .
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 -t housing-forecast:gpu .
FROM python:3.13-slim

# CPU PyTorch wheels by default (much smaller image). Point at a CUDA index for GPU builds.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MLFLOW_TRACKING_URI=sqlite:////app/mlruns/mlflow.db

WORKDIR /app

# Dependencies first so code changes don't invalidate this layer. Installing torch from the
# selected index first means the torch pin in requirements.txt is already satisfied.
COPY requirements.txt ./
RUN pip install "torch==2.11.0" --index-url "${TORCH_INDEX_URL}" \
 && grep -vE '^(-e |pytest|ruff|ipykernel)' requirements.txt > /tmp/requirements.txt \
 && pip install -r /tmp/requirements.txt \
 && rm /tmp/requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-deps . \
 && useradd --create-home --uid 1000 app \
 && mkdir -p data models outputs mlruns lightning_logs checkpoints \
 && chown -R app:app /app

USER app

# /app holds pyproject.toml + configs/, so it is the project root for relative config paths.
# Mount data/, models/, outputs/, and mlruns/ to persist results between runs.
VOLUME ["/app/data", "/app/models", "/app/outputs", "/app/mlruns"]

ENTRYPOINT ["housing-forecast"]
CMD ["--help"]
