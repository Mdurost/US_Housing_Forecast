# US Housing Forecast
![CI](https://github.com/Mdurost/US_Housing_Forecast/actions/workflows/ci.yml/badge.svg)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

## Motivation
The COVID-19 pandemic introduced significant changes to the U.S. housing market, including shifts in demand, migration patterns, and economic conditions. These changes drove rapid price growth which caused many individuals to feel home ownership is out of reach. 

This project uses housing and economic data to train a Temporal Fusion Transformer (TFT) that forecasts the median listing price 12 months ahead for every U.S. state, with 10–90% prediction intervals, to help first time home buyers understand and time their first home purchase. 

## Data sources
- **Housing Data (Realtor.com Research):**
  - Inventory Core Metrics – County History
  - Inventory Hotness Metrics – County History
  - https://www.realtor.com/research/data/
- **Economic Indicators:**
  - Freddie Mac Primary Mortgage Market Survey (weekly 30-year fixed rates) – https://www.freddiemac.com/pmms
  - FRED: Unemployment rate (`UNRATE`), CPI (`CPIAUCSL`), Building permits (`PERMIT`), Housing starts (`HOUST`) – https://fred.stlouisfed.org
- **Population:**
  - U.S. Census Bureau county population estimates – https://www2.census.gov/programs-surveys/popest/datasets/

## Project structure
```text
US_Housing_Forecast/
├── README.md
├── pyproject.toml                    # Package metadata, dependencies, `housing-forecast` CLI entry point
├── requirements.txt                  # Pinned, tested versions (installs this package in editable mode)
├── Dockerfile, .dockerignore         # Container image for the pipeline + API (CPU default, GPU via build arg)
├── docker-compose.yml, .env.example  # Airflow (scheduler) + Postgres + FastAPI service
├── airflow/                          # Airflow image, startup script, and the monthly DAG
├── .github/workflows/ci.yml          # CI: lint, tests, Docker build, Airflow DAG validation
├── LICENSE
├── configs/default.yaml              # All paths, data sources, and hyperparameters
├── src/housing_forecast/             # Production pipeline (importable package + CLI)
│   ├── config.py                     # Typed YAML config with --set overrides
│   ├── ingest.py                     # Download sources with retries; dated raw snapshots
│   ├── clean.py                      # Clean + merge sources into the county x month panel
│   ├── features.py                   # Fit/transform feature engineering (FeatureSpec)
│   ├── dataset.py                    # TimeSeriesDataSet / dataloaders
│   ├── modeling.py                   # TFT, trainer, callbacks, checkpoint loading
│   ├── train.py | evaluate.py | predict.py | tune.py
│   ├── artifacts.py                  # Load model + matching feature spec
│   └── cli.py                        # `housing-forecast <command>`
├── tests/                            # pytest suite (unit + end-to-end smoke test)
├── data_sourcing_and_cleaning.ipynb  # Exploratory version of the ingest/clean steps
├── TFT.ipynb                         # EDA, interpretation, and plots
├── data/
│   ├── raw/<YYYY-MM-DD>/                        # Raw source snapshots written by `ingest`
│   ├── processed/processed_data_pre_model.csv   # Model input written by `ingest`
│   └── studies/
│       ├── tft_optuna.db                        # Optuna study (SQLite) from hyperparameter tuning
│       └── optuna_trials/                       # Per-trial checkpoints written during tuning
├── models/
│   ├── tft_housing.ckpt              # Champion (production) model used by `predict`
│   ├── feature_spec.json             # Feature state fitted at training time
│   ├── model_card.json               # Provenance: data hash, git commit, metrics, MLflow run id
│   ├── candidate/                    # Latest trained model awaiting `promote` (git-ignored)
│   └── archive/<timestamp>/          # Previous champions, for rollback (git-ignored)
├── outputs/                          # Forecasts, evaluation metrics, promotion reports (git-ignored)
├── mlflow.db, mlruns/                # MLflow tracking store and run artifacts (git-ignored)
├── checkpoints/                      # ModelCheckpoint output (best-tft-{epoch}-{val_loss}.ckpt)
└── lightning_logs/                   # TensorBoard logs
```

## Results
- **Model:** Temporal Fusion Transformer (`pytorch-forecasting`) trained at the state level with a 36‑month encoder and 12‑month forecast horizon.
- **Evaluation:** The TFT is benchmarked on the hold-out 12-month validation window against a persistence (last-value) baseline and a 12-month seasonal-naive baseline using MASE, MAE, RMSE, and MAPE. Metrics are computed in dollars after the `log1p` target transform is reversed.
- **Interpretability:** Encoder/decoder variable importance and temporal attention are plotted via `interpret_output` / `plot_interpretation`, plus prediction-vs-actual charts by variable.
- **Forecasts produced:**
  1. **National** — mean of state-level median forecasts (unweighted), with trailing-36-month and full-history views.
  2. **State** — direct TFT output per state with 10–90% quantile bands and a month‑1 vs month‑12 summary table.

## Tech Stack
- Python, PyTorch, Lightning, pytorch-forecasting, pytorch-optimizer (Ranger)
- Optuna (TPE sampler + Hyperband pruner) for hyperparameter tuning
- Pandas, NumPy, statsmodels, `holidays`
- Matplotlib, Seaborn, TensorBoard
- MLflow for experiment tracking and the model registry

### Documentation
Tutorial
- https://pytorch-forecasting.readthedocs.io/en/v1.4.0/tutorials/stallion.html <br>

The paper that introduced TFT
- https://arxiv.org/abs/1912.09363

General pytorch loading documentation and second TFT example with darts.
- https://docs.pytorch.org/docs/stable/generated/torch.load.html <br>
- https://unit8co.github.io/darts/examples/13-TFT-examples.html <br>

## Installation

### Clone Repository
```bash
git clone "https://github.com/Mdurost/US_Housing_Forecast.git"
cd US_Housing_Forecast
```

### Create environment
```bash
python -m venv venv
# source venv/bin/activate  # Linux/macOS
venv\Scripts\activate  # Windows
```

### Install dependencies
```bash
pip install -r requirements.txt   # pinned versions + this package (editable)
# or, minimal runtime only:  pip install -e .
# optional extras:            pip install -e ".[tune,notebooks,dev]"
```

For a CUDA build of PyTorch, install it first from the PyTorch index (e.g. `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128`). A GPU is used automatically when available (`trainer.accelerator: auto`); otherwise the pipeline runs on CPU.

## Usage

### Pipeline (CLI)
```bash
housing-forecast ingest     # download sources -> data/raw/<date>/ snapshot -> data/processed/...csv
housing-forecast train      # train a candidate -> models/candidate/ (tracked as an MLflow run)
housing-forecast promote    # candidate vs baseline vs champion -> promote if better
housing-forecast evaluate   # hold-out metrics vs baselines -> outputs/evaluation/metrics_<YYYY-MM>.csv
housing-forecast predict    # champion forecasts -> outputs/forecasts/{state,national}_forecast_<YYYY-MM>.csv
housing-forecast tune       # LR finder + Optuna search (slow) -> outputs/tuning/<study>_best_params.yaml

housing-forecast run                 # ingest -> train -> promote -> predict
housing-forecast run --skip-ingest   # reuse the existing processed dataset
```
`python -m housing_forecast ...` is equivalent. Useful options:
- `--config path/to/config.yaml` to use another config file.
- `--set key.path=value` (repeatable) to override any config value, e.g. `--set trainer.max_epochs=5 --set trainer.accelerator=cpu`.
- `ingest --from-snapshot data/raw/2026-11-11` rebuilds the dataset from a saved snapshot without downloading.
- `evaluate --candidate` scores the candidate instead of the champion.
- `promote --force` (or `run --force`) promotes even when the gate rejects the candidate.

`train` saves a `feature_spec.json` next to the checkpoint. It records the correlated columns that were dropped, the imputation medians, the missingness indicators, and the `time_idx` origin. `evaluate` and `predict` apply that same spec to new data, so inference always matches training. Checkpoints created by the notebooks have no spec, so the spec is refit and checked against the feature list stored in the checkpoint.

### Model promotion (candidate → champion)
`train` never overwrites the production model. It writes `models/candidate/`. `promote` scores the candidate and the current champion on the same latest 12-month hold-out window. It promotes the candidate only if its MASE (`promotion.metric`):
1. beats the seasonal-naive baseline (`promotion.baseline_model`, set to `null` to skip), and
2. beats the champion by at least `promotion.min_improvement` (default: any improvement).

On promotion, the previous champion is copied to `models/archive/<timestamp>/`. The candidate files are copied into `models/`, and a new version of the registered model `us-housing-tft` is created in MLflow with the alias `champion`. Every decision, approved or rejected, is written to `outputs/evaluation/promotion_<YYYY-MM>.json`. To roll back, copy an archived folder's files back into `models/`.

> The champion may have seen part of the newest hold-out window during its own validation and early stopping. This slightly favours the champion, so the gate errs on the side of not promoting.

### Experiment tracking (MLflow)
Every `train` creates an MLflow run (experiment `us-housing-tft`) containing:
- **Parameters:** the full config, plus the data hash, row count, date range and state count.
- **Tags:** git commit and whether the working tree had uncommitted changes; after `promote`, `promoted` and `promotion_reason`.
- **Metrics:** per-epoch `train_loss_epoch`/`val_loss`, `best_val_loss`, and the hold-out MASE/MAE/RMSE/MAPE for the TFT and both baselines (`holdout_*`).
- **Artifacts:** `model/` (checkpoint, feature spec, model card), `evaluation/` (promotion report), and `forecasts/<YYYY-MM>/` (added by each `predict` run with this model).

Browse and compare runs:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db   # then open http://localhost:5000
```
Tracking is stored locally in `mlflow.db` + `mlruns/`. To use a shared server, set `MLFLOW_TRACKING_URI` (or `tracking.tracking_uri`). To turn tracking off, use `--set tracking.enabled=false`; the pipeline works the same, but nothing is logged to MLflow.

### Tests
```bash
pytest               # unit tests + end-to-end smoke tests on synthetic data
pytest -m "not slow" # skip the smoke tests
```

### Continuous integration
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and on pull requests into `main`:
1. **lint:** `ruff check src tests`
2. **test:** installs CPU PyTorch plus `requirements.txt`, then runs the unit tests and the end-to-end smoke tests (train → promote → predict with MLflow tracking, on synthetic data).
3. **docker:** after lint and test pass, builds the image and checks that the CLI and package imports work inside it.
4. **airflow-dag:** builds the Airflow image and checks that the DAG imports and has the expected ingest → train → promote → predict order.

### Scheduled pipeline (Airflow) + API (FastAPI)
`docker compose` runs the production setup:

| Service | URL | Role |
|---|---|---|
| `airflow` | http://localhost:8080 | Runs DAG `housing_forecast_monthly` on a schedule (default `0 6 15 * *`: 06:00 UTC on the 15th) |
| `api` | http://localhost:8000/docs | Serves forecasts and model info; starts and monitors pipeline runs in Airflow |
| `airflow-db` | (internal) | Postgres metadata database for Airflow |

```powershell
Copy-Item .env.example .env        # then replace every change-me value
docker compose up -d --build
```
The DAG runs each step as its own Airflow task (`ingest → train → promote → predict`), with its own retries, timeouts and logs. The steps call the `housing-forecast` CLI from a separate Python environment inside the Airflow image, so the pipeline's pinned packages never conflict with Airflow's. Trigger options: `skip_ingest` and `force_promote` (true/false). Only one run executes at a time; later triggers queue behind it.

> **First start:** even with `catchup=False`, Airflow immediately runs the most recent missed schedule slot (e.g. the 15th of the current month). Expect a full run, including download and training, right after the first `docker compose up`. To avoid it, pause the DAG in the UI before it starts, or set `AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=True`.

All services share `data/`, `models/`, `outputs/` and `mlruns/` from the host (or from `HOUSING_FORECAST_HOST_DIR`). The API mounts `models/` and `outputs/` read-only.

**API endpoints**

| Method & path | Auth | Description |
|---|---|---|
| `GET /health` | — | Service status, latest forecast month, whether Airflow is configured |
| `GET /forecasts` | — | Available forecast months |
| `GET /forecasts/{month}/national` | — | National forecast; `{month}` is `YYYY-MM` or `latest` |
| `GET /forecasts/{month}/states?state=ca&state=tx` | — | State forecasts, optionally filtered |
| `GET /model` | — | Champion model card (data hash, git commit, holdout metrics, MLflow run) |
| `GET /evaluation/{month}` | — | Promotion decision and holdout metrics for that data month |
| `POST /pipeline/runs` | `X-API-Key` | Start a run: `{"skip_ingest": false, "force_promote": false, "note": "..."}` |
| `GET /pipeline/runs?limit=10` | `X-API-Key` | Recent runs |
| `GET /pipeline/runs/{run_id}` | `X-API-Key` | Run state plus per-task states |

```powershell
$h = @{ "X-API-Key" = "<HOUSING_FORECAST_API_KEY>" }
Invoke-RestMethod http://localhost:8000/forecasts/latest/national
$run = Invoke-RestMethod -Method Post http://localhost:8000/pipeline/runs -Headers $h -ContentType application/json -Body '{"skip_ingest": true}'
Invoke-RestMethod "http://localhost:8000/pipeline/runs/$([uri]::EscapeDataString($run.run_id))" -Headers $h
```
- **Security:** the trigger endpoints only accept these true/false options, so callers can't pass shell commands or config overrides. Deployment-level CLI options go in `HOUSING_FORECAST_CLI_ARGS` in `.env`. Without `HOUSING_FORECAST_API_KEY`, the pipeline endpoints return 503.
- **Running the API without Docker:** `housing-forecast serve --port 8000` reads forecasts from the local `outputs/`. Set `AIRFLOW_API_URL`, `AIRFLOW_USERNAME` and `AIRFLOW_PASSWORD` to enable pipeline control.
- **Forecast metadata:** each `predict` run also writes `outputs/forecasts/forecast_metadata_<YYYY-MM>.json`, recording which model produced the forecast.

### Docker
The image packages the CLI with CPU PyTorch by default. Data, models, outputs and the MLflow store are mounted from the host, so results persist after the container exits.
```powershell
docker build -t housing-forecast:cpu .

docker run --rm `
  -v "${PWD}/data:/app/data" -v "${PWD}/models:/app/models" `
  -v "${PWD}/outputs:/app/outputs" -v "${PWD}/mlruns:/app/mlruns" `
  housing-forecast:cpu run --skip-ingest
```
Any CLI command or `--set` override works after the image name, e.g. `housing-forecast:cpu predict` or `housing-forecast:cpu --set trainer.max_epochs=5 train`.

- **MLflow store:** containers log to `mlruns/mlflow.db` (via `MLFLOW_TRACKING_URI`). This is separate from the local `mlflow.db`, because the stored artifact paths point inside the container. To browse container runs:
  ```powershell
  docker run --rm -p 5000:5000 -v "${PWD}/mlruns:/app/mlruns" --entrypoint mlflow housing-forecast:cpu `
    ui --host 0.0.0.0 --backend-store-uri sqlite:////app/mlruns/mlflow.db
  ```
- **GPU:** build with `--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 -t housing-forecast:gpu`, then run with `--gpus all`. This requires Docker Desktop with WSL2 GPU support, or the NVIDIA Container Toolkit on Linux.
- **Linux hosts:** the container runs as UID 1000. Add `--user "$(id -u):$(id -g)"` if the mounted folders belong to a different user.
- **GPU-trained models on CPU:** checkpoints trained on a GPU load fine on CPU-only machines such as Docker or CI. `load_model` resets the device recorded in the checkpoint's metric objects.

### Notebooks
The notebooks remain for exploration, interpretation, and plots:
1. **Build the dataset** — `data_sourcing_and_cleaning.ipynb` (exploratory version of `housing-forecast ingest`).
2. **Train / evaluate / forecast** — run `TFT.ipynb` from the top. Sections:
   - *Data Exploration* — missingness by year/state, correlation heatmap, target distribution.
   - *Data cleaning and model prep* — state-level aggregation, `time_idx`, variable classification, causal imputation, `TimeSeriesDataSet` construction.
   - *Training and Tuning* — trainer/callback setup and model definition. The learning-rate finder and Optuna tuning cells are **raw cells** (toggled off) because they are slow (tuning ran ~6 h); convert them to code cells to re-run.
   - *Train model* — `trainer.fit`, restores the best checkpoint and copies it to `models/tft_housing.ckpt`.
   - *Load model from checkpoint* — reload `models/tft_housing.ckpt` to skip training. A temporary `torch.load(weights_only=False)` patch is applied for compatibility with newer PyTorch versions.
   - *Baselines & interpretation* — metric table vs. naive baselines, variable importance, attention plots.
   - *Forecasting* — builds the encoder/decoder frame for the next 12 months and plots national and state forecasts. Edit `states_to_plot` to change the states shown.

## Methodology

### Data preprocessing (`TFT.ipynb`)
- Drop rows with a missing target (`median_listing_price`) and restrict to **2018‑01 onward** where features are dense.
- Drop lag features (`*_mm_*`, `*_yy_*`) — TFT learns temporal dependence directly.
- Add a monthly **US holiday indicator** (`is_holiday`) and `year` / `month` columns; sanitize column names; drop identifier columns (`county_fips`, `city`, `county_name`, `cbsa_code`, `cbsa_title`).
- Remove **highly correlated numeric features** (|ρ| > 0.90, dropping the member with the higher mean absolute correlation) to reduce noise and improve interpretability.
- Aggregate county rows to the **state × month** level (mean for numerics, max for categoricals).
- Build `time_idx` (months since first observation); training cutoff is the last 12 months held out for validation.

### Imputation
Causal, leakage-free imputation of covariates:
- `_was_missing` indicator columns are created for any covariate with ≥ 1% missingness and fed to the model as observed reals.
- Intermittent gaps are forward-filled **within state** only.
- Leading gaps are filled with **training-period medians**.
- The target is asserted to be unchanged by imputation.

### Feature classification
| Role | Variables |
|---|---|
| Static categorical | `state` |
| Time-varying known categorical | `month`, `is_holiday` |
| Time-varying known real | `time_idx` |
| Time-varying unknown real | `median_listing_price` (target) + all remaining numeric covariates + `_was_missing` indicators |

Target normalisation: `GroupNormalizer(groups=["state"], transformation="log1p")`.

### Model choice
TFT was chosen because it natively handles multiple series (states), static and time-varying covariates, built-in variable selection, multi-horizon quantile forecasts, and provides attention-based interpretability.

### Training strategy
- `QuantileLoss` (quantiles 0.02–0.98) with the **Ranger** optimizer and `reduce_on_plateau_patience=4`.
- Early stopping on `val_loss` (patience 10, `min_delta=1e-4`); `ModelCheckpoint` keeps only the best epoch.
- Hyperparameters were selected with Optuna (`TPESampler`, `HyperbandPruner`, 40 epochs/trial, up to 100 trials / 6 h) seeded by a Lightning learning-rate finder. Study `tft_housing_v2` is persisted in `data/studies/tft_optuna.db`; the location is defined once via `OPTUNA_STORAGE` in the seed cell and shared by the tuning and review cells.
- Random seeds fixed to `1` across Python, NumPy, PyTorch, and Lightning.

## Configuration / Parameters
All settings live in [`configs/default.yaml`](configs/default.yaml) (paths, source URLs, feature settings, dataset, model, trainer, and tuning). Override any value with `--set`, or copy the file and pass `--config`. Key values:

| Parameter | Config key | Value |
|---|---|---|
| `max_encoder_length` / `min_encoder_length` | `dataset.*` | 36 / 18 months |
| `max_prediction_length` | `dataset.max_prediction_length` | 12 months |
| `hidden_size` | `model.hidden_size` | 70 |
| `hidden_continuous_size` | `model.hidden_continuous_size` | 48 |
| `attention_head_size` | `model.attention_head_size` | 1 |
| `dropout` | `model.dropout` | 0.2336 |
| `learning_rate` | `model.learning_rate` | 2.34e‑4 |
| `gradient_clip_val` | `trainer.gradient_clip_val` | 0.1576 |
| `max_epochs` / `limit_train_batches` | `trainer.*` | 100 / 50 |
| `batch_size` (train / val) | `dataset.batch_size` (× `val_batch_size_multiplier`) | 128 / 1280 |
| `corr_threshold` | `features.corr_threshold` | 0.90 |
| `missing_indicator_threshold` | `features.missing_indicator_threshold` | 0.01 |
| `seed` | `seed` | 1 |

After tuning, copy the values from `outputs/tuning/<study>_best_params.yaml` into the `model` / `trainer` sections.

**Known data note:** the original cleaning notebook unintentionally dropped FRED housing starts (`HOUST`). The current model was trained without them, so `ingest.include_housing_starts` defaults to `false`. To add them, set it to `true`, then re-run `ingest` and `train`.

## Future Improvements
- County-level forecasts via proportional disaggregation of state forecasts
- Population-weighted national aggregate instead of an unweighted state mean
- Specify future covariates explicitly instead of forward-filling the last known values in the decoder
- Add more features and improve model accuracy
- Streamlit dashboard on top of the forecast API
- Region level predictions for migration patterns