# US Housing Forecast
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
├── requirements.txt
├── LICENSE
├── data_sourcing_and_cleaning.ipynb  # Download, clean, and merge raw sources
├── TFT.ipynb                         # EDA, feature prep, training, tuning, evaluation, forecasting
├── data/
│   ├── cleaned/cleaned_data.csv                 # Output of the cleaning notebook
│   ├── processed/processed_data_pre_model.csv   # Input to TFT.ipynb
│   └── studies/
│       ├── tft_optuna.db                        # Optuna study (SQLite) from hyperparameter tuning
│       └── optuna_trials/                       # Per-trial checkpoints written during tuning
├── models/tft_housing.ckpt           # Best checkpoint (lowest val_loss) copied here after training
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
pip install -r requirements.txt
```

A CUDA-capable GPU is detected automatically (`accelerator = "gpu" if torch.cuda.is_available() else "cpu"`); the notebooks fall back to CPU otherwise.

## Usage
1. **Build the dataset** — run `data_sourcing_and_cleaning.ipynb` end to end. It downloads the raw sources, cleans and merges them, and writes `data/processed/processed_data_pre_model.csv`.
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
Key hyperparameters are set directly in `TFT.ipynb` (there are no separate config files):

| Parameter | Value |
|---|---|
| `max_encoder_length` / `min_encoder_length` | 36 / 18 months |
| `max_prediction_length` | 12 months |
| `hidden_size` | 70 |
| `hidden_continuous_size` | 48 |
| `attention_head_size` | 1 |
| `dropout` | 0.2336 |
| `learning_rate` | 2.34e‑4 |
| `gradient_clip_val` | 0.1576 |
| `max_epochs` / `limit_train_batches` | 50 / 50 |
| `batch_size` (train / val) | 128 / 1280 |
| `corr_threshold` | 0.90 |
| `missing_indicator_threshold` | 0.01 |
| `seed` | 1 |

To modify behaviour, edit these values in the *Data cleaning and model prep* and *Training and Tuning* sections, or re-enable the raw tuning cells to search a new range.

## Future Improvements
- County-level forecasts via proportional disaggregation of state forecasts
- Population-weighted national aggregate instead of an unweighted state mean
- Specify future covariates explicitly instead of forward-filling the last known values in the decoder
- Add more features and improve model accuracy
- Deploy as API / Streamlit app
- Region level predictions for migration patterns