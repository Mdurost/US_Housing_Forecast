# US Housing Forecast
![License](https://img.shields.io/badge/License-MIT-green)

## Motivation
The COVID-19 pandemic introduced significant changes to the U.S. housing market, including shifts in demand, migration patterns, and economic conditions. These changes drove rapid price growth which caused many individuals to feel home ownership is out of reach. 

This project aims to use housing and economic data to forecast median housing prices to help first time home buyers understand and time their first home purchase. 

## Data sources
- **Housing Data:**
  - https://www.realtor.com/research/data/
- **Economic Indicators:**
  - https://www.bls.gov/data/
  - https://www.redfin.com/news/data-center/
  - https://www.freddiemac.com
  
## Results
Features
  Forecast future values
  Handle multiple time series
  Supports covariates
  Visualization tools

## Tech Stack
  Python, PyTorch, Pandas
  Libraries like pytorch-forecasting, darts, timm, etc.
  Optional: infrastructure (Docker, AWS, etc.)

## Installation
git clone <repo>
cd <repo>
pip install -r requirements.txt

## Usage

Show exactly how to run it:
  python train.py
  python predict.py

## Methodology
  Data preprocessing
  Feature engineering
  Model choice (e.g., why TFT)
  Training strategy

## Configuration / Parameters
  Key hyperparameters
  Config files
  How to modify behavior

## Future Improvements
  Add more features
  Improve model accuracy
  Deploy as API
