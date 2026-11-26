# DemandLens

Retail demand forecasting with LightGBM quantile models,
rolling-origin validation, SQLite, and an interactive Streamlit dashboard.

## Current scope

- 20 food-product series from M5 store CA_1.
- 38,260 daily sales observations.
- Calendar events, SNAP indicators, historical prices, and sales features.
- Three rolling-origin validation folds, each forecasting 28 days.
- P10, P50, and P90 forecasts with uncertainty diagnostics.
- SQL-backed dashboard with product filters and downloadable forecasts.

## Validation results

Results cover 1,680 item-day forecasts across three validation folds.

| Metric | Seasonal naive | LightGBM P50 |
|---|---:|---:|
| MAE | 2.017 | 1.647 |
| WAPE | 104.89% | 85.63% |

LightGBM reduced pooled validation MAE by 18.36%.

The nominal 80% prediction interval achieved 85.65% coverage
after quantile-crossing adjustment. However, 13.45% of actual
observations exceeded P90, showing uneven tail calibration.

A separate conservative upper-bound scenario uses errors from
earlier folds only. It is exploratory and does not guarantee
per-product coverage or inventory service levels.

## Setup

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On macOS, LightGBM may also require:

```bash
brew install libomp
```
## Forecast dashboard

Product-level demand forecasts showing historical sales,
actual validation demand, P50 predictions, and P10–P90 bounds.

![DemandLens forecast dashboard](screenshots/forecast-dashboard.png)
## Data

Download these M5 files and place them in `data/raw/`:

- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv`

Dataset source:
https://zenodo.org/records/10203108

Raw data is not included in this repository.
Review the source dataset's terms before use.

## Run the workflow

```bash
python check_data.py
python prepare_data.py
python evaluate_baseline.py
python build_features.py
python evaluate_lightgbm.py
python evaluate_quantiles.py
python diagnose_forecasts.py
python evaluate_calibration.py
python build_database.py
python -m streamlit run dashboard.py
```

## Evaluation safeguards

- Products are selected deterministically by item ID.
- Sales-based features use observations at least 28 days earlier.
- Historical price features avoid future realized prices.
- Category encoders are fitted separately within each fold.
- Baseline and LightGBM predictions are matched on identical observations.
- March 28–April 24, 2016 is reserved and remains unevaluated.

## Limitations and planned work

- Results describe a small subset of one store.
- Recorded sales may understate demand during stockouts.
- Calendar events and SNAP indicators are not direct promotion records.
- Daily quantiles cannot simply be summed into lead-time demand quantiles.
- Inventory reorder simulation, MLflow tracking, and final holdout
  evaluation are planned; they are not implemented yet.

## Author

Rashmi Thimmaraju