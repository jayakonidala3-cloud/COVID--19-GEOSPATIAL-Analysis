# Geo-Spatial Analysis Project

This folder contains a minimal, presentation-ready geospatial analytics demo and dashboard. It focuses on clarity and ease of explanation, with pure-Python modules that avoid compiled dependencies.

## Overview
- `src/dashboard/app.py`: Dash application that replicates the "COVID-19 Cases - Starter Dashboard" layout.
- `src/preprocessing/feature_engineering.py`: Readable feature engineering helpers (spatial lag, hotspot distance, rolling means) implemented in pure Python on list-of-dicts.
- `src/models/lightgbm_trainer.py`: A pure-Python baseline trainer that learns per-country moving averages (no LightGBM needed).
- `src/validation/spatial_cv.py`: Spatio-temporal cross-validation utilities (pandas/numpy). Useful for explanation; not required to run the dashboard.
- `requirements.txt`: Only includes easy-to-install packages used by the dashboard.

## How the Dashboard Works
1. Data
   - The app synthesizes realistic country-level time series for 2020-01-22 to 2020-03-11 (matching the screenshot period).
   - Each row has: `date`, `country`, `latitude`, `longitude`, `Confirmed`, `Recovered`, `Deaths`, `Active`.
   - If you have a real CSV, it can be wired in easily.
2. Interactions
   - Top bar shows the date range covered.
   - Controls: `DatePickerRange`, `Select Metric`, and `Select Country`.
3. Visuals
   - Four time-series panels: Confirmed, Recovered, Deaths, Active.
   - Four world maps showing bubble sizes per-country for the selected date.
   - Top Countries table lists the highest values for the selected metric on the last date in range.

## How the Feature Engineering Works
- Spatial lag: inverse-distance weighting of `Confirmed` from nearby countries using Haversine distance.
- Hotspot distance: distance to the nearest country above the 90th percentile of `Confirmed`.
- Rolling means: 7-day and 14-day country-level moving averages computed in date order.
- All implemented using straightforward loops for transparency during a presentation.

## How the Baseline Model Works
- `CountryMovingAverageModel` stores per-country `(date, Confirmed)` series and predicts as the average of the last `window` observations up to the prediction date.
- This demonstrates the `fit()`/`predict()` workflow without external ML libraries.

## Spatio-Temporal Cross-Validation (Advanced)
- `validation/spatial_cv.py` builds spatial blocks (quantile bins over lat/long) and temporal windows to yield train/validation indices.
- It uses pandas/numpy; if you want to run it, install those libraries separately.
- The dashboard does not depend on this module.

## Running the Dashboard
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the app:
   ```bash
   python src/dashboard/app.py
   ```
3. Open the URL:
   - `http://127.0.0.1:8050/`

## Using Your Own Data (Optional)
- Provide `data/covid_timeseries.csv` with columns:
  - `date` (YYYY-MM-DD), `country`, `latitude`, `longitude`, `confirmed`, `recovered`, `deaths`
- Update `src/dashboard/app.py` in the `load_data()` logic to read your CSV and compute `active = confirmed - recovered - deaths`.

## Talking Points for Your Presentation
- Clear separation of concerns: dashboard (UI), preprocessing (features), models (training), validation (CV strategy).
- Choice of pure-Python implementations for ease of understanding and to avoid build obstacles on Windows.
- Extensibility: swap the synthetic data for a real dataset, and swap the baseline model for a more advanced one when environment allows.