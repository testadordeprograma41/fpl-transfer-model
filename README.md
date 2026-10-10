# FPL Transfer Model

A Python machine-learning project for estimating **Fantasy Premier League (FPL) player points** for an upcoming Gameweek. The current pipeline builds historical training features, trains a Random Forest regressor, constructs live pre-match features and produces a ranked prediction CSV.

> **Scope:** This repository currently focuses on **next-Gameweek point predictions**. A multi-Gameweek forecasting system and automated transfer optimizer are not yet implemented.

## Stack

Python, pandas, NumPy, scikit-learn, joblib and the official FPL API.

## Setup

```bash
git clone https://github.com/testadordeprograma41/fpl-transfer-model.git
cd fpl-transfer-model
python -m venv .venv
```

Activate the environment (Windows PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

## Prediction workflow

Run the scripts from the **repository root** so their relative data paths resolve correctly.

```bash
# Prepare training data from historical datasets
python src/build_training_data.py

# Train and save the Random Forest model
python src/train_random_forest.py

# Fetch upcoming fixtures from the live FPL API
python src/fetch_fixtures.py

# Build next-Gameweek player feature rows
python src/build_live_features.py

# Generate ranked player point predictions
python src/predict.py
```

The final script writes `data/processed/next_gw_predictions.csv`, including player, team, position, price, opponent, home/away and predicted points.

## Data and model artifacts

- `data/historical/` — historical Gameweek datasets
- `data/raw/` — raw fixtures and other inputs
- `data/processed/training_data.csv` — engineered training rows
- `data/processed/live_features.csv` — features known ahead of the next Gameweek
- `data/processed/next_gw_predictions.csv` — predictions
- `models/random_forest.joblib` — trained model

Some datasets, generated outputs and the trained model are **Git-ignored**. A fresh clone may require the underlying historical data to be supplied or regenerated before the complete pipeline will run.

## Modeling notes

The model uses lagged and rolling performance features, plus fixture and player context. It uses stable team names instead of season-dependent numeric team IDs to avoid mixing clubs across seasons. Evaluation uses chronological separation to reduce leakage.

Predictions are estimates, not guaranteed returns. Newly promoted teams, players without sufficient history and changing FPL API data can reduce reliability.

For technical decisions, evaluations and known limitations, see [CHANGELOG.md](CHANGELOG.md).
