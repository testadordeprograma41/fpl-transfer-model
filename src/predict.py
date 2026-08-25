"""Load the saved production model and generate real next-Gameweek
predictions.

This is the reusable inference entrypoint for anything downstream (e.g. a
future transfer recommender) that needs predicted points for a player:
call load_model() once, then predict(model, df) for any dataframe that has
all of modeling.features.FEATURE_COLUMNS.

Running this file directly now predicts a REAL upcoming Gameweek: it loads
data/processed/live_features.csv (built by build_live_features.py from the
live FPL API + data/raw/fixtures.csv -- each player's actual next opponent/
was_home/value plus lag/rolling features computed from their most recent
played matches), predicts points for every player in it, and writes
data/processed/next_gw_predictions.csv.

Previously this demoed the inference path using each player's most recent
PLAYED fixture as a stand-in for the next one (see CHANGELOG.md) -- that
was a sanity check that the saved model loads and predicts sensibly, not a
real forecast. It has been replaced: live_features.csv is only ever built
from data that's genuinely known ahead of the upcoming fixture (see
build_live_features.py's module docstring for exactly how, and its
documented limitations -- in particular, some players have no match
history yet this early in a season, and predictions for/against
newly-promoted teams carry an extra caveat around the model's categorical
encoding).
"""

import pandas as pd
import joblib

from modeling.data import (
    LIVE_FEATURES_FILE,
    MODEL_FILE,
    NEXT_GW_PREDICTIONS_FILE,
)
from modeling.features import FEATURE_COLUMNS


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"No saved model found at {MODEL_FILE}. "
            f"Run train_random_forest.py first to train and save one."
        )

    return joblib.load(MODEL_FILE)


def predict(model, df):
    """Predict expected points for each row of df.

    df must contain every column in modeling.features.FEATURE_COLUMNS.
    """

    return model.predict(df[FEATURE_COLUMNS])


def load_live_features():
    if not LIVE_FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"No {LIVE_FEATURES_FILE} found. Run build_live_features.py "
            f"first to build real next-Gameweek prediction inputs."
        )

    live = pd.read_csv(LIVE_FEATURES_FILE)

    missing_columns = [c for c in FEATURE_COLUMNS if c not in live.columns]

    if missing_columns:
        raise ValueError(
            f"{LIVE_FEATURES_FILE} is missing required model columns: "
            f"{missing_columns}. modeling/features.py and "
            f"build_live_features.py have drifted out of sync -- rerun "
            f"build_live_features.py, or check it still writes every "
            f"FEATURE_COLUMNS entry."
        )

    return live


def build_predictions_table(model, live):
    live = live.copy()
    live["predicted_points"] = predict(model, live)

    output = live[
        [
            "name",
            "team",
            "position",
            "value",
            "opponent_name",
            "was_home",
            "predicted_points",
        ]
    ].rename(
        columns={
            "value": "price",
            "opponent_name": "opponent",
            "was_home": "home_away",
        }
    )

    # "value"/price is stored in FPL's raw tenths scale (see
    # build_live_features.py) -- convert to GBP millions for display.
    output["price"] = output["price"] / 10
    output["home_away"] = output["home_away"].map({True: "Home", False: "Away"})

    return output.sort_values(
        "predicted_points", ascending=False
    ).reset_index(drop=True)


def main():
    model = load_model()
    live = load_live_features()

    predictions = build_predictions_table(model, live)

    NEXT_GW_PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(NEXT_GW_PREDICTIONS_FILE, index=False)

    print(f"Players predicted: {len(predictions):,}")
    print(f"Saved to: {NEXT_GW_PREDICTIONS_FILE}")
    print()
    print("Top 20 predicted players for the next Gameweek:")
    print(predictions.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
