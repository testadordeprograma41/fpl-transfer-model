from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/processed/training_data.csv")


def load_training_data():
    return pd.read_csv(INPUT_FILE)


def split_by_season(df):
    train = df[df["season"].isin(["2023-24", "2024-25"])].copy()
    test = df[df["season"] == "2025-26"].copy()

    return train, test


def sort_chronologically(df):
    """Order rows by season then gameweek, ACROSS players.

    split_by_season / load_training_data leave rows ordered by
    (season, element, GW) -- convenient for per-player feature engineering,
    but not in time order overall. Time-series cross-validation (see
    modeling.cv) needs rows in the order predictions would actually be
    made in, so callers doing CV should sort with this first.
    """

    return df.sort_values(["season", "GW"]).reset_index(drop=True)