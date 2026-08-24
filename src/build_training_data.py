import pandas as pd

from modeling.features import (
    LAG_SOURCE_COLUMNS,
    ROLLING_SOURCE_COLUMNS,
    ROLLING_WINDOWS,
    TRAINING_COLUMNS,
)


INPUT_FILE = "data/historical/historical_gws.csv"
OUTPUT_FILE = "data/processed/training_data.csv"


def load_data():
    df = pd.read_csv(INPUT_FILE)

    # Make sure observations are chronologically ordered
    df = df.sort_values(
        ["season", "element", "GW"]
    ).reset_index(drop=True)

    return df


def convert_numeric_columns(df):
    numeric_columns = [
        "minutes",
        "total_points",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
        "starts",
        "value",
        "selected",
        "transfers_in",
        "transfers_out",
        "transfers_balance",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df


def create_features(df):
    # Identify the same player within the same season
    player_group = df.groupby(
        ["season", "element"],
        group_keys=False
    )

    # What happened in the player's PREVIOUS fixture?
    for column in LAG_SOURCE_COLUMNS:
        df[f"{column}_last"] = (
            player_group[column].shift(1)
        )

    # Rolling averages based ONLY on previous matches.
    #
    # shift(1) is crucial:
    # when predicting GW10, GW10 itself must not be
    # included in the average.
    for column in ROLLING_SOURCE_COLUMNS:
        for window in ROLLING_WINDOWS:
            df[f"{column}_avg_{window}"] = (
                player_group[column]
                .transform(
                    lambda x, window=window:
                        x.shift(1)
                        .rolling(window, min_periods=1)
                        .mean()
                )
            )

    return df


def select_training_columns(df):
    return df[TRAINING_COLUMNS]


def main():
    df = load_data()
    df = convert_numeric_columns(df)
    df = create_features(df)
    training = select_training_columns(df)

    # First appearance has no historical information,
    # so remove observations without previous minutes.
    training = training.dropna(
        subset=["minutes_last"]
    )

    training.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"Training rows: {len(training):,}")
    print(f"Training columns: {len(training.columns)}")

    print("\nExample:")
    print(training.head().to_string(index=False))


if __name__ == "__main__":
    main()