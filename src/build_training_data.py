import pandas as pd


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
    lag_columns = [
        "minutes",
        "total_points",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "starts",
    ]

    for column in lag_columns:
        df[f"{column}_last"] = (
            player_group[column].shift(1)
        )

    # Rolling averages based ONLY on previous matches.
    #
    # shift(1) is crucial:
    # when predicting GW10, GW10 itself must not be
    # included in the average.
    rolling_columns = [
        "minutes",
        "total_points",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
    ]

    for column in rolling_columns:

        df[f"{column}_avg_3"] = (
            player_group[column]
            .transform(
                lambda x:
                    x.shift(1)
                    .rolling(3, min_periods=1)
                    .mean()
            )
        )

        df[f"{column}_avg_5"] = (
            player_group[column]
            .transform(
                lambda x:
                    x.shift(1)
                    .rolling(5, min_periods=1)
                    .mean()
            )
        )

    return df


def select_training_columns(df):
    columns = [
        # Identification
        "season",
        "GW",
        "element",
        "name",
        "position",
        "team",

        # Information known for the upcoming fixture
        "value",
        "opponent_team",
        "was_home",

        # Previous fixture
        "minutes_last",
        "total_points_last",
        "expected_goals_last",
        "expected_assists_last",
        "expected_goal_involvements_last",
        "starts_last",

        # Previous 3 fixtures
        "minutes_avg_3",
        "total_points_avg_3",
        "expected_goals_avg_3",
        "expected_assists_avg_3",
        "expected_goal_involvements_avg_3",

        # Previous 5 fixtures
        "minutes_avg_5",
        "total_points_avg_5",
        "expected_goals_avg_5",
        "expected_assists_avg_5",
        "expected_goal_involvements_avg_5",

        # TARGET
        "total_points",
    ]

    return df[columns]


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