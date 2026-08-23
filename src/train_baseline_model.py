from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


INPUT_FILE = Path("data/processed/training_data.csv")


def load_data():
    return pd.read_csv(INPUT_FILE)


def split_data(df):
    # Train on older seasons
    train = df[df["season"].isin(["2023-24", "2024-25"])].copy()

    # Test on the most recent completed season
    test = df[df["season"] == "2025-26"].copy()

    return train, test


def build_model():
    numeric_features = [
        "value",
        "minutes_last",
        "total_points_last",
        "expected_goals_last",
        "expected_assists_last",
        "expected_goal_involvements_last",
        "starts_last",
        "minutes_avg_3",
        "total_points_avg_3",
        "expected_goals_avg_3",
        "expected_assists_avg_3",
        "expected_goal_involvements_avg_3",
        "minutes_avg_5",
        "total_points_avg_5",
        "expected_goals_avg_5",
        "expected_assists_avg_5",
        "expected_goal_involvements_avg_5",
    ]

    categorical_features = [
        "position",
        "team",
        "opponent_team",
        "was_home",
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    return model, numeric_features, categorical_features


def main():
    df = load_data()

    train, test = split_data(df)

    model, numeric_features, categorical_features = build_model()

    feature_columns = (
        numeric_features + categorical_features
    )

    X_train = train[feature_columns]
    y_train = train["total_points"]

    X_test = test[feature_columns]
    y_test = test["total_points"]

    print(f"Train rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # --------------------------------------------------
    # Evaluation: all players
    # --------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    print()
    print("ALL PLAYERS")
    print("-" * 50)
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")


    # --------------------------------------------------
    # Evaluation: likely starters
    # --------------------------------------------------

    starter_mask = test["minutes_avg_3"] >= 60

    starter_actual = y_test[starter_mask]
    starter_predictions = predictions[starter_mask]

    starter_mae = mean_absolute_error(
        starter_actual,
        starter_predictions
    )

    starter_rmse = mean_squared_error(
        starter_actual,
        starter_predictions
    ) ** 0.5

    print()
    print(
        "LIKELY STARTERS "
        "(previous 3-match avg >= 60 mins)"
    )
    print("-" * 50)
    print(f"Rows: {starter_mask.sum():,}")
    print(f"MAE:  {starter_mae:.3f}")
    print(f"RMSE: {starter_rmse:.3f}")

    results = test[
        [
            "season",
            "GW",
            "name",
            "position",
            "team",
            "total_points",
        ]
    ].copy()

    results["predicted_points"] = predictions
    results["error"] = (
        results["predicted_points"]
        - results["total_points"]
    ).abs()

    print()
    print("Sample predictions:")
    print(
        results[
            [
                "GW",
                "name",
                "total_points",
                "predicted_points",
                "error",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()