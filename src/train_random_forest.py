import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from modeling.data import (
    load_training_data,
    split_by_season,
)
from modeling.evaluate import print_metrics
from modeling.features import FEATURE_COLUMNS
from modeling.preprocessing import build_preprocessor


def build_model():
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor()
            ),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                )
            ),
        ]
    )


def main():
    df = load_training_data()

    train, test = split_by_season(df)

    X_train = train[FEATURE_COLUMNS]
    y_train = train["total_points"]

    X_test = test[FEATURE_COLUMNS]
    y_test = test["total_points"]

    print(f"Train rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")

    model = build_model()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    print()

    print_metrics(
        "ALL PLAYERS",
        y_test,
        predictions
    )

    starter_mask = (
        test["minutes_avg_3"] >= 60
    )

    starter_actual = (
        y_test[starter_mask]
    )

    starter_predictions = (
        predictions[starter_mask]
    )

    print()
    print(
        "LIKELY STARTERS "
        "(previous 3-match avg >= 60 mins)"
    )
    print(f"Rows: {starter_mask.sum():,}")

    print_metrics(
        "",
        starter_actual,
        starter_predictions
    )

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

    results["predicted_points"] = (
        predictions
    )

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

    # Feature importances give a quick sanity check that the model
    # is leaning on sensible signals (not noise).
    print()
    print("Top 15 feature importances:")

    feature_names = (
        model
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )

    importances = (
        model
        .named_steps["regressor"]
        .feature_importances_
    )

    importance_table = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values(
        "importance",
        ascending=False
    )

    print(
        importance_table
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
