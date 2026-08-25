import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from modeling.data import (
    MODEL_DIR,
    MODEL_FILE,
    load_training_data,
    split_by_season,
)
from modeling.evaluate import print_metrics
from modeling.features import FEATURE_COLUMNS, TARGET_COLUMN
from modeling.preprocessing import build_preprocessor


# Hyperparameters found by compare_models.py's cross-validated search
# (best CV MAE 0.985, confirmed on the 2025-26 hold-out test: MAE 0.961 vs.
# the untuned 0.970). Re-run compare_models.py and update these if the
# training data or feature set changes meaningfully.
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
                    max_depth=15,
                    min_samples_leaf=5,
                    max_features="log2",
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
    y_train = train[TARGET_COLUMN]

    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

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

    # -----------------------------------------------------------------
    # Save a production model for reuse (see predict.py), refit on ALL
    # available data (train + test).
    #
    # The evaluation above, on the held-out 2025-26 test season, is our
    # honest estimate of how this model performs on data it hasn't seen.
    # Once that estimate is in hand, refitting on everything we have --
    # including that most recent season -- gives the saved model the most
    # information possible for predicting weeks that haven't happened yet.
    # -----------------------------------------------------------------

    X_all = df[FEATURE_COLUMNS]
    y_all = df[TARGET_COLUMN]

    production_model = build_model()

    production_model.fit(
        X_all,
        y_all
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        production_model,
        MODEL_FILE
    )

    print()
    print(f"Saved production model (fit on all {len(df):,} rows) to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
