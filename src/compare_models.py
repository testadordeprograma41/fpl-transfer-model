from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

from modeling.cv import build_time_series_cv
from modeling.data import (
    load_training_data,
    sort_chronologically,
    split_by_season,
)
from modeling.evaluate import print_metrics
from modeling.features import FEATURE_COLUMNS, TARGET_COLUMN
from modeling.preprocessing import build_preprocessor


N_SPLITS = 5
N_SEARCH_ITER = 10
RANDOM_STATE = 42

# scikit-learn scorers are "greater is better" by convention, so MAE
# (where lower is better) needs negating -- cross_val_score/RandomizedSearchCV
# then report negative MAE and we flip the sign back for printing.
MAE_SCORER = make_scorer(mean_absolute_error, greater_is_better=False)

RF_PARAM_DISTRIBUTIONS = {
    "regressor__n_estimators": [100, 200, 300],
    "regressor__max_depth": [5, 8, 10, 15, None],
    "regressor__min_samples_leaf": [1, 3, 5, 10, 20],
    "regressor__max_features": ["sqrt", "log2", 0.5, None],
}


def build_linear_model():
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor()
            ),
            (
                "regressor",
                LinearRegression()
            ),
        ]
    )


def build_random_forest_model():
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor()
            ),
            (
                "regressor",
                # n_jobs=1 here: RandomizedSearchCV below parallelizes
                # across candidates/folds instead, which avoids the two
                # layers of parallelism fighting over the same CPUs.
                RandomForestRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                )
            ),
        ]
    )


def report_cv_score(model, X, y, cv, label):
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring=MAE_SCORER,
        n_jobs=-1,
    )

    mae_per_fold = -scores

    fold_summary = ", ".join(
        f"{value:.3f}" for value in mae_per_fold
    )

    print(
        f"{label:<26}"
        f"mean MAE: {mae_per_fold.mean():.3f}   "
        f"std: {mae_per_fold.std():.3f}   "
        f"folds: [{fold_summary}]"
    )

    return mae_per_fold.mean()


def main():
    df = load_training_data()

    train, test = split_by_season(df)

    # CV folds must respect chronological order (see modeling.cv) --
    # split_by_season leaves rows ordered per-player, so re-sort here.
    train = sort_chronologically(train)

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]

    cv = build_time_series_cv(n_splits=N_SPLITS)

    print(f"Train rows: {len(train):,}")
    print(
        f"CV: {N_SPLITS}-fold TimeSeriesSplit "
        f"(each fold trains on earlier weeks, validates on later ones)"
    )
    print()

    print("=" * 72)
    print("CROSS-VALIDATED COMPARISON (training seasons only, test untouched)")
    print("=" * 72)

    report_cv_score(
        build_linear_model(),
        X_train,
        y_train,
        cv,
        "Linear regression"
    )

    print()
    print(
        f"Tuning random forest "
        f"({N_SEARCH_ITER} candidates x {N_SPLITS} folds "
        f"= {N_SEARCH_ITER * N_SPLITS} fits)..."
    )

    search = RandomizedSearchCV(
        build_random_forest_model(),
        param_distributions=RF_PARAM_DISTRIBUTIONS,
        n_iter=N_SEARCH_ITER,
        scoring=MAE_SCORER,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    print(f"Best random forest params: {search.best_params_}")
    print(f"Best random forest CV MAE: {-search.best_score_:.3f}")

    print()
    print("=" * 72)
    print("FINAL HOLD-OUT COMPARISON (fit on full train, scored on 2025-26 test)")
    print("=" * 72)

    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

    linear_model = build_linear_model()
    linear_model.fit(X_train, y_train)

    print_metrics(
        "Linear regression",
        y_test,
        linear_model.predict(X_test)
    )

    print()

    best_random_forest = search.best_estimator_
    best_random_forest.fit(X_train, y_train)

    print_metrics(
        "Random forest (tuned)",
        y_test,
        best_random_forest.predict(X_test)
    )


if __name__ == "__main__":
    main()
