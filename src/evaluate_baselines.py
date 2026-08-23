import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


INPUT_FILE = "data/processed/training_data.csv"


def calculate_metrics(actual, predicted, label):
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5

    print(
        f"{label:<30}"
        f"MAE: {mae:>6.3f}   "
        f"RMSE: {rmse:>6.3f}"
    )


def main():
    df = pd.read_csv(INPUT_FILE)

    test = df[df["season"] == "2025-26"].copy()

    print(f"Test rows: {len(test):,}")
    print()

    # -------------------------
    # Baseline 1:
    # Predict zero for everyone
    # -------------------------

    test["pred_zero"] = 0

    # -------------------------
    # Baseline 2:
    # Predict previous match
    # -------------------------

    test["pred_last"] = test["total_points_last"]

    # -------------------------
    # Baseline 3:
    # Predict recent 3-match avg
    # -------------------------

    test["pred_avg_3"] = test["total_points_avg_3"]

    # -------------------------
    # Baseline 4:
    # Predict recent 5-match avg
    # -------------------------

    test["pred_avg_5"] = test["total_points_avg_5"]

    print("ALL PLAYERS")
    print("-" * 60)

    calculate_metrics(
        test["total_points"],
        test["pred_zero"],
        "Always predict 0"
    )

    calculate_metrics(
        test["total_points"],
        test["pred_last"],
        "Previous match"
    )

    calculate_metrics(
        test["total_points"],
        test["pred_avg_3"],
        "3-match average"
    )

    calculate_metrics(
        test["total_points"],
        test["pred_avg_5"],
        "5-match average"
    )

    # -------------------------
    # Players likely to play
    #
    # We only know past minutes,
    # which is acceptable for prediction.
    # -------------------------

    likely_players = test[
        test["minutes_avg_3"] >= 60
    ].copy()

    print()
    print(
        f"LIKELY STARTERS "
        f"(previous 3-match avg >= 60 mins)"
    )
    print(
        f"Rows: {len(likely_players):,}"
    )
    print("-" * 60)

    calculate_metrics(
        likely_players["total_points"],
        likely_players["pred_zero"],
        "Always predict 0"
    )

    calculate_metrics(
        likely_players["total_points"],
        likely_players["pred_last"],
        "Previous match"
    )

    calculate_metrics(
        likely_players["total_points"],
        likely_players["pred_avg_3"],
        "3-match average"
    )

    calculate_metrics(
        likely_players["total_points"],
        likely_players["pred_avg_5"],
        "5-match average"
    )


if __name__ == "__main__":
    main()