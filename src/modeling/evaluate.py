from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = mean_squared_error(
        actual,
        predicted
    ) ** 0.5

    return mae, rmse


def print_metrics(label, actual, predicted):
    mae, rmse = calculate_metrics(
        actual,
        predicted
    )

    print(label)
    print("-" * 50)
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")

    return mae, rmse