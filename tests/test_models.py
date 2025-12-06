import pandas as pd

from src.models.training import (
    evaluate_regression_model,
    time_series_train_test_split,
    train_regression_model,
)


def test_time_series_train_test_split_preserves_order():
    X = pd.DataFrame({"feat": range(10)})
    y = pd.Series(range(10))

    split = time_series_train_test_split(X, y, test_size=0.3)

    assert len(split.X_train) == 7
    assert split.X_train.index[0] == 0
    assert split.X_test.index[0] == 7


def test_train_and_evaluate_regression_model():
    # simple increasing series where lag is predictive
    data = pd.DataFrame({
        "lag": [0, 1, 2, 3, 4, 5],
        "target": [1, 2, 3, 4, 5, 6],
    })
    X = data[["lag"]]
    y = data["target"]

    split = time_series_train_test_split(X, y, test_size=0.33)
    model = train_regression_model(split.X_train, split.y_train)
    mae, rmse = evaluate_regression_model(model, split.X_test, split.y_test)

    assert mae >= 0
    assert rmse >= 0
