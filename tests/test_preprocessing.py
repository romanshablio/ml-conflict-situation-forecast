import pandas as pd

from src.preprocessing import build_features, clean_dataset


def test_clean_dataset_orders_and_fills_missing_values():
    raw = pd.DataFrame(
        {
            "date": ["2020-01-03", None, "2020-01-01", "2020-01-02"],
            "value": [3.0, 4.0, None, 2.0],
            "category": [None, "A", "B", None],
        }
    )

    cleaned = clean_dataset(
        raw,
        date_column="date",
        numeric_columns=["value"],
        categorical_columns=["category"],
    )

    assert cleaned["date"].iloc[0] == pd.to_datetime("2020-01-01")
    assert cleaned["value"].isna().sum() == 0
    assert cleaned["category"].isna().sum() == 0


def test_build_features_creates_lags_and_rolling_mean():
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=5, freq="D"), "target": [1, 2, 3, 4, 5]})
    features = build_features(df, target_column="target", lag_features=[1, 2], rolling_windows=[2])

    expected_columns = {
        "date",
        "target",
        "target_lag_1",
        "target_lag_2",
        "target_rolling_mean_2",
    }
    assert expected_columns.issubset(set(features.columns))
    assert len(features) == 3  # first two rows dropped because of lag 2
