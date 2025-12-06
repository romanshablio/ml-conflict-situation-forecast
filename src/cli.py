"""Command-line entry point for running a demo forecasting pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_ingestion import load_csv
from src.models.training import (
    evaluate_regression_model,
    time_series_train_test_split,
    train_regression_model,
)
from src.preprocessing import build_features, clean_dataset


def _build_sample_data() -> pd.DataFrame:
    """Construct a small synthetic dataset for quick experiments."""

    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=21, freq="D"),
            "target": [
                5,
                6,
                5,
                7,
                8,
                7,
                9,
                10,
                12,
                11,
                13,
                15,
                16,
                18,
                17,
                19,
                20,
                21,
                22,
                23,
                25,
            ],
        }
    )


def _prepare_dataset(csv_path: Path | None) -> pd.DataFrame:
    if csv_path is None:
        data = _build_sample_data()
    else:
        data = load_csv(csv_path, date_column="date")

    cleaned = clean_dataset(data, date_column="date", numeric_columns=["target"])
    return build_features(cleaned, target_column="target")


def _train_and_evaluate(features: pd.DataFrame) -> tuple[float, float]:
    feature_columns = [
        column
        for column in features.columns
        if column not in {"target", "date"}
    ]
    split = time_series_train_test_split(
        features[feature_columns],
        features["target"],
        test_size=0.25,
    )
    model = train_regression_model(split.X_train, split.y_train)
    return evaluate_regression_model(model, split.X_test, split.y_test)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a quick end-to-end demo that cleans data, engineers features,"
            " trains a forecasting model, and reports validation metrics."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        help=(
            "Optional path to a CSV file with 'date' and 'target' columns. If"
            " omitted, a synthetic dataset is used."
        ),
    )
    args = parser.parse_args()

    features = _prepare_dataset(args.data)
    mae, rmse = _train_and_evaluate(features)
    print(f"Validation MAE: {mae:.3f}\nValidation RMSE: {rmse:.3f}")


if __name__ == "__main__":
    main()
