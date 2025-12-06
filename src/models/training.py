"""Model utilities for training and evaluating forecasting models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class TrainTestSplit:
    """Container for train/test splits preserving chronological order."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def time_series_train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = 0.2,
) -> TrainTestSplit:
    """Chronologically split features and target for forecasting validation."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    split_index = int(len(features) * (1 - test_size))
    if split_index == 0 or split_index == len(features):
        raise ValueError("Dataset too small for requested test_size")

    return TrainTestSplit(
        X_train=features.iloc[:split_index],
        X_test=features.iloc[split_index:],
        y_train=target.iloc[:split_index],
        y_test=target.iloc[split_index:],
    )


def train_regression_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """Train a simple baseline regression model suitable for forecasting."""

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_regression_model(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[float, float]:
    """Evaluate regression model returning MAE and RMSE."""

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    return mae, rmse
