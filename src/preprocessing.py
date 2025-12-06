"""Preprocessing utilities for time series forecasting datasets."""
from __future__ import annotations

from typing import Iterable, List

import pandas as pd


def clean_dataset(
    df: pd.DataFrame,
    *,
    date_column: str,
    numeric_columns: Iterable[str] | None = None,
    categorical_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Clean raw data by handling missing values, duplicates, and ordering.

    Rows missing the date column are removed to ensure proper chronological
    ordering for downstream modeling.
    """

    df = df.copy()
    df = df.dropna(subset=[date_column])
    df = df.drop_duplicates()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(by=date_column)

    numeric_columns = list(numeric_columns or [])
    categorical_columns = list(categorical_columns or [])

    for col in numeric_columns:
        if col in df:
            df[col] = df[col].fillna(df[col].mean())

    for col in categorical_columns:
        if col in df:
            mode = df[col].mode()
            if not mode.empty:
                df[col] = df[col].fillna(mode.iloc[0])

    return df.reset_index(drop=True)


def build_features(
    df: pd.DataFrame,
    *,
    target_column: str,
    lag_features: List[int] | None = None,
    rolling_windows: List[int] | None = None,
) -> pd.DataFrame:
    """Create lagged and rolling statistics features for forecasting models."""

    lag_features = lag_features or [1, 7]
    rolling_windows = rolling_windows or [3]

    df = df.copy()
    df = df.sort_values(df.columns[0])

    for lag in lag_features:
        df[f"{target_column}_lag_{lag}"] = df[target_column].shift(lag)

    for window in rolling_windows:
        df[f"{target_column}_rolling_mean_{window}"] = (
            df[target_column].rolling(window=window, min_periods=1).mean()
        )

    df = df.dropna().reset_index(drop=True)
    return df
