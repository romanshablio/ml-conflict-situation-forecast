"""Utilities for loading and validating raw data for forecasting tasks."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd


def load_csv(
    file_path: str | Path,
    *,
    date_column: Optional[str] = None,
    column_mapping: Optional[Dict[str, str]] = None,
    drop_na_subset: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load a CSV file into a DataFrame with optional cleaning steps.

    Args:
        file_path: Path to the CSV file.
        date_column: Optional column name to parse as dates.
        column_mapping: Optional mapping from raw to canonical column names.
        drop_na_subset: Optional column list to drop rows with missing values.

    Returns:
        A pandas DataFrame with standardized columns and optional NA handling.
    """

    parse_dates = [date_column] if date_column else None
    df = pd.read_csv(file_path, parse_dates=parse_dates)

    if column_mapping:
        df = df.rename(columns=column_mapping)

    if drop_na_subset:
        df = df.dropna(subset=list(drop_na_subset))

    return df.reset_index(drop=True)


def merge_sources(frames: Iterable[pd.DataFrame], *, on: Optional[str] = None) -> pd.DataFrame:
    """Merge multiple data sources into a single DataFrame.

    Args:
        frames: Iterable of pandas DataFrames to combine.
        on: Optional join key. If provided, a left join is used. Otherwise
            the frames are concatenated row-wise.

    Returns:
        Combined DataFrame preserving original ordering when concatenating.
    """

    frames = list(frames)
    if not frames:
        raise ValueError("No data frames provided for merging.")

    if on:
        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on=on, how="left")
        return merged

    return pd.concat(frames, ignore_index=True)
