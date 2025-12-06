"""Utilities for loading and validating raw data for forecasting tasks."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator


logger = logging.getLogger(__name__)


@dataclass
class RawRecord:
    """Representation of an unvalidated raw row from a data source."""

    date: str
    target: float
    location: str | None = None


class ValidatedRecord(BaseModel):
    """Strongly typed representation of a cleaned record."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    date: pd.Timestamp
    target: float
    location: str | None = None

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, value: object) -> pd.Timestamp:
        parsed = pd.to_datetime(value)
        if pd.isna(parsed):
            msg = "Date values must be coercible to datetime."
            raise ValueError(msg)
        return parsed

    @field_validator("target", mode="before")
    @classmethod
    def coerce_target(cls, value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            msg = "Target values must be numeric."
            raise ValueError(msg) from exc


def _normalize_columns(
    df: pd.DataFrame,
    *,
    column_mapping: Optional[Dict[str, str]] = None,
    drop_na_subset: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    if column_mapping:
        df = df.rename(columns=column_mapping)

    if drop_na_subset:
        df = df.dropna(subset=list(drop_na_subset))

    return df.reset_index(drop=True)


def _parse_date_columns(date_column: Optional[str | Sequence[str]]) -> list[str] | None:
    if date_column is None:
        return None
    if isinstance(date_column, str):
        return [date_column]
    return list(date_column)


def load_csv(
    file_path: str | Path,
    *,
    date_column: Optional[str | Sequence[str]] = None,
    column_mapping: Optional[Dict[str, str]] = None,
    drop_na_subset: Optional[Iterable[str]] = None,
    dtype_mapping: Optional[Mapping[str, str | type]] = None,
) -> pd.DataFrame:
    """Load a CSV file into a DataFrame with optional cleaning steps."""

    parse_dates = _parse_date_columns(date_column)
    df = pd.read_csv(file_path, parse_dates=parse_dates)
    df = _normalize_columns(df, column_mapping=column_mapping, drop_na_subset=drop_na_subset)

    if dtype_mapping:
        df = coerce_dtypes(df, dtype_mapping)

    return df


def load_parquet(
    file_path: str | Path,
    *,
    date_column: Optional[str | Sequence[str]] = None,
    column_mapping: Optional[Dict[str, str]] = None,
    drop_na_subset: Optional[Iterable[str]] = None,
    dtype_mapping: Optional[Mapping[str, str | type]] = None,
) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame with optional cleaning steps."""

    df = pd.read_parquet(file_path)

    parse_dates = _parse_date_columns(date_column)
    if parse_dates:
        for column in parse_dates:
            if column in df.columns:
                df[column] = pd.to_datetime(df[column])

    df = _normalize_columns(df, column_mapping=column_mapping, drop_na_subset=drop_na_subset)

    if dtype_mapping:
        df = coerce_dtypes(df, dtype_mapping)

    return df


def validate_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Ensure required columns are present in the dataframe."""

    missing = set(required_columns) - set(df.columns)
    if missing:
        msg = f"Missing required columns: {sorted(missing)}"
        raise ValueError(msg)


def coerce_dtypes(df: pd.DataFrame, dtype_mapping: Mapping[str, str | type]) -> pd.DataFrame:
    """Coerce dataframe columns into the provided dtypes."""

    df = df.copy()
    for column, dtype in dtype_mapping.items():
        if column not in df.columns:
            msg = f"Column '{column}' not found for dtype coercion."
            raise ValueError(msg)

        try:
            df[column] = df[column].astype(dtype)
        except (TypeError, ValueError) as exc:
            msg = f"Failed to convert column '{column}' to {dtype}."
            raise ValueError(msg) from exc

    return df


def validate_records(df: pd.DataFrame) -> list[ValidatedRecord]:
    """Validate rows using the RawRecord dataclass and ValidatedRecord model."""

    records: list[ValidatedRecord] = []
    for row in df.to_dict(orient="records"):
        try:
            raw = RawRecord(
                date=row["date"],
                target=row["target"],
                location=row.get("location"),
            )
        except KeyError as exc:
            msg = "Required keys missing from raw record."
            raise ValueError(msg) from exc

        try:
            records.append(ValidatedRecord.model_validate(raw.__dict__))
        except ValidationError as exc:
            msg = "Validation failed for record."
            raise ValueError(msg) from exc

    return records


def log_data_stats(df: pd.DataFrame, *, logger_: Optional[logging.Logger] = None) -> None:
    """Log simple statistics about the dataset for observability."""

    active_logger = logger_ or logger
    active_logger.info("Loaded %d rows and %d columns", len(df), len(df.columns))
    missing = df.isna().sum().to_dict()
    active_logger.info("Missing values per column: %s", missing)
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    active_logger.info("Column dtypes: %s", dtypes)


def merge_sources(frames: Iterable[pd.DataFrame], *, on: Optional[str] = None) -> pd.DataFrame:
    """Merge multiple data sources into a single DataFrame."""

    frames = list(frames)
    if not frames:
        raise ValueError("No data frames provided for merging.")

    if on:
        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on=on, how="left")
        return merged

    return pd.concat(frames, ignore_index=True)
