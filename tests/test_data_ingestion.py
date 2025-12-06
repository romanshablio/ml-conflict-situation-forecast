import pandas as pd
import pytest

from src.data_ingestion import (
    ValidatedRecord,
    coerce_dtypes,
    load_csv,
    load_parquet,
    log_data_stats,
    merge_sources,
    validate_columns,
    validate_records,
)


def test_load_csv_parses_dates_and_coerces_dtypes(tmp_path):
    csv_path = tmp_path / "data.csv"
    data = """date,value,category\n2020-01-01,1,A\n2020-01-02,2,B\n"""
    csv_path.write_text(data)

    df = load_csv(
        csv_path,
        date_column="date",
        column_mapping={"value": "count"},
        drop_na_subset=["value"],
        dtype_mapping={"count": "int64"},
    )

    assert list(df.columns) == ["date", "count", "category"]
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df.loc[0, "count"] == 1
    assert str(df["count"].dtype) == "int64"


def test_load_parquet_handles_dates_and_mapping(tmp_path):
    df = pd.DataFrame({"date": ["2020-01-01"], "target": ["3"], "value": [1]})
    parquet_path = tmp_path / "sample.parquet"
    df.to_parquet(parquet_path)

    loaded = load_parquet(
        parquet_path,
        date_column="date",
        column_mapping={"value": "feature"},
        dtype_mapping={"target": float},
    )

    assert pd.api.types.is_datetime64_any_dtype(loaded["date"])
    assert "feature" in loaded.columns
    assert loaded["target"].iloc[0] == 3.0


def test_validate_columns_raises_on_missing_columns():
    df = pd.DataFrame({"a": [1]})

    with pytest.raises(ValueError):
        validate_columns(df, ["a", "b"])


def test_coerce_dtypes_errors_on_invalid_cast():
    df = pd.DataFrame({"value": ["abc"]})

    with pytest.raises(ValueError):
        coerce_dtypes(df, {"value": "int64"})


def test_validate_records_returns_models():
    df = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "target": ["1", 2],
            "location": ["X", None],
        }
    )

    records = validate_records(df)

    assert all(isinstance(record, ValidatedRecord) for record in records)
    assert records[0].date.year == 2020
    assert records[1].target == 2.0


def test_validate_records_raises_on_bad_data():
    df = pd.DataFrame({"date": ["not-a-date"], "target": [1]})

    with pytest.raises(ValueError):
        validate_records(df)


def test_log_data_stats_outputs_messages(caplog):
    df = pd.DataFrame({"a": [1, None], "b": ["x", "y"]})

    with caplog.at_level("INFO"):
        log_data_stats(df)

    assert any("Loaded" in message for message in caplog.text.splitlines())
    assert any("Missing values" in message for message in caplog.text.splitlines())


def test_merge_sources_concatenates_by_default():
    df1 = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
    df2 = pd.DataFrame({"id": [3], "value": [30]})

    merged = merge_sources([df1, df2])
    assert len(merged) == 3
    assert list(merged["id"]) == [1, 2, 3]


def test_merge_sources_left_join():
    left = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
    right = pd.DataFrame({"id": [1, 2], "extra": [100, 200]})

    merged = merge_sources([left, right], on="id")
    assert "extra" in merged.columns
    assert merged.loc[merged["id"] == 1, "extra"].iloc[0] == 100
