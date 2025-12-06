import pandas as pd

from src.data_ingestion import load_csv, merge_sources


def test_load_csv_parses_dates_and_renames(tmp_path):
    csv_path = tmp_path / "data.csv"
    data = """date,value,category\n2020-01-01,1,A\n2020-01-02,2,B\n"""
    csv_path.write_text(data)

    df = load_csv(
        csv_path,
        date_column="date",
        column_mapping={"value": "count"},
        drop_na_subset=["value"],
    )

    assert list(df.columns) == ["date", "count", "category"]
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df.loc[0, "count"] == 1


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
