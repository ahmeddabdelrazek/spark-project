import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from pyspark_job import clean_data

SCHEMA = StructType(
    [
        StructField("name", StringType(), True),
        StructField("amount", DoubleType(), True),
    ]
)


@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder.master("local[1]")
        .appName("pyspark-job-tests")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def make_df(spark, rows):
    return spark.createDataFrame(rows, schema=SCHEMA)


def test_valid_records_are_kept(spark):
    df = make_df(spark, [("Alice", 100.0), ("Bob", 50.0)])

    result = clean_data(df)

    names = sorted(row["name"] for row in result.collect())
    assert result.count() == 2
    assert names == ["Alice", "Bob"]


def test_amount_less_than_or_equal_to_zero_is_removed(spark):
    df = make_df(
        spark,
        [
            ("Alice", 100.0),
            ("Bob", 0.0),
            ("Carol", -10.0),
            ("Dave", -0.01),
        ],
    )

    result = clean_data(df)

    assert [row["name"] for row in result.collect()] == ["Alice"]
    assert result.filter("amount <= 0").count() == 0


def test_null_names_are_removed(spark):
    df = make_df(spark, [("Alice", 100.0), (None, 200.0), (None, 5.0)])

    result = clean_data(df)

    assert result.count() == 1
    assert result.filter("name IS NULL").count() == 0
    assert result.collect()[0]["name"] == "Alice"


def test_amount_with_tax_is_calculated_correctly(spark):
    df = make_df(spark, [("Alice", 100.0), ("Bob", 50.0), ("Carol", 0.5)])

    result = clean_data(df)

    taxed = {row["name"]: row["amount_with_tax"] for row in result.collect()}
    assert taxed["Alice"] == pytest.approx(120.0)
    assert taxed["Bob"] == pytest.approx(60.0)
    assert taxed["Carol"] == pytest.approx(0.6)


def test_amount_with_tax_column_is_added(spark):
    df = make_df(spark, [("Alice", 100.0)])

    result = clean_data(df)

    assert result.columns == ["name", "amount", "amount_with_tax"]


def test_mixed_input_only_valid_rows_remain(spark):
    df = make_df(
        spark,
        [
            ("Alice", 100.0),   # valid
            ("Bob", -5.0),      # negative amount
            (None, 80.0),       # null name
            ("Carol", 0.0),     # zero amount
            ("Dave", 10.0),     # valid
        ],
    )

    result = clean_data(df)

    rows = {row["name"]: row["amount_with_tax"] for row in result.collect()}
    assert set(rows) == {"Alice", "Dave"}
    assert rows["Alice"] == pytest.approx(120.0)
    assert rows["Dave"] == pytest.approx(12.0)


def test_empty_dataframe_returns_empty_result(spark):
    df = make_df(spark, [])

    result = clean_data(df)

    assert result.count() == 0
    assert "amount_with_tax" in result.columns
