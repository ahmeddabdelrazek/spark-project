"""PySpark data-cleaning job.

Contains the ``clean_data`` transformation, which is kept free of any I/O so it
can be unit tested in isolation.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

TAX_RATE = 1.20


def clean_data(df: DataFrame) -> DataFrame:
    """Clean a transactions DataFrame.

    Steps:
        1. Drop rows where ``amount`` is less than or equal to zero.
        2. Drop rows where ``name`` is NULL.
        3. Add ``amount_with_tax`` computed as ``amount * 1.20``.

    Rows with a NULL ``amount`` are also dropped, because the comparison
    ``amount > 0`` evaluates to NULL and the filter discards them.

    Args:
        df: Input DataFrame with at least the columns ``name`` and ``amount``.

    Returns:
        A new DataFrame containing only valid rows plus the
        ``amount_with_tax`` column.
    """
    return (
        df.filter(F.col("amount") > 0)
        .filter(F.col("name").isNotNull())
        .withColumn("amount_with_tax", F.col("amount") * F.lit(TAX_RATE))
    )
