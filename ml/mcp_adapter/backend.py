"""SQL backends: Spark (MCP workbench) and DuckDB (local tests on synthetic OMOP).

The adapter writes portable ANSI SQL and calls `days_between` for the one construct whose
syntax differs between Spark SQL and DuckDB. Everything returns pandas DataFrames; queries
are always filtered to the cohort before results reach Python.
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd


class Backend(Protocol):
    dialect: str

    def sql(self, query: str) -> pd.DataFrame: ...
    def register(self, df: pd.DataFrame, name: str) -> None: ...


def days_between(dialect: str, later: str, earlier: str) -> str:
    """SQL expression for (later - earlier) in whole days."""
    if dialect == "spark":
        return f"datediff(CAST({later} AS DATE), CAST({earlier} AS DATE))"
    if dialect == "duckdb":
        return f"date_diff('day', CAST({earlier} AS DATE), CAST({later} AS DATE))"
    raise ValueError(f"unknown dialect {dialect}")


class SparkBackend:
    """Use inside the MCP workbench: `SparkBackend(spark)` with the session MCP provides."""

    dialect = "spark"

    def __init__(self, spark):
        self.spark = spark

    def sql(self, query: str) -> pd.DataFrame:
        return self.spark.sql(query).toPandas()

    def register(self, df: pd.DataFrame, name: str) -> None:
        self.spark.createDataFrame(df).createOrReplaceTempView(name)


class DuckDBBackend:
    dialect = "duckdb"

    def __init__(self, con=None, path: str = ":memory:"):
        import duckdb
        self.con = con or duckdb.connect(path)

    def sql(self, query: str) -> pd.DataFrame:
        return self.con.execute(query).df()

    def register(self, df: pd.DataFrame, name: str) -> None:
        self.con.register(name, df)
