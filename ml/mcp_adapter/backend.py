"""SQL backends: BigQuery (MCP warehouse), Spark and DuckDB (local tests on synthetic OMOP).

MCP's OMOP guide (28 Oct 2026) places the warehouse in BigQuery: Mayo-native FACT_* / DIM_*
tables in dataset `mcp`, OMOP CDM v5.4 in `OMOP_v6_1`, and a project work dataset for saved
tables. The adapter writes portable SQL and calls `days_between` / `regex_match` for the
constructs whose syntax differs between dialects. Everything returns pandas DataFrames; queries
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
    if dialect == "bigquery":
        return f"DATE_DIFF(CAST({later} AS DATE), CAST({earlier} AS DATE), DAY)"
    raise ValueError(f"unknown dialect {dialect}")


def regex_match(dialect: str, expr: str, pattern: str) -> str:
    """SQL predicate: `expr` contains a match of the regular expression `pattern`."""
    pat = pattern.replace("\\", "\\\\").replace("'", "\\'") if dialect == "bigquery" else pattern.replace("'", "''")
    if dialect == "duckdb":
        return f"regexp_matches({expr}, '{pat}')"
    if dialect == "spark":
        return f"{expr} RLIKE '{pat}'"
    if dialect == "bigquery":
        return f"REGEXP_CONTAINS({expr}, '{pat}')"
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


class BigQueryBackend:
    """MCP warehouse in BigQuery. Helper tables registered by the pipeline (the cohort's person
    ids and index dates, code lists) are written to the project's work dataset (MCP, 28 Oct
    2026: projects can write saved tables there) and unqualified table names in queries resolve
    to it. Row-level data stays inside the workspace.

        from google.cloud import bigquery
        be = BigQueryBackend(bigquery.Client(), work_dataset="<project>.cardioai_work")
    """

    dialect = "bigquery"

    def __init__(self, client=None, work_dataset: str | None = None, location: str | None = None):
        if client is None:
            from google.cloud import bigquery
            client = bigquery.Client()
        if not work_dataset:
            raise ValueError("BigQueryBackend needs work_dataset (platform.results_schema in study.yaml)")
        if work_dataset.count(".") == 0:
            work_dataset = f"{client.project}.{work_dataset}"
        self.client, self.work_dataset, self.location = client, work_dataset.strip("`"), location

    def sql(self, query: str) -> pd.DataFrame:
        from google.cloud import bigquery
        job = self.client.query(query, location=self.location,
                                job_config=bigquery.QueryJobConfig(default_dataset=self.work_dataset))
        return job.result().to_dataframe()

    def register(self, df: pd.DataFrame, name: str) -> None:
        from google.cloud import bigquery
        cfg = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        self.client.load_table_from_dataframe(df, f"{self.work_dataset}.{name}", job_config=cfg,
                                              location=self.location).result()

    def save_table(self, df: pd.DataFrame, name: str) -> str:
        """Write a table to keep (e.g. the locked validation/test split). Returns its full name."""
        self.register(df, name)
        return f"{self.work_dataset}.{name}"


def make_backend(cfg, client=None):
    """The backend for cfg.dialect inside the MCP workspace (BigQuery or Spark).

    BigQuery: pass the client MCP's notebook already created (`bigquery.Client()` in
    My_First_Query.ipynb) to reuse its project and credentials; otherwise one is created
    for platform.bigquery_project (default: the workspace's default project)."""
    if cfg.dialect == "bigquery":
        if client is None:
            from google.cloud import bigquery
            project = getattr(cfg, "bigquery_project", None)
            client = bigquery.Client(project=project) if project else bigquery.Client()
        return BigQueryBackend(client, work_dataset=cfg.results_schema,
                               location=getattr(cfg, "bigquery_location", None))
    if cfg.dialect == "spark":
        from pyspark.sql import SparkSession
        return SparkBackend(SparkSession.builder.getOrCreate())
    if cfg.dialect == "duckdb":
        return DuckDBBackend()
    raise ValueError(f"unknown dialect {cfg.dialect}")


class DuckDBBackend:
    dialect = "duckdb"

    def __init__(self, con=None, path: str = ":memory:"):
        import duckdb
        self.con = con or duckdb.connect(path)

    def sql(self, query: str) -> pd.DataFrame:
        return self.con.execute(query).df()

    def register(self, df: pd.DataFrame, name: str) -> None:
        self.con.register(name, df)
