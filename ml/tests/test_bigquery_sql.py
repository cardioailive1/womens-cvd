"""The study SQL is valid BigQuery SQL (MCP runs it through the BigQuery client).

The pipeline runs with dialect=bigquery on the synthetic OMOP data: every query is parsed by
sqlglot's BigQuery grammar (a parse error fails the test), then translated to DuckDB to run, so the
results are checked too. Datasets use the shipped names (OMOP_v6_1, mcp)."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from mcp_adapter.config import StudyConfig
from tests.synthetic_omop import load_into_duckdb

sqlglot = pytest.importorskip("sqlglot")
ROOT = Path(__file__).resolve().parents[1]


class BigQuerySQLOnDuckDB:
    dialect = "bigquery"

    def __init__(self, con):
        self.con, self.queries = con, 0

    def sql(self, q):
        self.queries += 1
        tree = sqlglot.parse_one(q, read="bigquery", error_level=sqlglot.ErrorLevel.RAISE)
        return self.con.execute(tree.sql(dialect="duckdb")).df()

    def register(self, df, name):
        self.con.register(name, df)


def test_pipeline_sql_is_bigquery(tmp_path):
    from annotate.phenotype import fetch_context
    from mcp_adapter.pipeline import build_pretraining_pool, build_study, run_feasibility
    con = duckdb.connect()
    load_into_duckdb(con, schema="OMOP_v6_1")
    con.execute("ALTER TABLE OMOP_v6_1.condition_occurrence ADD COLUMN condition_status_concept_id INTEGER DEFAULT 32902")
    be = BigQuerySQLOnDuckDB(con)
    cfg = StudyConfig.load(ROOT / "config/study.yaml", imaging_required=False, report_dir=str(tmp_path / "rep"))
    assert cfg.dialect == "bigquery" and cfg.cdm_schema == "OMOP_v6_1" and cfg.results_schema
    cfg.mace_primary.update(mode="omop")
    run_feasibility(be, cfg)
    t = build_study(be, cfg, tmp_path / "t")
    again = build_study(be, cfg, tmp_path / "t2")
    assert len(t["cohort"]) and (t["cohort"].set_index("person_id").split
                                  == again["cohort"].set_index("person_id").split).all()
    pool = build_pretraining_pool(be, cfg, t["cohort"], tmp_path / "pool")
    assert len(pool["pool"])
    ctx = fetch_context(be, cfg, t["cohort"], t["cohort"].person_id.head(20))
    assert len(ctx["diagnoses"])
    assert be.queries >= 30
