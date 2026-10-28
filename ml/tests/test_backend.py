"""SQL dialect helpers and the BigQuery backend (MCP warehouse, per the OMOP guide of 28 Oct 2026)."""
from __future__ import annotations

import sys
import types

import duckdb
import pandas as pd
import pytest

from mcp_adapter.backend import BigQueryBackend, days_between, make_backend, regex_match


def test_days_between_dialects():
    assert days_between("bigquery", "a.d", "b.d") == "DATE_DIFF(CAST(a.d AS DATE), CAST(b.d AS DATE), DAY)"
    assert days_between("spark", "a.d", "b.d").startswith("datediff(")
    with pytest.raises(ValueError):
        days_between("oracle", "a", "b")


def test_regex_match_runs_in_duckdb_and_quotes_safely():
    con = duckdb.connect()
    sql = regex_match("duckdb", "x", "FETAL|TEE")
    got = con.execute(f"SELECT x, {sql} AS m FROM (VALUES ('FETAL ECHO'), ('TTE ECHO'), ('TEE')) t(x)").df()
    assert got.m.tolist() == [True, False, True]
    assert regex_match("bigquery", "x", "it's\\b") == "REGEXP_CONTAINS(x, 'it\\'s\\\\b')"
    assert regex_match("spark", "x", "A|B") == "x RLIKE 'A|B'"


class _FakeJob:
    def __init__(self, df): self.df = df
    def result(self): return self
    def to_dataframe(self): return self.df


class _FakeClient:
    project = "proj-1"

    def __init__(self):
        self.queries, self.loads = [], []

    def query(self, q, location=None, job_config=None):
        self.queries.append((q, job_config.default_dataset))
        return _FakeJob(pd.DataFrame({"n": [1]}))

    def load_table_from_dataframe(self, df, dest, job_config=None, location=None):
        self.loads.append((dest, len(df), job_config.write_disposition))
        return _FakeJob(None)


@pytest.fixture
def fake_bigquery(monkeypatch):
    """A stand-in for google.cloud.bigquery so the backend can be tested without the package."""
    bq = types.ModuleType("google.cloud.bigquery")

    class QueryJobConfig:
        def __init__(self, default_dataset=None): self.default_dataset = default_dataset

    class LoadJobConfig:
        def __init__(self, write_disposition=None): self.write_disposition = write_disposition

    bq.QueryJobConfig, bq.LoadJobConfig = QueryJobConfig, LoadJobConfig
    cloud = types.ModuleType("google.cloud"); cloud.bigquery = bq
    google = sys.modules.get("google") or types.ModuleType("google"); google.cloud = cloud
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.bigquery", bq)
    return bq


def test_bigquery_backend_uses_the_work_dataset(fake_bigquery):
    c = _FakeClient()
    be = BigQueryBackend(c, work_dataset="cardioai_work")
    assert be.dialect == "bigquery" and be.work_dataset == "proj-1.cardioai_work"
    be.register(pd.DataFrame({"person_id": [1, 2]}), "hcman_cohort")
    assert c.loads == [("proj-1.cardioai_work.hcman_cohort", 2, "WRITE_TRUNCATE")]
    assert be.sql("SELECT 1 AS n").n.tolist() == [1]
    assert c.queries[0][1] == "proj-1.cardioai_work"          # unqualified helper tables resolve here
    assert be.save_table(pd.DataFrame({"person_id": [1]}), "hcman_splits") == "proj-1.cardioai_work.hcman_splits"


def test_bigquery_backend_needs_a_work_dataset(fake_bigquery):
    with pytest.raises(ValueError, match="work_dataset"):
        BigQueryBackend(_FakeClient(), work_dataset=None)


def test_make_backend_duckdb():
    cfg = types.SimpleNamespace(dialect="duckdb", results_schema=None)
    assert make_backend(cfg).dialect == "duckdb"


def test_lock_splits_keeps_evaluation_women_out_of_training():
    from mcp_adapter.backend import DuckDBBackend
    from mcp_adapter.splits import lock_splits
    be = DuckDBBackend()
    cfg = types.SimpleNamespace(split_lock_table="hcman_locked_splits")
    first = pd.DataFrame({"person_id": [1, 2, 3], "split": ["train", "val", "test"]})
    out, n = lock_splits(be, cfg, first)
    assert out.split.tolist() == ["train", "val", "test"] and n["newly assigned"] == 3
    # rebuild after a data refresh: the new draw would move 2 and 3 into training; 4 is new
    redraw = pd.DataFrame({"person_id": [1, 2, 3, 4], "split": ["val", "train", "train", "test"]})
    out, n = lock_splits(be, cfg, redraw)
    assert out.set_index("person_id").split.to_dict() == {1: "train", 2: "val", 3: "test", 4: "test"}
    assert n == {"reused from the locked split": 3, "newly assigned": 1, "kept locked where the new draw differed": 3}
    saved = be.sql("SELECT * FROM hcman_locked_splits ORDER BY person_id")
    assert saved.person_id.tolist() == [1, 2, 3, 4]


def test_lock_splits_off():
    from mcp_adapter.splits import lock_splits
    df = pd.DataFrame({"person_id": [1], "split": ["train"]})
    assert lock_splits(None, types.SimpleNamespace(split_lock_table=None), df) == (df, {})


def test_make_backend_reuses_the_notebook_client(fake_bigquery):
    c = _FakeClient()
    cfg = types.SimpleNamespace(dialect="bigquery", results_schema="cardioai_work",
                                bigquery_project=None, bigquery_location="US")
    be = make_backend(cfg, client=c)
    assert be.client is c and be.location == "US" and be.work_dataset == "proj-1.cardioai_work"


def test_real_bigquery_client_builds_the_jobs():
    """With the real google-cloud-bigquery package (no network): the query carries the work dataset
    as default dataset, and helper tables load into it, replacing any earlier copy."""
    bq = pytest.importorskip("google.cloud.bigquery")
    from unittest import mock
    from google.auth.credentials import AnonymousCredentials
    client = bq.Client(project="mcp-proj", credentials=AnonymousCredentials())
    be = BigQueryBackend(client, work_dataset="cardioai_work", location="US")
    with mock.patch.object(client, "query") as q, mock.patch.object(client, "load_table_from_dataframe") as ld:
        q.return_value.result.return_value.to_dataframe.return_value = pd.DataFrame({"n": [3]})
        assert be.sql("SELECT COUNT(*) AS n FROM OMOP_v6_1.person").n.tolist() == [3]
        jc = q.call_args.kwargs["job_config"]
        assert str(jc.default_dataset) == "mcp-proj.cardioai_work" and q.call_args.kwargs["location"] == "US"
        be.register(pd.DataFrame({"person_id": [1]}), "hcman_cohort")
        dest, = ld.call_args.args[1:2]
        assert dest == "mcp-proj.cardioai_work.hcman_cohort"
        assert ld.call_args.kwargs["job_config"].write_disposition == "WRITE_TRUNCATE"
