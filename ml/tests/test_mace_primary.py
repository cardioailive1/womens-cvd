"""Optional MACE rule: MI / stroke must also be a primary diagnosis in Mayo FACT_DIAGNOSIS."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from mcp_adapter.backend import DuckDBBackend
from mcp_adapter.config import StudyConfig
from mcp_adapter.labels import require_primary
from mcp_adapter.pipeline import build_study
from tests.synthetic_omop import load_into_duckdb

ROOT = Path(__file__).resolve().parents[1]


def test_require_primary_unit():
    ev = pd.DataFrame({"person_id": [1, 1, 2, 3, 3], "grp": ["out_mi", "out_heart_failure", "out_stroke", "out_mi", "cond_SCAD"],
                       "event_date": ["2020-01-10", "2020-01-10", "2020-02-01", "2020-03-01", "2020-03-01"]})
    prim = pd.DataFrame({"person_id": [1, 2], "grp": ["out_mi", "out_stroke"], "event_date": ["2020-01-11", "2020-02-05"]})
    out, dropped = require_primary(ev, prim, match_days=1)
    assert dropped == 2                                     # person 2 too far apart, person 3 no primary
    assert set(zip(out.person_id, out.grp)) == {(1, "out_mi"), (1, "out_heart_failure"), (3, "cond_SCAD")}


def test_primary_rule_in_the_pipeline(tmp_path):
    con = duckdb.connect(); load_into_duckdb(con)
    be = DuckDBBackend(con)
    cfg = StudyConfig.load(ROOT / "config/study.yaml", dialect="duckdb", report_dir=str(tmp_path / "rep"),
                           imaging_required=False)
    assert cfg.mace_primary["mode"] == "off"
    base = build_study(be, cfg, tmp_path / "off")["labels"]

    # FACT_DIAGNOSIS mirror of every MI / stroke code, flagged primary for even person_ids only
    mi = con.execute(f"""
SELECT co.person_id, co.condition_start_date AS dx_date, c.concept_code AS code
FROM omop_cdm.condition_occurrence co JOIN omop_cdm.concept c ON c.concept_id = co.condition_source_concept_id
WHERE c.concept_code LIKE 'I21%' OR c.concept_code LIKE 'I22%' OR c.concept_code LIKE 'I6%'""").df()
    assert len(mi)
    fd = pd.DataFrame({"PATIENT_DK": mi.person_id + 500000, "DIAGNOSIS_DTM": mi.dx_date, "DIAGNOSIS_CODE": mi.code,
                       "PRIMARY_DIAGNOSIS_FLAG": ["Y" if p % 2 == 0 else "N" for p in mi.person_id]})
    dims = con.execute("SELECT person_id + 500000 AS PATIENT_DK, person_source_value AS PATIENT_CLINIC_NUMBER FROM omop_cdm.person").df()
    con.register("fd", fd); con.register("dims", dims)
    con.execute("CREATE TABLE fact_diagnosis AS SELECT * FROM fd; CREATE TABLE dim_patient AS SELECT * FROM dims")
    cfg.mace_primary.update(mode="fact_diagnosis", table="fact_diagnosis")
    cfg.dim_patient.update(table="dim_patient")
    on = build_study(be, cfg, tmp_path / "on")["labels"]
    merged = base.merge(on, on="person_id", suffixes=("_off", "_on"))
    assert (merged.y_mace_on <= merged.y_mace_off).all()               # the rule only removes events
    lost = merged[(merged.y_mace_off == 1) & (merged.y_mace_on == 0)]
    assert len(lost) and (lost.person_id % 2 == 1).all()                # only women without a primary flag
    rep = pd.read_csv(tmp_path / "rep" / "04b_mace_primary_diagnosis.csv")
    assert rep["item"].iloc[0].startswith("MI / stroke codes dropped")
