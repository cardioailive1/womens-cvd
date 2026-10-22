"""OMOP CDM v5.4 extraction queries (portable Spark SQL / DuckDB).

Pass 1 (pre-cohort): concept resolution, feasibility counts, stratum-defining code events and
cardiac-evaluation anchors for women only.
Pass 2 (post-cohort): everything else, joined to the registered cohort table `hcman_cohort`
(person_id, index_date) and limited to each woman's lookback / outcome windows.

ICD-10-CM Z/R codes often land in the OMOP OBSERVATION table rather than
CONDITION_OCCURRENCE, so code events always union both tables on their *source* concept.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from . import concepts as C
from .backend import Backend, days_between
from .config import StudyConfig

ICD_VIEW = "hcman_icd_concepts"
COHORT_VIEW = "hcman_cohort"


def _q(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _code_predicate(codes: Iterable[str]) -> str:
    parts = []
    for c in codes:
        parts.append(f"concept_code LIKE {_q(c)}" if c.endswith("%") else f"concept_code = {_q(c)}")
    return "(" + " OR ".join(parts) + ")"


# ------------------------------------------------------------------ concept resolution
def resolve_icd_groups(be: Backend, cfg: StudyConfig, groups: dict[str, list[str]]) -> pd.DataFrame:
    """Resolve ICD-10-CM code patterns to concept_ids and register them as ICD_VIEW."""
    selects = [
        f"SELECT concept_id, concept_code, {_q(g)} AS grp FROM {cfg.vocab_schema}.concept "
        f"WHERE vocabulary_id = 'ICD10CM' AND {_code_predicate(codes)}"
        for g, codes in groups.items()
    ]
    df = be.sql("\nUNION ALL\n".join(selects))
    be.register(df, ICD_VIEW)
    return df


def resolve_loinc(be: Backend, cfg: StudyConfig) -> pd.DataFrame:
    rows = [(code, name) for name, spec in C.LABS.items() for code in spec["loinc"]]
    rows += [(code, name) for name, codes in C.VITALS.items() for code in codes]
    be.register(pd.DataFrame(rows, columns=["code", "item"]), "hcman_loinc_map")
    df = be.sql(
        f"SELECT c.concept_id, c.concept_code, m.item FROM {cfg.vocab_schema}.concept c "
        f"JOIN hcman_loinc_map m ON c.concept_code = m.code "
        f"WHERE c.vocabulary_id = 'LOINC'"
    )
    be.register(df, "hcman_loinc")
    return df


def lab_code_coverage(be: Backend, cfg: StudyConfig) -> pd.DataFrame:
    """Rows and women per LOINC code in the lab / vitals lists. A code with zero rows means the
    platform maps that test to a different LOINC: look it up and add it in concepts.py."""
    resolve_loinc(be, cfg)
    cdm = cfg.cdm_schema
    got = be.sql(f"""
SELECT l.concept_code AS loinc, COUNT(*) AS n_rows, COUNT(DISTINCT m.person_id) AS n_women
FROM {cdm}.measurement m JOIN hcman_loinc l ON l.concept_id = m.measurement_concept_id
JOIN {cdm}.person p ON p.person_id = m.person_id AND p.gender_concept_id = {C.FEMALE_GENDER_CONCEPT_ID}
GROUP BY l.concept_code""")
    want = pd.DataFrame([(item, code) for item, spec in C.LABS.items() for code in spec["loinc"]]
                        + [(item, code) for item, codes in C.VITALS.items() for code in codes],
                        columns=["item", "loinc"])
    out = want.merge(got, on="loinc", how="left").fillna({"n_rows": 0, "n_women": 0})
    return out.astype({"n_rows": int, "n_women": int}).sort_values(["item", "n_women"], ascending=[True, False])


# ------------------------------------------------------------------ pass 1
def _events_sql(cfg: StudyConfig, grp_filter: str, person_filter: str = "",
                full_history: bool = False) -> str:
    # full_history: no lower date bound — exclusions and reproductive history (e.g. a
    # preeclampsia years before the study window) must be visible.
    lower = "" if full_history else f"AND e.event_date >= CAST({_q(cfg.study_start)} AS DATE)"
    cdm = cfg.cdm_schema
    return f"""
SELECT e.person_id, g.grp, g.concept_code, CAST(e.event_date AS DATE) AS event_date,
       v.visit_concept_id
FROM (
    SELECT person_id, condition_source_concept_id AS src, condition_start_date AS event_date,
           visit_occurrence_id
    FROM {cdm}.condition_occurrence
    UNION ALL
    SELECT person_id, observation_source_concept_id, observation_date, visit_occurrence_id
    FROM {cdm}.observation
) e
JOIN {ICD_VIEW} g ON e.src = g.concept_id
JOIN {cdm}.person p ON p.person_id = e.person_id AND p.gender_concept_id = {C.FEMALE_GENDER_CONCEPT_ID}
LEFT JOIN {cdm}.visit_occurrence v ON v.visit_occurrence_id = e.visit_occurrence_id
{person_filter}
WHERE {grp_filter}
  {lower}
  AND e.event_date <= CAST({_q(cfg.study_end)} AS DATE)
"""


def feasibility_counts(be: Backend, cfg: StudyConfig) -> pd.DataFrame:
    """Distinct women per code group — aggregate only, safe to report after suppression."""
    sql = _events_sql(cfg, "1 = 1")
    return be.sql(f"SELECT grp, COUNT(DISTINCT person_id) AS n_women FROM ({sql}) t GROUP BY grp")


STRATUM_GROUPS = [
    "repro_pregnant_state", "repro_high_risk_pregnancy", "repro_delivery",
    "repro_preeclampsia", "repro_hellp", "repro_gestational_htn",
    "repro_menopause_natural", "repro_menopause_premature", "repro_menopause_surgical",
    "repro_pcos", "repro_endometriosis", "repro_sle", "repro_ra",
    "cond_SCAD", "cond_FMD", "cond_TAKOTSUBO", "cond_CMD_INOCA", "cond_PPCM",
    "cond_PREECLAMPSIA_CVD", "excl",
]


def stratum_events(be: Backend, cfg: StudyConfig) -> pd.DataFrame:
    grp_in = "g.grp IN (" + ", ".join(_q(g) for g in STRATUM_GROUPS) + ")"
    return be.sql(f"SELECT DISTINCT * FROM ({_events_sql(cfg, grp_in, full_history=True)}) t")


def cardiac_eval_events(be: Backend, cfg: StudyConfig) -> pd.DataFrame:
    """Dates of ECGs, echos (CPT4) and troponin/BNP draws, for women — index anchors."""
    cdm, voc = cfg.cdm_schema, cfg.vocab_schema
    loinc = ", ".join(_q(c) for c in C.CARDIAC_EVAL_LOINC)
    cpt = ", ".join(_q(c) for c in C.CARDIAC_EVAL_CPT)
    sql = f"""
SELECT DISTINCT person_id, event_date, kind FROM (
  SELECT m.person_id, CAST(m.measurement_date AS DATE) AS event_date, 'lab' AS kind
  FROM {cdm}.measurement m JOIN {voc}.concept c ON c.concept_id = m.measurement_concept_id
  WHERE c.vocabulary_id = 'LOINC' AND c.concept_code IN ({loinc})
  UNION ALL
  SELECT po.person_id, CAST(po.procedure_date AS DATE), 'proc'
  FROM {cdm}.procedure_occurrence po JOIN {voc}.concept c ON c.concept_id = po.procedure_concept_id
  WHERE c.vocabulary_id = 'CPT4' AND c.concept_code IN ({cpt})
) t
JOIN {cdm}.person p USING (person_id)
WHERE p.gender_concept_id = {C.FEMALE_GENDER_CONCEPT_ID}
  AND event_date >= CAST({_q(cfg.study_start)} AS DATE)
  AND event_date <= CAST({_q(cfg.study_end)} AS DATE)
"""
    return be.sql(sql)


def persons(be: Backend, cfg: StudyConfig, person_view: str | None = None) -> pd.DataFrame:
    cdm = cfg.cdm_schema
    join = f"JOIN {person_view} pv ON pv.person_id = p.person_id" if person_view else ""
    return be.sql(f"""
SELECT p.person_id, p.year_of_birth, p.race_concept_id, p.ethnicity_concept_id,
       p.care_site_id, cs.care_site_name,
       op.observation_period_start_date AS obs_start, op.observation_period_end_date AS obs_end,
       d.death_date, d.cause_concept_id
FROM {cdm}.person p {join}
LEFT JOIN {cdm}.care_site cs ON cs.care_site_id = p.care_site_id
LEFT JOIN {cdm}.observation_period op ON op.person_id = p.person_id
LEFT JOIN {cdm}.death d ON d.person_id = p.person_id
WHERE p.gender_concept_id = {C.FEMALE_GENDER_CONCEPT_ID}
""")


# ------------------------------------------------------------------ pass 2 (cohort-scoped)
def register_cohort(be: Backend, cohort: pd.DataFrame) -> None:
    be.register(cohort[["person_id", "index_date"]].copy(), COHORT_VIEW)


def _window(cfg: StudyConfig, date_col: str, lo: int, hi: int) -> str:
    d = days_between(cfg.dialect, date_col, "ch.index_date")
    return f"{d} BETWEEN {lo} AND {hi}"


def cohort_code_events(be: Backend, cfg: StudyConfig) -> pd.DataFrame:
    """All ICD groups for cohort women, from lookback start to the end of follow-up."""
    pf = f"JOIN {COHORT_VIEW} ch ON ch.person_id = e.person_id"
    sql = _events_sql(cfg, "1 = 1", pf, full_history=True).replace(
        "SELECT e.person_id,", "SELECT e.person_id, ch.index_date,", 1)
    lo, hi = -max(cfg.lookback_days, 3650), cfg.outcome_window_days
    return be.sql(f"SELECT DISTINCT * FROM ({sql}) t WHERE "
                  f"{days_between(cfg.dialect, 't.event_date', 't.index_date')} BETWEEN {lo} AND {hi}")


def cohort_measurements(be: Backend, cfg: StudyConfig) -> pd.DataFrame:
    cdm, voc = cfg.cdm_schema, cfg.vocab_schema
    lo = -cfg.lookback_days
    return be.sql(f"""
SELECT m.person_id, ch.index_date, l.item, CAST(m.measurement_date AS DATE) AS event_date,
       m.value_as_number, COALESCE(u.concept_code, m.unit_source_value) AS unit
FROM {cdm}.measurement m
JOIN {COHORT_VIEW} ch ON ch.person_id = m.person_id
JOIN hcman_loinc l ON l.concept_id = m.measurement_concept_id
LEFT JOIN {voc}.concept u ON u.concept_id = m.unit_concept_id
WHERE m.value_as_number IS NOT NULL AND {_window(cfg, 'm.measurement_date', lo, 0)}
""")


def cohort_ehr_tokens(be: Backend, cfg: StudyConfig) -> pd.DataFrame:
    """Stream 3 tokens: ICD-10-CM diagnoses + ATC-4 drug classes in the lookback window."""
    cdm, voc = cfg.cdm_schema, cfg.vocab_schema
    lo = -cfg.lookback_days
    return be.sql(f"""
SELECT co.person_id, 'DX' AS ttype, c.concept_code AS code,
       {days_between(cfg.dialect, 'co.condition_start_date', 'ch.index_date')} AS rel_day
FROM {cdm}.condition_occurrence co
JOIN {COHORT_VIEW} ch ON ch.person_id = co.person_id
JOIN {voc}.concept c ON c.concept_id = co.condition_source_concept_id AND c.vocabulary_id = 'ICD10CM'
WHERE {_window(cfg, 'co.condition_start_date', lo, 0)}
UNION ALL
SELECT de.person_id, 'RX', atc.concept_code,
       {days_between(cfg.dialect, 'de.drug_exposure_start_date', 'ch.index_date')}
FROM {cdm}.drug_exposure de
JOIN {COHORT_VIEW} ch ON ch.person_id = de.person_id
JOIN {voc}.concept_ancestor ca ON ca.descendant_concept_id = de.drug_concept_id
JOIN {voc}.concept atc ON atc.concept_id = ca.ancestor_concept_id
     AND atc.vocabulary_id = 'ATC' AND atc.concept_class_id = 'ATC 4th'
WHERE {_window(cfg, 'de.drug_exposure_start_date', lo, 0)}
""")


def cohort_cardiotoxic_drugs(be: Backend, cfg: StudyConfig) -> pd.DataFrame:
    cdm, voc = cfg.cdm_schema, cfg.vocab_schema
    names = " OR ".join(f"LOWER(i.concept_name) = {_q(n)}" for n in C.CARDIOTOXIC_INGREDIENTS)
    return be.sql(f"""
SELECT DISTINCT de.person_id,
       {days_between(cfg.dialect, 'de.drug_exposure_start_date', 'ch.index_date')} AS rel_day
FROM {cdm}.drug_exposure de
JOIN {COHORT_VIEW} ch ON ch.person_id = de.person_id
JOIN {voc}.concept_ancestor ca ON ca.descendant_concept_id = de.drug_concept_id
JOIN {voc}.concept i ON i.concept_id = ca.ancestor_concept_id
     AND i.vocabulary_id = 'RxNorm' AND i.concept_class_id = 'Ingredient' AND ({names})
WHERE {days_between(cfg.dialect, 'de.drug_exposure_start_date', 'ch.index_date')} <= 0
""")


# ------------------------------------------------------------------ optional: primary-diagnosis check
PRIMARY_GROUP_CODES = {"out_mi": ["I21", "I22"], "out_stroke": ["I60", "I61", "I63"]}


def primary_diagnoses(be: Backend, cfg: StudyConfig) -> pd.DataFrame:
    """Cohort women's MI / stroke diagnoses flagged primary in Mayo FACT_DIAGNOSIS (MCP, 21 Oct
    2026: PRIMARY_DIAGNOSIS_FLAG). -> person_id, event_date, grp. Used only when
    outcomes.mace_primary_diagnosis.mode is fact_diagnosis."""
    from .imaging_sources import _require, person_join
    m = cfg.mace_primary
    _require("mace_primary_diagnosis", m, ("table", "patient_column", "date_column", "code_column", "flag_column"))
    groups = [g for g in m.get("applies_to", list(PRIMARY_GROUP_CODES)) if g in PRIMARY_GROUP_CODES]
    case = " ".join(f"WHEN " + " OR ".join(f"REPLACE(f.{m['code_column']}, '.', '') LIKE '{c}%'" for c in PRIMARY_GROUP_CODES[g])
                    + f" THEN '{g}'" for g in groups)
    if m.get("patient_key", "patient_dk") == "patient_dk":
        dim = cfg.dim_patient
        _require("dim_patient", dim, ("table", "dk_column", "clinic_number_column"))
        pj = (f"JOIN {dim['table']} d ON d.{dim['dk_column']} = f.{m['patient_column']} "
              + person_join(cfg, "p", f"d.{dim['clinic_number_column']}"))
    else:
        pj = person_join(cfg, "p", f"f.{m['patient_column']}")
    flag = str(m.get("flag_value", "Y")).replace("'", "")
    return be.sql(f"""
SELECT DISTINCT p.person_id, CAST(f.{m['date_column']} AS DATE) AS event_date,
       CASE {case} END AS grp
FROM {m['table']} f
{pj}
JOIN {COHORT_VIEW} ch ON ch.person_id = p.person_id
WHERE UPPER(CAST(f.{m['flag_column']} AS STRING)) = '{flag.upper()}'
  AND CASE {case} END IS NOT NULL""")
