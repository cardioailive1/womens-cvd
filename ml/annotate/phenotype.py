"""Chart review of the automatic condition labels (phenotype validation).

The six condition labels are produced automatically from ICD-10-CM codes (mcp_adapter.labels).
Before training on them, a cardiologist reviews a sample of records to measure how often the
automatic label is right: PPV on sampled positives, NPV on sampled negatives. This is also where
problems like iatrogenic coronary dissections coded as I25.42 (not SCAD) show up.

  1. sample   — per condition, n positives and n negatives (half of the negatives are "near
                misses": a code of that condition that did not meet the labelling rule).
  2. context  — pulls each sampled woman's diagnoses, procedures, key labs and, when the platform
                exposes the OMOP NOTE table, clinical notes, from 30 days before index to the end
                of the label window. Stays inside the workbench.
  3. review   — reviewers work through the queue in the Jupyter tool (annotate.review_ui), blinded
                to the automatic label.
  4. report   — PPV and NPV with 95% Wilson intervals per condition, small cells suppressed.

    from annotate.phenotype import sample_for_review, fetch_context, validation_report
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from mcp_adapter import concepts as C
from mcp_adapter.backend import Backend, days_between
from mcp_adapter.config import StudyConfig

from .store import AnnotationStore

REVIEW_VIEW = "hcman_review_people"
VERDICTS = ["yes", "no", "unsure"]

QUESTION = {
    "SCAD": "Spontaneous coronary artery dissection (not iatrogenic or traumatic)?",
    "PPCM": "Peripartum cardiomyopathy (new LV dysfunction late pregnancy to 5 months postpartum, no other cause)?",
    "PREECLAMPSIA_CVD": "Preeclampsia, eclampsia or HELLP in this pregnancy?",
    "TAKOTSUBO": "Takotsubo (stress) cardiomyopathy?",
    "CMD_INOCA": "Coronary microvascular dysfunction / INOCA (ischaemia without obstructive CAD)?",
    "FMD": "Fibromuscular dysplasia?",
}


def task_name(condition: str) -> str:
    return f"phenotype:{condition}"


def sample_for_review(labels: pd.DataFrame, code_events: pd.DataFrame, conditions=None,
                      n_pos: int = 50, n_neg: int = 50, seed: int = 0) -> pd.DataFrame:
    """-> person_id, condition, algo_label, stratum ('positive', 'near-miss negative', 'negative')."""
    rng = np.random.default_rng(seed)
    conditions = conditions or C.CONDITION_ORDER
    rows = []
    for c in conditions:
        y = labels.set_index("person_id")[f"y_{c}"]
        pos = y[y == 1].index.to_numpy()
        coded = set(code_events.loc[code_events.grp == f"cond_{c}", "person_id"])
        neg = y[y == 0].index
        near = np.array([p for p in neg if p in coded])
        far = np.array([p for p in neg if p not in coded])
        take = lambda arr, k: rng.choice(arr, min(k, len(arr)), replace=False) if len(arr) else []
        k_near = min(n_neg // 2, len(near))
        for p in take(pos, n_pos):
            rows.append((int(p), c, 1, "positive"))
        for p in take(near, k_near):
            rows.append((int(p), c, 0, "near-miss negative"))
        for p in take(far, n_neg - k_near):
            rows.append((int(p), c, 0, "negative"))
    s = pd.DataFrame(rows, columns=["person_id", "condition", "algo_label", "stratum"])
    # random order so reviewers cannot infer the label from position in the queue
    return s.sample(frac=1, random_state=seed).reset_index(drop=True)


def fetch_context(be: Backend, cfg: StudyConfig, cohort: pd.DataFrame, person_ids,
                  before_days: int = 30) -> dict[str, pd.DataFrame]:
    """Diagnoses, procedures, labs and (if available) notes for the sampled women, relative days."""
    who = cohort[cohort.person_id.isin(set(person_ids))][["person_id", "index_date"]]
    be.register(who, REVIEW_VIEW)
    cdm, voc = cfg.cdm_schema, cfg.vocab_schema
    lo, hi = -before_days, max(cfg.label_window_days, cfg.outcome_window_days)
    d = lambda col: days_between(cfg.dialect, col, "w.index_date")

    dx = be.sql(f"""
SELECT x.person_id, {d('x.condition_start_date')} AS rel_day, c.concept_code AS code,
       c.concept_name AS name, v.visit_concept_id
FROM {cdm}.condition_occurrence x JOIN {REVIEW_VIEW} w ON w.person_id = x.person_id
JOIN {voc}.concept c ON c.concept_id = x.condition_source_concept_id
LEFT JOIN {cdm}.visit_occurrence v ON v.visit_occurrence_id = x.visit_occurrence_id
WHERE {d('x.condition_start_date')} BETWEEN {lo} AND {hi}""")
    px = be.sql(f"""
SELECT x.person_id, {d('x.procedure_date')} AS rel_day, c.vocabulary_id AS vocab,
       c.concept_code AS code, c.concept_name AS name
FROM {cdm}.procedure_occurrence x JOIN {REVIEW_VIEW} w ON w.person_id = x.person_id
JOIN {voc}.concept c ON c.concept_id = x.procedure_concept_id
WHERE {d('x.procedure_date')} BETWEEN {lo} AND {hi}""")
    labs = be.sql(f"""
SELECT x.person_id, {d('x.measurement_date')} AS rel_day, c.concept_name AS name,
       x.value_as_number AS value, x.unit_source_value AS unit
FROM {cdm}.measurement x JOIN {REVIEW_VIEW} w ON w.person_id = x.person_id
JOIN {voc}.concept c ON c.concept_id = x.measurement_concept_id
WHERE x.value_as_number IS NOT NULL AND c.vocabulary_id = 'LOINC'
  AND c.concept_code IN ({", ".join("'" + x + "'" for x in C.CARDIAC_EVAL_LOINC + ["2160-0", "718-7"])})
  AND {d('x.measurement_date')} BETWEEN {lo} AND {hi}""")
    try:
        notes = be.sql(f"""
SELECT x.person_id, {d('x.note_date')} AS rel_day, x.note_title AS title, x.note_text AS text
FROM {cdm}.note x JOIN {REVIEW_VIEW} w ON w.person_id = x.person_id
WHERE {d('x.note_date')} BETWEEN {lo} AND {hi}""")
    except Exception:          # NOTE table not exposed on this platform/schema
        notes = pd.DataFrame(columns=["person_id", "rel_day", "title", "text"])
    return {"diagnoses": dx, "procedures": px, "labs": labs, "notes": notes}


def save_context(ctx: dict[str, pd.DataFrame], path: str | Path) -> None:
    pd.to_pickle(ctx, path)


def load_context(path: str | Path) -> dict[str, pd.DataFrame]:
    return pd.read_pickle(path)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(c - h, 3), round(c + h, 3))


def validation_report(sample: pd.DataFrame, store: AnnotationStore, min_cell: int = 11) -> pd.DataFrame:
    """PPV (reviewer agrees on algorithm positives) and NPV (on negatives), 'unsure' excluded."""
    rows = []
    for c, g in sample.groupby("condition"):
        lat = store.latest(task_name(c)).set_index("item_id").value
        g = g.assign(verdict=g.person_id.astype(str).map(lat))
        for lab, name in ((1, "PPV"), (0, "NPV")):
            gg = g[(g.algo_label == lab) & g.verdict.isin(["yes", "no"])]
            agree = int((gg.verdict == ("yes" if lab == 1 else "no")).sum())
            n = len(gg)
            lo, hi = wilson(agree, n)
            rows.append({"condition": c, "metric": name, "n_reviewed": n, "n_agree": agree,
                         "value": round(agree / n, 3) if n else None, "ci_low": lo, "ci_high": hi,
                         "n_unsure": int(((g.algo_label == lab) & (g.verdict == "unsure")).sum()),
                         "n_pending": int(((g.algo_label == lab) & g.verdict.isna()).sum())})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.astype(object)
    small = pd.to_numeric(out.n_reviewed) < min_cell
    out.loc[small, ["value", "ci_low", "ci_high"]] = "suppressed"
    for col in ("n_reviewed", "n_agree", "n_unsure", "n_pending"):
        v = pd.to_numeric(out[col])
        out.loc[(v > 0) & (v < min_cell), col] = f"<{min_cell}"
    return out
