"""End-to-end study build: feasibility -> cohort -> labels -> features -> splits -> reports.

Row-level outputs go to `out_dir` INSIDE the MCP workbench and never leave it. Only the
suppressed aggregate reports in `cfg.report_dir` are candidates for export.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import concepts as C
from . import queries as Q
from .backend import Backend
from .cohort import apply_eligibility, build_candidates, known_birth_year, stratum_shortfall
from .config import StudyConfig
from .features.labs_ehr import harmonize_units
from .features.repro import REPRO_PROVENANCE, build_repro
from .features.waveforms import nearest_record
from .labels import build_labels
from .privacy import suppress, write_report
from .splits import lock_splits, make_splits

log = logging.getLogger("hcman.pipeline")


def run_feasibility(be: Backend, cfg: StudyConfig) -> pd.DataFrame:
    Q.resolve_icd_groups(be, cfg, C.all_icd_groups())
    counts = Q.feasibility_counts(be, cfg).sort_values("grp")
    rep = suppress(counts, ["n_women"], cfg.min_cell_size)
    write_report(rep, "01_feasibility_counts", cfg.report_dir)
    labs = Q.lab_code_coverage(be, cfg)
    write_report(suppress(labs, ["n_rows", "n_women"], cfg.min_cell_size), "01b_lab_code_coverage", cfg.report_dir)
    return counts


def _load_manifest(src, be: Backend | None = None) -> pd.DataFrame | None:
    """Imaging manifest from a platform table ("sql:<schema.table>") or a CSV / Parquet file."""
    if src.kind == "none" or not src.manifest:
        return None
    man = str(src.manifest)
    if man.startswith("sql:"):
        if be is None:
            raise ValueError("a sql: manifest needs a backend")
        m = be.sql(f"SELECT * FROM {man[4:]}")
    else:
        m = pd.read_csv(man) if man.endswith(".csv") else pd.read_parquet(man)
    missing = {"person_id", "record_date", "path"} - set(m.columns)
    if missing:
        raise ValueError(f"imaging manifest {man} lacks columns {sorted(missing)}")
    if src.path:
        m["path"] = m.path.map(lambda x: str(Path(src.path) / x))
    return m


def check_imaging_config(cfg: StudyConfig) -> None:
    """Imaging is in scope (confirmed under the MCP contract): fail early if it is not wired."""
    if not cfg.imaging_required:
        return
    for name, src in (("ecg_source", cfg.ecg_source), ("echo_source", cfg.echo_source)):
        if src.kind == "none" or not src.manifest:
            raise ValueError(
                f"{name} is not configured but imaging_required is true. Set platform.{name}.kind "
                f"and .manifest in config/study.yaml to the MCP table or files (protocol section 10).")


def _imaging_selection(be, cfg, people_idx: pd.DataFrame):
    ecg_m, echo_m = _load_manifest(cfg.ecg_source, be), _load_manifest(cfg.echo_source, be)
    ecg_sel = (nearest_record(ecg_m, people_idx, cfg.ecg_echo_window_days)
               if ecg_m is not None else pd.DataFrame(columns=["person_id", "path"]))
    echo_sel = (nearest_record(echo_m, people_idx, cfg.ecg_echo_window_days, by=("person_id", "view"))
                if echo_m is not None and "view" in echo_m else pd.DataFrame(columns=["person_id", "path", "view"]))
    return ecg_sel, echo_sel


def build_study(be: Backend, cfg: StudyConfig, out_dir: str | Path) -> dict[str, pd.DataFrame]:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    check_imaging_config(cfg)
    Q.resolve_icd_groups(be, cfg, C.all_icd_groups())
    Q.resolve_loinc(be, cfg)

    log.info("pass 1: strata, cardiac evaluations, persons")
    ev1 = Q.stratum_events(be, cfg)
    evals = Q.cardiac_eval_events(be, cfg)
    people = Q.persons(be, cfg)
    cands = build_candidates(ev1, evals, people, cfg)
    cohort, attrition = apply_eligibility(cands, ev1, people, cfg)
    Q.register_cohort(be, cohort)

    log.info("pass 2: cohort-scoped events, labs, tokens (%d women)", len(cohort))
    ev = Q.cohort_code_events(be, cfg)
    meas = harmonize_units(Q.cohort_measurements(be, cfg))
    tokens = Q.cohort_ehr_tokens(be, cfg)
    tox = Q.cohort_cardiotoxic_drugs(be, cfg)

    mode = str((cfg.mace_primary or {}).get("mode", "off"))
    if mode not in ("off", "omop", "fact_diagnosis"):
        raise ValueError(f"outcomes.mace_primary_diagnosis.mode must be off, omop or fact_diagnosis, not {mode!r}")
    if mode != "off":
        from .labels import require_primary
        mp = cfg.mace_primary
        ev, dropped = require_primary(ev, Q.primary_diagnoses(be, cfg),
                                      tuple(mp.get("applies_to", ["out_mi", "out_stroke"])),
                                      int(mp.get("match_days", 1)))
        write_report(suppress(pd.DataFrame([{"item": "MI / stroke codes dropped: not a primary diagnosis",
                                             "n": dropped}]), ["n"], cfg.min_cell_size),
                     "04b_mace_primary_diagnosis", cfg.report_dir)
        log.info("MACE primary-diagnosis rule: %d MI / stroke codes dropped", dropped)
    labels = build_labels(ev, cohort, cfg)
    repro = build_repro(ev, cohort, tox)
    split = make_splits(cohort, labels, cfg).rename("split").reset_index()
    split, lock = lock_splits(be, cfg, split)
    if lock:
        write_report(suppress(pd.DataFrame([{"item": k, "n_women": v} for k, v in lock.items()]),
                              ["n_women"], cfg.min_cell_size), "03b_split_lock", cfg.report_dir)
    cohort = cohort.merge(split, on="person_id")

    ecg_sel, echo_sel = _imaging_selection(be, cfg, cohort)

    tables = {"cohort": cohort, "labels": labels, "repro": repro, "measurements": meas,
              "ehr_tokens": tokens, "ecg_records": ecg_sel, "echo_records": echo_sel,
              "code_events": ev}            # used by annotate.phenotype for chart review sampling
    for name, df in tables.items():
        df.to_pickle(out / f"{name}.pkl")

    _reports(cfg, cohort, attrition, labels, meas, ecg_sel, echo_sel)
    return tables


def _reports(cfg, cohort, attrition, labels, meas, ecg_sel, echo_sel) -> None:
    k, rd = cfg.min_cell_size, cfg.report_dir
    write_report(suppress(attrition, ["n_women"], k), "02_attrition", rd)
    write_report(suppress(stratum_shortfall(cohort, cfg), ["available"], k), "03_stratum_vs_target", rd)

    df = cohort[["person_id", "split", "stratum"]].merge(labels, on="person_id")
    rows = []
    heads = [f"y_{c}" for c in C.CONDITION_ORDER] + ["y_mace", "y_postpartum_cvd", "y_atypical",
                                                    "y_fmd_trigger"]
    masks = {"y_mace": "m_mace", "y_postpartum_cvd": "m_postpartum_cvd", "y_atypical": "m_atypical"}
    for sp, g in df.groupby("split"):
        for h in heads:
            m = g[masks[h]] > 0 if h in masks else slice(None)
            rows.append({"split": sp, "label": h, "n_eligible": int(g.loc[m, h].count()),
                         "n_positive": int(g.loc[m, h].sum())})
    write_report(suppress(pd.DataFrame(rows), ["n_eligible", "n_positive"], k), "04_label_prevalence", rd)

    has_labs = set(meas[meas.item.isin(C.LAB_ORDER) & meas.rel_day.between(-cfg.labs_window_days, 0)].person_id)
    serial = meas[meas.item.isin(C.LAB_ORDER) & meas.rel_day.between(-cfg.labs_window_days, 0)]
    serial = set(serial.groupby(["person_id", "item"]).size().loc[lambda s: s >= 2]
                 .reset_index().person_id)
    avail = pd.DataFrame([
        {"stream": "1_ecg", "n_women": cohort.person_id.isin(ecg_sel.person_id).sum()},
        {"stream": "2_echo", "n_women": cohort.person_id.isin(echo_sel.person_id).sum()},
        {"stream": "3_ehr", "n_women": len(cohort)},
        {"stream": "4_repro", "n_women": len(cohort)},
        {"stream": "5_labs_any", "n_women": cohort.person_id.isin(has_labs).sum()},
        {"stream": "5_labs_serial(>=2)", "n_women": cohort.person_id.isin(serial).sum()},
    ])
    write_report(suppress(avail, ["n_women"], k), "05_modality_availability", rd)
    write_report({"repro_feature_provenance": REPRO_PROVENANCE,
                  "condition_codes": C.CONDITIONS, "label_rule": cfg.condition_label_rule},
                 "06_definitions", rd)


def build_pretraining_pool(be: Backend, cfg: StudyConfig, cohort: pd.DataFrame,
                           out_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Self-supervised pre-training pool for first-time training (no prior weights exist).

    Women with any cardiac evaluation in the study window, EXCLUDING every woman in the
    cohort's validation and test splits (so pre-training never sees evaluation patients).
    Each woman's pseudo-index is her latest cardiac evaluation; her EHR tokens, labs, ECG and
    echo are taken relative to it exactly as for the cohort. No labels are used.
    Run after build_study (it re-registers the cohort view).
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    check_imaging_config(cfg)
    Q.resolve_icd_groups(be, cfg, C.all_icd_groups())
    Q.resolve_loinc(be, cfg)
    held_out = set(cohort.loc[cohort.split.isin(["val", "test", "test_temporal", "test_site"]), "person_id"])
    ev = Q.cardiac_eval_events(be, cfg)
    ev["event_date"] = pd.to_datetime(ev.event_date)
    pool = ev[~ev.person_id.isin(held_out)].groupby("person_id").event_date.max().rename("index_date").reset_index()
    # adults only, as in the cohort: age >= min_age at the pseudo-index, birth year known
    people = Q.persons(be, cfg).drop_duplicates("person_id")[["person_id", "year_of_birth"]]
    pool = pool.merge(people, on="person_id", how="inner")
    n_before = len(pool)
    pool = pool[known_birth_year(pool.year_of_birth, cfg)
                & (pool.index_date.dt.year - pool.year_of_birth >= cfg.min_age)]
    pool = pool.drop(columns="year_of_birth")
    n_adult = len(pool)
    log.info("pre-training pool: %d of %d women kept as adults with a known birth year", n_adult, n_before)
    if len(pool) > cfg.pretrain["max_women"]:
        pool = pool.sample(cfg.pretrain["max_women"], random_state=cfg.seed)
    Q.register_cohort(be, pool)
    meas = harmonize_units(Q.cohort_measurements(be, cfg))
    tokens = Q.cohort_ehr_tokens(be, cfg)
    ecg_sel, echo_sel = _imaging_selection(be, cfg, pool)
    tables = {"pool": pool, "measurements": meas, "ehr_tokens": tokens,
              "ecg_records": ecg_sel, "echo_records": echo_sel}
    for name, df in tables.items():
        df.to_pickle(out / f"{name}.pkl")
    k = cfg.min_cell_size
    write_report(suppress(pd.DataFrame([
        {"item": "pool_women", "n_women": len(pool)},
        {"item": "with_ecg", "n_women": int(pool.person_id.isin(ecg_sel.person_id).sum())},
        {"item": "with_echo", "n_women": int(pool.person_id.isin(echo_sel.person_id).sum())},
        {"item": "excluded_val_test", "n_women": len(held_out)},
        {"item": "excluded_under_min_age_or_birth_year_unknown", "n_women": n_before - n_adult},
    ]), ["n_women"], k), "07_pretraining_pool", cfg.report_dir)
    return tables
