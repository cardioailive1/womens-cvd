"""Cohort construction: strata, index date, eligibility.

Design: retrospective cohort of women undergoing cardiac evaluation. Each woman gets ONE index
date — a cardiac evaluation event (ECG, echo, troponin or BNP) that falls inside one of the six
strata from the Model Details Form. Inputs come from before/at index, labels from after it.

Strata (priority order when a woman qualifies for several — rarest first):
  C_scad_fmd               first SCAD code; index = earliest cardiac eval in [-30, 0] days of it
  D_postmeno_takotsubo_cmd first Takotsubo / CMD code in a postmenopausal woman; same anchoring
  B_postpartum             cardiac eval within 182 days after a delivery
  A_pregnant_high_risk     cardiac eval during a high-risk pregnancy (O09) or pregnancy at age >= 35
  F_prior_preeclampsia     cardiac eval >= 42 days after a preeclampsia code, not currently pregnant
  E_pcos_endo_autoimmune   cardiac eval on/after a PCOS, endometriosis, SLE or RA code
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StudyConfig

STRATA = ["C_scad_fmd", "D_postmeno_takotsubo_cmd", "B_postpartum",
          "A_pregnant_high_risk", "F_prior_preeclampsia", "E_pcos_endo_autoimmune"]
DISEASE_ANCHORED = {"C_scad_fmd", "D_postmeno_takotsubo_cmd"}


def _first(ev: pd.DataFrame, groups: list[str]) -> pd.Series:
    sub = ev[ev.grp.isin(groups)]
    return sub.groupby("person_id").event_date.min()


def _earliest_eval_in(evals: pd.DataFrame, anchor: pd.Series, lo: int, hi: int,
                      fallback_to_anchor: bool) -> pd.Series:
    """Earliest cardiac evaluation within [anchor+lo, anchor+hi] per person."""
    a = anchor.rename("anchor").reset_index()
    m = evals.merge(a, on="person_id")
    d = (m.event_date - m.anchor).dt.days
    m = m[(d >= lo) & (d <= hi)]
    out = m.groupby("person_id").event_date.min()
    if fallback_to_anchor:
        out = out.reindex(anchor.index).fillna(anchor)
    return out


def build_candidates(ev: pd.DataFrame, evals: pd.DataFrame, people: pd.DataFrame,
                     cfg: StudyConfig) -> pd.DataFrame:
    """One row per (person, stratum) with that stratum's index date."""
    ev = ev.copy(); evals = evals.copy()
    ev["event_date"] = pd.to_datetime(ev.event_date)
    evals["event_date"] = pd.to_datetime(evals.event_date)
    yob = people.drop_duplicates("person_id").set_index("person_id").year_of_birth
    cands: list[pd.DataFrame] = []

    def add(stratum: str, idx: pd.Series) -> None:
        idx = idx.dropna()
        if len(idx):
            cands.append(pd.DataFrame({"person_id": idx.index, "stratum": stratum,
                                       "index_date": idx.values}))

    # C — SCAD
    scad = _first(ev, ["cond_SCAD"])
    add("C_scad_fmd", _earliest_eval_in(evals, scad, -30, 0, fallback_to_anchor=True))

    # D — postmenopausal Takotsubo / CMD (menopause code before, or age >= 52 as proxy)
    tc = _first(ev, ["cond_TAKOTSUBO", "cond_CMD_INOCA"])
    meno = _first(ev, ["repro_menopause_natural", "repro_menopause_premature",
                       "repro_menopause_surgical"])
    age_at = tc.dt.year - yob.reindex(tc.index)
    post = (meno.reindex(tc.index) <= tc) | (age_at >= 52)
    add("D_postmeno_takotsubo_cmd",
        _earliest_eval_in(evals, tc[post.fillna(False)], -30, 0, fallback_to_anchor=True))

    # B — postpartum: each delivery opens a 182-day window; take the first eval in any window
    deliv = ev[ev.grp == "repro_delivery"][["person_id", "event_date"]].drop_duplicates()
    m = evals.merge(deliv.rename(columns={"event_date": "deliv"}), on="person_id")
    d = (m.event_date - m.deliv).dt.days
    add("B_postpartum", m[(d >= 0) & (d <= cfg.postpartum_max_days)]
        .groupby("person_id").event_date.min())

    # A — high-risk pregnancy: O09 window, or pregnancy at >= 35; eval within 280 days after
    hr = _first(ev, ["repro_high_risk_pregnancy"])
    preg = _first(ev, ["repro_pregnant_state"])
    amg = preg[(preg.dt.year - yob.reindex(preg.index)) >= 35]
    a_anchor = pd.concat([hr, amg]).groupby(level=0).min()
    add("A_pregnant_high_risk", _earliest_eval_in(evals, a_anchor, 0, 280, False))

    # F — prior preeclampsia, >= 42 days later, not pregnant at index
    pe = _first(ev, ["repro_preeclampsia", "repro_hellp"])
    f_idx = _earliest_eval_in(evals, pe, 42, 10 * 365, False)
    preg_dates = ev[ev.grp == "repro_pregnant_state"][["person_id", "event_date"]]
    chk = preg_dates.merge(f_idx.rename("idx").reset_index(), on="person_id")
    pregnant_now = chk[((chk.event_date - chk.idx).dt.days.abs() <= 90)].person_id.unique()
    add("F_prior_preeclampsia", f_idx.drop(index=pregnant_now, errors="ignore"))

    # E — PCOS / endometriosis / autoimmune
    e_anchor = _first(ev, ["repro_pcos", "repro_endometriosis", "repro_sle", "repro_ra"])
    add("E_pcos_endo_autoimmune", _earliest_eval_in(evals, e_anchor, 0, 50 * 365, False))

    if not cands:
        return pd.DataFrame(columns=["person_id", "stratum", "index_date"])
    return pd.concat(cands, ignore_index=True)


def apply_eligibility(cands: pd.DataFrame, ev: pd.DataFrame, people: pd.DataFrame,
                      cfg: StudyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cohort, attrition table). Cohort has one row per woman."""
    attr = []
    def log(step: str, df: pd.DataFrame) -> None:
        attr.append({"step": step, "n_women": int(df.person_id.nunique())})

    log("candidates in any stratum", cands)
    p = people.copy()
    for c in ("obs_start", "obs_end", "death_date"):
        p[c] = pd.to_datetime(p[c])
    # people can have several observation periods: keep the one containing the index
    df = cands.merge(p, on="person_id", how="left")
    df["index_date"] = pd.to_datetime(df.index_date)
    in_period = (df.obs_start <= df.index_date) & (df.obs_end >= df.index_date)
    df = df[in_period | df.obs_start.isna()]

    excl = set(ev.loc[ev.grp == "excl", "person_id"])
    df = df[~df.person_id.isin(excl)]; log("after congenital HD / transplant exclusion", df)

    df["age"] = df.index_date.dt.year - df.year_of_birth
    ok_age = df.age >= cfg.min_age
    preg_strata = df.stratum.isin(["A_pregnant_high_risk", "B_postpartum"])
    ok_age &= ~preg_strata | (df.age <= cfg.max_age_pregnancy_strata)
    df = df[ok_age]; log(f"age >= {cfg.min_age} (<= {cfg.max_age_pregnancy_strata} in A/B)", df)

    prior = (df.index_date - df.obs_start).dt.days
    df = df[prior >= cfg.min_prior_observation_days]
    log(f">= {cfg.min_prior_observation_days} days observation before index", df)

    start, end = pd.Timestamp(cfg.study_start), pd.Timestamp(cfg.study_end)
    df = df[(df.index_date >= start + pd.Timedelta(days=cfg.lookback_days)) & (df.index_date <= end)]
    log("index inside study window", df)

    # one index per woman: highest-priority stratum, then earliest date
    df["prio"] = df.stratum.map({s: i for i, s in enumerate(STRATA)})
    memberships = df.groupby("person_id").stratum.apply(lambda s: sorted(set(s)))
    df = df.sort_values(["person_id", "prio", "index_date"]).drop_duplicates("person_id")
    df["all_strata"] = df.person_id.map(memberships)

    follow = (df.obs_end - df.index_date).dt.days
    died = df.death_date.notna() & ((df.death_date - df.index_date).dt.days <= cfg.outcome_window_days)
    df["outcome_observable"] = (follow >= cfg.outcome_window_days) | died
    log("one index per woman (final cohort)", df)
    log("  of which with full 12-month follow-up", df[df.outcome_observable])

    if cfg.max_per_stratum:
        df = (df.groupby("stratum", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), cfg.max_per_stratum), random_state=cfg.seed)))
        log(f"capped at {cfg.max_per_stratum} per stratum", df)

    df["site"] = df.care_site_name.fillna(df.care_site_id.astype("string")).fillna("unknown")
    cols = ["person_id", "stratum", "all_strata", "index_date", "age", "year_of_birth",
            "race_concept_id", "ethnicity_concept_id", "site", "obs_start", "obs_end",
            "death_date", "cause_concept_id", "outcome_observable"]
    return df[cols].reset_index(drop=True), pd.DataFrame(attr)


def stratum_shortfall(cohort: pd.DataFrame, cfg: StudyConfig) -> pd.DataFrame:
    got = cohort.stratum.value_counts()
    rows = [{"stratum": s, "target": t, "available": int(got.get(s, 0)),
             "meets_target": bool(got.get(s, 0) >= t)} for s, t in cfg.stratum_targets.items()]
    return pd.DataFrame(rows)
