"""Label derivation for the six HCMAN heads. Every label has a matching mask (1 = usable).

All windows are in days relative to the index date (day 0). Inputs never use day > 0, labels
never use day < 0, except the explicit washout check that makes condition labels *incident*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import concepts as C
from .config import StudyConfig

# Condition labels are incident: a code in [0, label_window] and none in the washout before
# index. Preeclampsia uses one pregnancy (280 d), so a prior pregnancy's preeclampsia is
# history (a stream-4 input), not a label. Others: any earlier code = prevalent -> label 0.
WASHOUT_DAYS = {"PREECLAMPSIA_CVD": 280}
INPATIENT = set(C.INPATIENT_VISIT_IDS) | set(C.ED_VISIT_IDS)


def _rel(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.copy()
    ev["event_date"] = pd.to_datetime(ev.event_date)
    ev["index_date"] = pd.to_datetime(ev.index_date)
    ev["rel_day"] = (ev.event_date - ev.index_date).dt.days
    return ev


def condition_labels(ev: pd.DataFrame, cohort: pd.DataFrame, cfg: StudyConfig) -> pd.DataFrame:
    ev = _rel(ev)
    out = pd.DataFrame(index=cohort.person_id)
    W = cfg.label_window_days
    for cond in C.CONDITION_ORDER:
        e = ev[ev.grp == f"cond_{cond}"]
        win = e[(e.rel_day >= 0) & (e.rel_day <= W)]
        if cfg.condition_label_rule == "two_codes_or_inpatient":
            n_dates = win.groupby("person_id").event_date.nunique()
            inpt = win[win.visit_concept_id.isin(INPATIENT)].person_id.unique()
            pos = set(n_dates[n_dates >= 2].index) | set(inpt)
        else:
            pos = set(win.person_id)
        wash = WASHOUT_DAYS.get(cond)
        prior = e[(e.rel_day < 0) & ((e.rel_day >= -wash) if wash else True)]
        pos -= set(prior.person_id)
        out[f"y_{cond}"] = out.index.isin(pos).astype(np.float32)
    out["m_condition"] = 1.0
    return out


def outcome_labels(ev: pd.DataFrame, cohort: pd.DataFrame, cfg: StudyConfig,
                   cv_death_concepts: set[int] | None = None) -> pd.DataFrame:
    """MACE (12 m), postpartum CVD, risk tier, atypical presentation, FMD screening trigger."""
    ev = _rel(ev)
    co = cohort.set_index("person_id")
    H = cfg.outcome_window_days
    out = pd.DataFrame(index=co.index)

    # Blanking: events in the first `outcome_blanking_days` belong to the index episode (the
    # MI coded the day after a SCAD work-up is the presentation, not a future MACE).
    B = cfg.outcome_blanking_days
    fut = ev[(ev.rel_day > B) & (ev.rel_day <= H)]
    mi_stroke = fut[fut.grp.isin(["out_mi", "out_stroke"])]
    hf = fut[(fut.grp == "out_heart_failure") & fut.visit_concept_id.isin(INPATIENT)]
    first_event = pd.concat([mi_stroke, hf]).groupby("person_id").rel_day.min()

    death_rel = (pd.to_datetime(co.death_date) - pd.to_datetime(co.index_date)).dt.days
    cv_death = death_rel.between(1, H)          # deaths count from day 1
    if cv_death_concepts is not None:   # restrict to CV causes when cause of death is populated
        cv_death &= co.cause_concept_id.isin(cv_death_concepts)
    death_day = death_rel.where(cv_death)

    t_event = pd.concat([first_event.reindex(co.index), death_day], axis=1).min(axis=1)
    out["y_mace"] = t_event.notna().astype(np.float32)
    out["m_mace"] = (co.outcome_observable | t_event.notna()).astype(np.float32)
    out["t_mace_days"] = t_event

    # Postpartum CVD: delivery within postpartum_max_days before index; CV event to delivery+365
    deliv = ev[(ev.grp == "repro_delivery") & (ev.rel_day <= 0)
               & (ev.rel_day >= -cfg.postpartum_max_days)].groupby("person_id").rel_day.max()
    pp_grps = ["out_mi", "out_stroke", "out_heart_failure", "out_cardiomyopathy",
               "cond_SCAD", "cond_TAKOTSUBO", "cond_PPCM"]
    pp = ev[ev.grp.isin(pp_grps) & (ev.rel_day > B)].merge(
        deliv.rename("deliv_rel").reset_index(), on="person_id")
    pp = pp[pp.rel_day <= pp.deliv_rel + 365]
    is_pp = out.index.isin(deliv.index)
    out["y_postpartum_cvd"] = out.index.isin(pp.person_id).astype(np.float32)
    out["m_postpartum_cvd"] = (is_pp & (co.outcome_observable | out.y_postpartum_cvd.astype(bool))
                               ).astype(np.float32)

    # Risk tier (0 Low, 1 Moderate, 2 High, 3 Critical) — observed-outcome definition
    tier = np.zeros(len(out), dtype=np.int64)
    any_cond = ev[ev.grp.str.startswith("cond_") & ev.rel_day.between(0, cfg.label_window_days)]
    tier[out.index.isin(any_cond.person_id)] = 1
    tier[(t_event > 30).values] = 2
    tier[(t_event <= 30).values] = 3
    out["y_risk_tier"] = tier
    out["m_risk_tier"] = out.m_mace

    # Atypical presentation: ACS coded within +-3 d of index; positive when atypical symptoms
    # are coded in that window and chest pain is not. Mask = women with an ACS episode.
    near = ev[ev.rel_day.between(-3, 3)]
    acs = set(near[near.grp == "out_acs"].person_id)
    atyp = set(near[near.grp == "sym_atypical"].person_id)
    chest = set(near[near.grp == "sym_chest_pain"].person_id)
    out["y_atypical"] = out.index.isin((atyp - chest) & acs).astype(np.float32)
    out["m_atypical"] = out.index.isin(acs).astype(np.float32)
    return out


def fmd_trigger_labels(cond: pd.DataFrame, ev: pd.DataFrame, cfg: StudyConfig) -> pd.DataFrame:
    """Guidelines recommend head-to-pelvis FMD imaging for every SCAD patient, so the trigger
    target is SCAD itself; `fmd_confirmed` (any FMD code up to +365 d) is kept for evaluating
    yield among triggered patients."""
    ev = _rel(ev)
    out = pd.DataFrame(index=cond.index)
    out["y_fmd_trigger"] = cond["y_SCAD"]
    out["m_fmd_trigger"] = 1.0
    fmd = ev[(ev.grp == "cond_FMD") & (ev.rel_day <= cfg.outcome_window_days)]
    out["fmd_confirmed"] = out.index.isin(fmd.person_id).astype(np.float32)
    return out


def build_labels(ev: pd.DataFrame, cohort: pd.DataFrame, cfg: StudyConfig) -> pd.DataFrame:
    cond = condition_labels(ev, cohort, cfg)
    outc = outcome_labels(ev, cohort, cfg)
    fmd = fmd_trigger_labels(cond, ev, cfg)
    return pd.concat([cond, outc, fmd], axis=1).reset_index().rename(columns={"index": "person_id"})


def require_primary(ev: pd.DataFrame, primary: pd.DataFrame, groups=("out_mi", "out_stroke"),
                    match_days: int = 1) -> tuple[pd.DataFrame, int]:
    """Optional MACE rule (outcomes.mace_primary_diagnosis): keep an MI or stroke code only when
    the same woman has that diagnosis flagged PRIMARY within `match_days` of it (Mayo
    FACT_DIAGNOSIS). Drops history-of and rule-out codes recorded as secondary diagnoses, at the
    cost of missing real events coded only as secondary. Other groups are untouched.
    -> (filtered events, number of MI / stroke events dropped)."""
    is_g = ev.grp.isin(groups)
    target = ev[is_g].copy()
    if target.empty:
        return ev, 0
    target["_d"] = pd.to_datetime(target.event_date)
    p = primary.copy(); p["_pd"] = pd.to_datetime(p.event_date)
    m = target.reset_index().merge(p[["person_id", "grp", "_pd"]], on=["person_id", "grp"], how="left")
    m["_ok"] = (m._d - m._pd).abs().dt.days <= match_days
    ok_idx = set(m.loc[m._ok.fillna(False), "index"])
    keep = ~is_g | ev.index.isin(ok_idx)
    return ev[keep], int((~keep).sum())
