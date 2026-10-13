"""Stream 4 — the 15 women-specific features of the Model Details Form, built from OMOP codes.

Each feature records where it came from. Two cannot be observed directly in a de-identified
EHR and are approximated (flagged in REPRO_PROVENANCE): gravida is unavailable (para is the
count of distinct deliveries), and menopausal status falls back to age when no menopause code
exists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import concepts as C

PREG_STATUS = ["not_pregnant", "pregnant", "postpartum"]
MENO_STATUS = ["pre", "peri", "post", "surgical"]

REPRO_PROVENANCE = {
    "pregnancy_status": "Z33.1/O09/Z34/Z3A in [-60, 0] d -> pregnant; delivery in [-182, 0] d -> postpartum",
    "gestational_age_weeks": "latest Z3A.xx code in [-60, 0] d + days since it / 7",
    "gravida_para": "APPROXIMATION: para only = distinct deliveries >= 180 d apart before index",
    "preeclampsia_history": "O11/O14/O15 before index",
    "gestational_diabetes": "O24.4 before index",
    "hellp_history": "O14.2 before index",
    "menopausal_status": "codes N95.1/Z78.0/E28.31/N95.3/E89.4/Z90.72; APPROXIMATION by age when absent",
    "age_at_menopause": "age at first menopause code (missing otherwise)",
    "pcos_diagnosis": "E28.2 before index",
    "endometriosis": "N80 before index",
    "autoimmune_disease_sle_ra": "M32 / M05 / M06 before index",
    "migraine_with_aura": "G43.1 before index",
    "fmd_history": "I77.3 before index (incident FMD after index is a label, never both)",
    "breast_cancer_cardiotoxic_treatment": "C50/Z85.3 AND anthracycline or HER2 therapy before index",
    "emotional_stress_trigger_takotsubo": "F43.0/F43.2/Z63/Z65 in [-30, 0] d",
}

# Encoded vector layout (24 dims): one-hots for the two categoricals, value + missing flag for
# the two continuous features, the rest binary.
FEATURE_NAMES = (
    [f"pregnancy_status={s}" for s in PREG_STATUS]
    + ["gestational_age_weeks", "gestational_age_missing", "gravida_para",
       "preeclampsia_history", "gestational_diabetes", "hellp_history"]
    + [f"menopausal_status={s}" for s in MENO_STATUS]
    + ["menopausal_status_inferred", "age_at_menopause", "age_at_menopause_missing",
       "pcos_diagnosis", "endometriosis", "autoimmune_disease_sle_ra", "migraine_with_aura",
       "fmd_history", "breast_cancer_cardiotoxic_treatment", "emotional_stress_trigger_takotsubo"]
)
REPRO_DIM = len(FEATURE_NAMES)


def _gest_weeks(code: str) -> float:
    # Z3A.08 .. Z3A.42 = weeks; Z3A.01 < 8 weeks; Z3A.49 > 42 weeks; Z3A.00 unspecified
    try:
        n = int(code.split(".")[1][:2])
    except (IndexError, ValueError):
        return np.nan
    return {0: np.nan, 1: 6.0, 49: 43.0}.get(n, float(n))


def build_repro(ev: pd.DataFrame, cohort: pd.DataFrame, cardiotoxic: pd.DataFrame) -> pd.DataFrame:
    ev = ev.copy()
    ev["event_date"] = pd.to_datetime(ev.event_date)
    ev["rel_day"] = (ev.event_date - pd.to_datetime(ev.index_date)).dt.days
    past = ev[ev.rel_day <= 0]
    co = cohort.set_index("person_id")
    f = pd.DataFrame(index=co.index)

    def any_before(grps, lo=None):
        s = past[past.grp.isin(grps)]
        if lo is not None:
            s = s[s.rel_day >= lo]
        return f.index.isin(s.person_id)

    # pregnancy status
    preg = any_before(["repro_pregnant_state"], -60)
    deliv = past[past.grp == "repro_delivery"]
    pp = f.index.isin(deliv[deliv.rel_day >= -182].person_id)
    last_preg = past[past.grp == "repro_pregnant_state"].groupby("person_id").rel_day.max()
    last_del = deliv.groupby("person_id").rel_day.max()
    delivered_after_preg = (last_del.reindex(f.index) >= last_preg.reindex(f.index)).fillna(False).values
    status = np.where(pp & (delivered_after_preg | ~preg), "postpartum",
                      np.where(preg, "pregnant", "not_pregnant"))
    for s in PREG_STATUS:
        f[f"pregnancy_status={s}"] = (status == s).astype(np.float32)

    gw = past[(past.grp == "repro_gest_weeks") & (past.rel_day >= -60)]
    gw = gw.sort_values("rel_day").groupby("person_id").tail(1)
    weeks = pd.Series([_gest_weeks(c) for c in gw.concept_code], index=gw.person_id.values)
    weeks = weeks + (-gw.rel_day.values) / 7.0
    weeks = weeks.reindex(f.index).where(status == "pregnant")
    f["gestational_age_weeks"] = (weeks / 40.0).fillna(0).astype(np.float32)
    f["gestational_age_missing"] = weeks.isna().astype(np.float32).values

    d = deliv.sort_values("rel_day")
    def n_deliveries(days: pd.Series) -> int:
        n, last = 0, None
        for x in days:
            if last is None or x - last >= 180:
                n, last = n + 1, x
        return n
    para = d.groupby("person_id").rel_day.apply(n_deliveries).reindex(f.index).fillna(0)
    f["gravida_para"] = (np.minimum(para, 6) / 6.0).astype(np.float32).values

    f["preeclampsia_history"] = any_before(["repro_preeclampsia"]).astype(np.float32)
    f["gestational_diabetes"] = any_before(["repro_gestational_diabetes"]).astype(np.float32)
    f["hellp_history"] = any_before(["repro_hellp"]).astype(np.float32)

    # menopause
    surg = any_before(["repro_menopause_surgical"])
    nat = any_before(["repro_menopause_natural", "repro_menopause_premature"])
    peri = any_before(["repro_perimenopause"], -730)
    age = co.age.reindex(f.index).values
    coded = surg | nat | peri
    inferred_status = np.where(age >= 52, "post", np.where(age >= 45, "peri", "pre"))
    meno = np.where(surg, "surgical", np.where(nat, "post", np.where(peri, "peri", inferred_status)))
    meno = np.where(status != "not_pregnant", "pre", meno)
    for s in MENO_STATUS:
        f[f"menopausal_status={s}"] = (meno == s).astype(np.float32)
    f["menopausal_status_inferred"] = (~coded).astype(np.float32)

    mcodes = past[past.grp.isin(["repro_menopause_natural", "repro_menopause_premature",
                                 "repro_menopause_surgical"])]
    first_meno = mcodes.groupby("person_id").event_date.min().reindex(f.index)
    yob = co.year_of_birth.reindex(f.index)
    a_meno = first_meno.dt.year - yob
    f["age_at_menopause"] = (a_meno / 60.0).fillna(0).astype(np.float32).values
    f["age_at_menopause_missing"] = a_meno.isna().astype(np.float32).values

    f["pcos_diagnosis"] = any_before(["repro_pcos"]).astype(np.float32)
    f["endometriosis"] = any_before(["repro_endometriosis"]).astype(np.float32)
    f["autoimmune_disease_sle_ra"] = any_before(["repro_sle", "repro_ra"]).astype(np.float32)
    f["migraine_with_aura"] = any_before(["repro_migraine_with_aura"]).astype(np.float32)
    f["fmd_history"] = f.index.isin(past[(past.grp == "cond_FMD") & (past.rel_day < 0)].person_id
                                    ).astype(np.float32)
    bc = any_before(["repro_breast_cancer"])
    tox = f.index.isin(cardiotoxic.person_id) if len(cardiotoxic) else np.zeros(len(f), bool)
    f["breast_cancer_cardiotoxic_treatment"] = (bc & tox).astype(np.float32)
    f["emotional_stress_trigger_takotsubo"] = any_before(["repro_emotional_stress"], -30
                                                         ).astype(np.float32)
    return f[FEATURE_NAMES].reset_index().rename(columns={"index": "person_id"})
