"""Automatic tier-1 audit (annotate.auto_audit) and its triage / replace modes in LabelAudit."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from annotate import echo as E
from annotate.audit import AUTO_AUDITOR, ECHO_TASK, LabelAudit, echo_audit_items, phenotype_audit_items
from annotate.audit_ui import AuditScreen
from annotate.auto_audit import audit_echo, audit_phenotypes, lvef_values, mentions, phenotype_verdict
from annotate.store import AnnotationStore

DS, CARD = ["ds_ana", "ds_ben"], ["dr_cole"]


def test_mentions_negation_history():
    assert mentions("Angiogram consistent with spontaneous coronary artery dissection of the LAD.", "SCAD") == ["affirmed"]
    assert mentions("No evidence of spontaneous coronary artery dissection.", "SCAD") == ["negated"]
    assert mentions("History of SCAD in 2015. Now chest pain.", "SCAD") == ["historical"]
    assert mentions("Mother had fibromuscular dysplasia.", "FMD") == ["family"]
    # negation does not leak across sentences
    assert mentions("No chest pain. Takotsubo cardiomyopathy confirmed.", "TAKOTSUBO") == ["affirmed"]
    assert lvef_values("LVEF 30-35%. Prior EF 60 %.") == [32.5, 60.0]


@pytest.mark.parametrize("auto,ev,expect", [
    ("yes", dict(affirmed=2, negated=0, historical=0, n_notes=3, support=[], contrary=[]), ("agree", None)),
    ("yes", dict(affirmed=0, negated=0, historical=0, n_notes=3, support=[], contrary=["iatrogenic"]), ("disagree", "no")),
    ("yes", dict(affirmed=0, negated=0, historical=0, n_notes=0, support=[], contrary=[]), ("uncertain", None)),
    ("yes", dict(affirmed=1, negated=0, historical=0, n_notes=2, support=[], contrary=["iatrogenic"]), ("uncertain", None)),
    ("no", dict(affirmed=2, negated=0, historical=0, n_notes=2, support=[], contrary=[]), ("disagree", "yes")),
    ("no", dict(affirmed=0, negated=0, historical=1, n_notes=2, support=[], contrary=[]), ("uncertain", None)),
    ("no", dict(affirmed=0, negated=0, historical=0, n_notes=4, support=[], contrary=[]), ("agree", None)),
    ("no", dict(affirmed=0, negated=0, historical=0, n_notes=0, support=[], contrary=[]), ("uncertain", None)),
])
def test_phenotype_verdict(auto, ev, expect):
    v, conf, why, value = phenotype_verdict(auto, ev)
    assert (v, value) == expect and 0 <= conf <= 1 and why


def make_case(n=120, seed=1):
    """SCAD sample with notes that mostly reflect the truth; 12 rule errors."""
    rng = np.random.default_rng(seed)
    pids = np.arange(1000, 1000 + n)
    algo = rng.integers(0, 2, n)
    truth = algo.copy()
    wrong = rng.choice(n, 12, replace=False)
    truth[wrong] = 1 - truth[wrong]
    notes, procs, dx = [], [], []
    for k, p in enumerate(pids):
        if k % 15 == 4:
            continue                                                     # no notes at all
        if truth[k] == 1:
            notes.append((p, 1, "Cath report", "Findings consistent with spontaneous coronary artery dissection of the LAD."))
        else:
            notes.append((p, 1, "Cardiology consult", "Chest pain. Angiogram: no evidence of spontaneous coronary artery dissection."))
            if algo[k] == 1 and k % 3 == 0:                              # iatrogenic dissection coded as I25.42
                dx.append((p, 3, "I25.42", "Coronary artery dissection"))
                procs.append((p, 2, "92928", "PCI with stent"))
    ctx = {"notes": pd.DataFrame(notes, columns=["person_id", "rel_day", "title", "text"]),
           "procedures": pd.DataFrame(procs, columns=["person_id", "rel_day", "code", "name"]),
           "diagnoses": pd.DataFrame(dx, columns=["person_id", "rel_day", "code", "name"]),
           "labs": pd.DataFrame(columns=["person_id", "rel_day", "name", "value", "unit"])}
    sample = pd.DataFrame({"person_id": pids, "condition": "SCAD", "algo_label": algo, "truth": truth})
    return sample, ctx, set(wrong.tolist())


def test_auto_audit_phenotypes_finds_the_errors():
    sample, ctx, wrong = make_case()
    items = phenotype_audit_items(sample)
    v = audit_phenotypes(items, ctx).set_index("item_id")
    pid = sample.person_id.astype(str)
    no_notes = {str(p) for k, p in enumerate(sample.person_id) if k % 15 == 4}
    for k, i in enumerate(pid):
        if i in no_notes:
            assert v.at[i, "verdict"] == "uncertain"                     # never confirms without evidence
        elif k in wrong:
            assert v.at[i, "verdict"] == "disagree"
        else:
            assert v.at[i, "verdict"] == "agree"
    assert v.note.str.len().min() > 0


def truth_map(sample):
    return dict(zip(sample.person_id.astype(str), np.where(sample.truth == 1, "yes", "no")))


def test_triage_mode(tmp_path):
    sample, ctx, wrong = make_case()
    truth = truth_map(sample)
    items = phenotype_audit_items(sample)
    audit = LabelAudit(items, AnnotationStore(tmp_path / "a.csv"), DS, CARD, spot_check_fraction=0.10,
                       spot_check_min=5, auto_tier1="triage", auto_spot_check_fraction=0.20, seed=2)
    av = audit_phenotypes(items, ctx)
    assert audit.submit_auto(av) == len(items)
    flagged = set(av[av.verdict != "agree"].item_id)

    q = audit.tier1_queue("ds_ana")
    assert set(q.item_id) == flagged                                     # data scientists only see flagged items
    scr = AuditScreen(audit, "ds_ana", tier=1, phenotype_context=ctx)
    c = scr.current()
    assert c["auto_audit"]["verdict"] in ("disagree", "uncertain") and c["auto_audit"]["note"]
    while (c := scr.current()) is not None:                               # DS answers correctly
        right = truth[c["item_id"]] == c["auto_value"]
        scr.submit(verdict="agree" if right else "disagree")
    assert audit.tier1_queue("ds_ben").empty

    q2 = audit.tier2_queue("dr_cole")
    n_auto_acc = len(items) - len(flagged)
    n_dis = sum(1 for i in flagged if truth[i] != dict(zip(items.item_id, items.auto_value))[i])
    n_hum_acc = len(flagged) - n_dis
    assert (q2.reason == "disagreement").sum() == n_dis
    assert (q2.reason == "spot-check").sum() == (max(5, int(np.ceil(0.2 * n_auto_acc)))
                                                 + min(n_hum_acc, max(5, int(np.ceil(0.1 * n_hum_acc)))))
    card = AuditScreen(audit, "dr_cole", tier=2, phenotype_context=ctx)
    assert "auto_audit" not in card.current() and "auto_value" not in card.current()   # still blinded
    while (c := card.current()) is not None:
        card.submit(value=truth[c["item_id"]])

    f = audit.final_labels()
    assert set(f.tier1_by) == {"auto", "human"}
    assert (f.status.isin(["cardiologist", "tier 1 confirmed"])).all()
    assert all(f.final_value == f.item_id.map(truth))                     # every final label is right here
    rep = audit.report(min_cell=1).set_index("tier1_by")
    assert rep.at["auto", "spot_checks_done"] == max(5, int(np.ceil(0.2 * n_auto_acc)))
    assert rep.at["auto", "spot_check_agreement"] == 1.0
    assert rep.at["auto", "tier1_reviews"] == n_auto_acc
    cal = audit.auto_calibration(min_cell=1)
    assert set(cal.auto_verdict) == {"agree", "disagree", "uncertain"}
    dis = cal[cal.auto_verdict == "disagree"].iloc[0]
    assert dis.ds_same_verdict == dis.ds_reviewed                        # auto was right on every flag here


def test_replace_mode_skips_data_scientists(tmp_path):
    sample, ctx, wrong = make_case()
    truth = truth_map(sample)
    items = phenotype_audit_items(sample)
    audit = LabelAudit(items, AnnotationStore(tmp_path / "a.csv"), DS, CARD, spot_check_min=5,
                       auto_tier1="replace", seed=0)
    assert audit.tier1_queue("ds_ana").shape[0] == len(items)            # nothing run yet: people would audit
    av = audit_phenotypes(items, ctx)
    audit.submit_auto(av)
    assert audit.tier1_queue("ds_ana").empty
    q2 = audit.tier2_queue("dr_cole")
    assert (q2.reason == "disagreement").sum() == (av.verdict == "disagree").sum()
    assert (q2.reason == "uncertain").sum() == (av.verdict == "uncertain").sum()
    for r in q2.itertuples():
        audit.submit_tier2(r.item_id, r.task, truth[r.item_id], "dr_cole")
    f = audit.final_labels()
    assert (f.tier1_by == "auto").all() and f.final_value.notna().all()
    # re-running the auditor replaces its earlier verdicts, never people's
    audit.submit_auto(av.assign(verdict="uncertain", value=None))
    assert (audit.auto_verdicts().verdict == "uncertain").all()
    assert len(audit._tier(2)) == len(q2)


def test_mode_and_reserved_name(tmp_path):
    items = pd.DataFrame({"item_id": ["1"], "task": ["phenotype:SCAD"], "auto_value": ["yes"]})
    with pytest.raises(ValueError):
        LabelAudit(items, AnnotationStore(tmp_path / "a.csv"), DS, CARD, auto_tier1="always")
    with pytest.raises(ValueError):
        LabelAudit(items, AnnotationStore(tmp_path / "a.csv"), [AUTO_AUDITOR], CARD)
    from pathlib import Path
    from mcp_adapter.config import StudyConfig
    cfg = StudyConfig.load(Path(__file__).resolve().parents[1] / "config/study.yaml")
    cfg.audit.update(data_scientists=["ds_ana"], cardiologists=["dr_cole"])
    a = LabelAudit.from_config(items, AnnotationStore(tmp_path / "a.csv"), cfg)
    assert a.auto_tier1 == "off" and a.auto_spot_check_fraction == 0.20
    cfg.audit["auto_tier1"] = "triage"
    assert LabelAudit.from_config(items, AnnotationStore(tmp_path / "b.csv"), cfg).auto_tier1 == "triage"


def test_audit_echo(tmp_path):
    from tests.test_annotate import PERIOD, make_clip, write_dicom
    rows = []
    for k, v in enumerate(["PLAX", "PSAX", "A4C", "A2C"] * 2):
        fr, ed = make_clip(v, n=40, size=96, phase0=k % 5, seed=k)
        p = tmp_path / f"c{k}.dcm"
        write_dicom(p, fr, desc=v)
        rows.append({"person_id": k, "record_date": "2020-01-01", "path": str(p)})
    man = pd.DataFrame(rows)
    store = AnnotationStore(tmp_path / "echo.csv")
    E.run_auto(man, store, tmp_path / "cache")
    items = echo_audit_items(store, man)
    # one clip gets a wrong ES (before ED) to simulate a phase error
    bad = items.item_id.iloc[0]
    v0 = json.loads(items.auto_value.iloc[0])
    items.loc[0, "auto_value"] = json.dumps({**v0, "es": max(0, v0["ed"] - 1)}, sort_keys=True)
    out = audit_echo(items, store, tmp_path / "cache").set_index("item_id")
    assert out.at[bad, "verdict"] == "uncertain" and "ES" in out.at[bad, "note"]
    good = out.drop(index=bad)
    assert (good.verdict == "agree").all(), good[["verdict", "note"]]
    audit = LabelAudit(items, AnnotationStore(tmp_path / "audit.csv"), DS, CARD, spot_check_min=1,
                       auto_tier1="triage")
    audit.submit_auto(out.reset_index())
    assert set(audit.tier1_queue("ds_ana").item_id) == set(out[out.verdict != "agree"].index)
    assert (audit.tier2_queue().reason == "spot-check").sum() >= 1
