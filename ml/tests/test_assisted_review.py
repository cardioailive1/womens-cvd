"""Assisted review: the data scientists accept or reject the automatic audit's recommendation and
sign off; the cardiologist then adjudicates the escalated items the same way and signs off."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from annotate.audit import LabelAudit, apply_to_labels, echo_value, phenotype_audit_items
from annotate.audit_ui import AuditScreen
from annotate.store import AnnotationStore

DS, CARD = ["ds_ana", "ds_ben"], ["dr_cole"]
TASK = "phenotype:SCAD"


@pytest.fixture
def assisted(tmp_path):
    rng = np.random.default_rng(7)
    n = 80
    sample = pd.DataFrame({"person_id": np.arange(2000, 2000 + n), "condition": "SCAD",
                           "algo_label": rng.integers(0, 2, n)})
    items = phenotype_audit_items(sample)
    audit = LabelAudit(items, AnnotationStore(tmp_path / "a.csv"), DS, CARD, spot_check_min=5,
                       spot_check_fraction=0.10, seed=1, auto_tier1="assisted", ds_review="all")
    flip = {"yes": "no", "no": "yes"}
    verdicts = []
    for k, r in enumerate(items.itertuples()):
        if k % 10 == 0:          # the automatic audit proposes a correction
            verdicts.append((r.item_id, r.task, "disagree", flip[r.auto_value]))
        elif k % 10 == 1:        # it could not check
            verdicts.append((r.item_id, r.task, "uncertain", None))
        else:
            verdicts.append((r.item_id, r.task, "agree", None))
    audit.submit_auto(pd.DataFrame(verdicts, columns=["item_id", "task", "verdict", "value"]).assign(
        confidence=0.9, note="stub"))
    return audit, items


def test_recommendations(assisted):
    audit, items = assisted
    r0 = audit.recommendation(items.item_id[0], TASK)
    assert r0["kind"] == "correct" and r0["proposed"] != r0["auto_value"]
    assert audit.recommendation(items.item_id[1], TASK)["kind"] == "unconfirmed"
    r2 = audit.recommendation(items.item_id[2], TASK)
    assert r2["kind"] == "keep" and r2["proposed"] == r2["auto_value"]


def test_assisted_flow_with_signoffs(assisted):
    audit, items = assisted
    # every item reaches the data scientists in assisted mode (auto-agreed ones too)
    assert len(audit.tier1_queue("ds_ana")) == len(items)
    with pytest.raises(RuntimeError, match="signed off"):
        audit.tier2_queue("dr_cole")                                   # cardiologist waits for sign-off
    with pytest.raises(RuntimeError, match="still in"):
        audit.sign_off("ds_ana", 1)                                    # cannot sign off with work left
    for k, r in enumerate(items.itertuples()):
        ds = DS[k % 2]
        if k % 10 == 0 and k % 20 == 0:
            audit.submit_tier1_decision(r.item_id, r.task, "reject", ds)      # keeps the automatic label
        elif k == 3:
            audit.submit_tier1_decision(r.item_id, r.task, "uncertain", ds, note="unclear notes")
        else:
            audit.submit_tier1_decision(r.item_id, r.task, "accept", ds)
    st = audit._t1_state().set_index("item_id").outcome
    assert st[items.item_id[0]] == "accepted"                          # rejected a wrong correction
    assert st[items.item_id[10]] == "disagreement"                     # accepted a correction
    assert st[items.item_id[3]] == "uncertain"
    audit.sign_off("ds_ana", 1)
    assert not audit.tier1_signed_off()                                # ds_ben has not signed off
    audit.sign_off("ds_ben", 1)
    assert audit.tier1_signed_off()

    scr = AuditScreen(audit, "dr_cole", tier=2, phenotype_context=_ctx())
    seen_rec, seen_blind = 0, 0
    labels = pd.DataFrame({"person_id": items.item_id.astype(int), "y_SCAD":
                           (items.auto_value == "yes").astype(float), "y_fmd_trigger": 0.0})
    with pytest.raises(RuntimeError, match="signed off"):
        apply_to_labels(audit, labels)                                 # not before the cardiologist signs
    while (c := scr.current()) is not None:
        if "recommendation" in c:
            seen_rec += 1; scr.submit(decision="accept")
        else:
            seen_blind += 1; scr.submit(value="no")                    # spot-check: labelled blind
        assert "auto_value" not in c and "reason" not in c
    assert seen_rec == 5 and seen_blind >= 5                           # 4 accepted corrections + 1 uncertain
    scr.sign_off()
    assert audit.final_signed_off()
    new, changed = apply_to_labels(audit, labels)
    assert changed > 0
    rep = audit.assist_report(min_cell=1).set_index(["tier", "kind"])
    assert int(rep.loc[(1, "correct"), "reject"]) == 4 and int(rep.loc[(2, "correct"), "accept"]) == 4
    st = audit.signoff_status()
    assert st.signed_off.all()


def test_reject_echo_needs_correction(tmp_path):
    from annotate.audit import ECHO_TASK
    items = pd.DataFrame({"item_id": ["c1"], "task": [ECHO_TASK], "auto_value": [echo_value("A4C", 3, 9)]})
    audit = LabelAudit(items, AnnotationStore(tmp_path / "e.csv"), DS, CARD, auto_tier1="assisted")
    audit.submit_auto(pd.DataFrame([("c1", ECHO_TASK, "agree", None, 0.9, "ok")],
                                   columns=["item_id", "task", "verdict", "value", "confidence", "note"]))
    with pytest.raises(ValueError, match="own label"):
        audit.submit_tier1_decision("c1", ECHO_TASK, "reject", "ds_ana")
    audit.submit_tier1_decision("c1", ECHO_TASK, "reject", "ds_ana", value=echo_value("A2C", 3, 9))
    assert audit._t1_state().outcome.iloc[0] == "disagreement"


def test_from_config_assisted(tmp_path):
    from pathlib import Path
    from mcp_adapter.config import StudyConfig
    cfg = StudyConfig.load(Path(__file__).resolve().parents[1] / "config/study.yaml")
    assert cfg.audit["auto_tier1"] == "assisted" and cfg.audit["blind_spot_check"] is True
    assert cfg.audit["ds_review"] == "uncertain"
    cfg.audit.update(data_scientists=DS, cardiologists=CARD)
    items = pd.DataFrame({"item_id": ["1"], "task": [TASK], "auto_value": ["yes"]})
    a = LabelAudit.from_config(items, tmp_path / "s.csv", cfg)
    assert a.auto_tier1 == "assisted" and a.blind_spot_check and a.ds_review == "uncertain"
    with pytest.raises(ValueError, match="ds_review"):
        LabelAudit(items, tmp_path / "t.csv", DS, CARD, auto_tier1="assisted", ds_review="some")


def _ctx():
    cols = {"diagnoses": ["person_id", "rel_day", "code", "name"], "procedures": ["person_id", "rel_day", "code", "name"],
            "labs": ["person_id", "rel_day", "name", "value", "unit"], "notes": ["person_id", "rel_day", "title", "text"]}
    return {k: pd.DataFrame(columns=v) for k, v in cols.items()}


def test_ds_review_uncertain_only(assisted):
    """ds_review = uncertain: the data scientists see only the automatic 'uncertain' items; 'agree'
    items count as accepted (auto spot-check), 'disagree' items go straight to the cardiologist."""
    audit, items = assisted
    audit.ds_review = "uncertain"
    unc = {r.item_id for k, r in enumerate(items.itertuples()) if k % 10 == 1}
    dis = {r.item_id for k, r in enumerate(items.itertuples()) if k % 10 == 0}
    q = pd.concat([audit.tier1_queue(d) for d in DS]).drop_duplicates()
    assert set(q.item_id) == unc and len(unc) == 8
    for k, r in enumerate(audit.tier1_queue("ds_ana").itertuples()):
        audit.submit_tier1_decision(r.item_id, r.task, "accept" if k % 2 else "reject", "ds_ana")
    assert audit.tier1_queue("ds_ben").empty
    audit.sign_off("ds_ana", 1)
    assert audit.tier1_signed_off()                                   # ds_ben answered nothing: no sign-off needed
    q2 = audit.tier2_queue("dr_cole")
    reasons = q2.set_index("item_id").reason
    assert dis <= set(q2.item_id) and all(reasons[i] == "disagreement" for i in dis)
    spot = set(reasons[reasons == "spot-check"].index)
    agreed = set(items.item_id) - unc - dis
    assert not spot & dis and spot & agreed                           # spot-checks: auto- and DS-accepted items
    r = audit.tier2_recommendation(next(iter(dis)), TASK, "disagreement")
    assert r["kind"] == "correct"
    st = audit.final_labels().set_index("item_id")
    assert (st.loc[list(agreed - spot), "status"] == "tier 1 confirmed").all()
    assert (st.loc[list(agreed), "tier1_by"] == "auto").all()
