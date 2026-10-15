"""Two-tier label audit: data scientists audit everything, a cardiologist adjudicates
disagreements, uncertain items and a blinded spot-check."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from annotate import echo as E
from annotate.audit import (ECHO_TASK, LabelAudit, apply_to_echo_store, apply_to_labels, echo_audit_items,
                            echo_value, phenotype_audit_items, same)
from annotate.audit_ui import AuditScreen
from annotate.store import AnnotationStore

DS, CARD = ["ds_ana", "ds_ben"], ["dr_cole"]


def empty_ctx():
    cols = {"diagnoses": ["person_id", "rel_day", "code", "name"],
            "procedures": ["person_id", "rel_day", "code", "name"],
            "labs": ["person_id", "rel_day", "name", "value", "unit"],
            "notes": ["person_id", "rel_day", "title", "text"]}
    return {k: pd.DataFrame(columns=v) for k, v in cols.items()}


@pytest.fixture
def pheno(tmp_path):
    rng = np.random.default_rng(1)
    n = 120
    sample = pd.DataFrame({"person_id": np.arange(1000, 1000 + n), "condition": "SCAD",
                           "algo_label": rng.integers(0, 2, n)})
    truth = sample.algo_label.to_numpy().copy()
    wrong = rng.choice(n, 12, replace=False)                    # the rules got 12 labels wrong
    truth[wrong] = 1 - truth[wrong]
    sample["truth"] = truth
    audit = LabelAudit(phenotype_audit_items(sample), AnnotationStore(tmp_path / "audit.csv"), DS, CARD,
                       spot_check_fraction=0.10, spot_check_min=10, seed=3)
    return audit, sample, set(wrong.tolist())


def test_roles_are_enforced(pheno, tmp_path):
    audit, sample, _ = pheno
    i = str(sample.person_id.iloc[0])
    with pytest.raises(PermissionError):
        audit.submit_tier1(i, "phenotype:SCAD", "agree", "dr_cole")          # cardiologist at tier 1
    with pytest.raises(PermissionError):
        audit.submit_tier2(i, "phenotype:SCAD", "yes", "ds_ana")             # data scientist at tier 2
    with pytest.raises(ValueError):
        LabelAudit(audit.items, audit.store, ["x"], ["x"])                   # same person in both tiers
    audit.submit_tier1(i, "phenotype:SCAD", "uncertain", "ds_ana")
    later = LabelAudit(audit.items, audit.store, ["ds_ben"], ["ds_ana"])     # roles change later...
    with pytest.raises(PermissionError):
        later.submit_tier2(i, "phenotype:SCAD", "yes", "ds_ana")             # ...still can't self-adjudicate


def test_two_tier_flow(pheno):
    audit, sample, wrong = pheno
    truth = dict(zip(sample.person_id.astype(str), np.where(sample.truth == 1, "yes", "no")))
    # tier 1: the data scientists split the work; they catch most errors, miss 3, unsure on 4
    missed = set(sorted(wrong)[:3])
    k = 0
    for ds in DS:
        while True:
            q = audit.tier1_queue(ds)
            if q.empty or k >= 60 * (DS.index(ds) + 1):
                break
            r = q.iloc[0]
            idx = int(r.item_id) - 1000
            if idx in wrong and idx not in missed:
                audit.submit_tier1(r.item_id, r.task, "disagree", ds)       # binary: flips the label
            elif k % 30 == 7:
                audit.submit_tier1(r.item_id, r.task, "uncertain", ds, note="conflicting notes")
            else:
                audit.submit_tier1(r.item_id, r.task, "agree", ds)
            k += 1
    assert all(audit.tier1_queue(d).empty for d in DS)

    q2 = audit.tier2_queue("dr_cole")
    counts = q2.reason.value_counts()
    n_dis, n_unc = counts.get("disagreement", 0), counts.get("uncertain", 0)
    assert n_dis == len(wrong) - len(missed)                               # every caught error goes up
    assert n_unc >= 1
    accepted = len(sample) - n_dis - n_unc
    assert counts.get("spot-check", 0) == max(10, int(np.ceil(0.10 * accepted)))
    assert q2.reason.tolist() != sorted(q2.reason.tolist())                  # shuffled, not grouped

    # tier 2 screen is blinded
    scr = AuditScreen(audit, "dr_cole", tier=2, phenotype_context=empty_ctx())
    c = scr.current()
    assert "auto_value" not in c and "reason" not in c
    while scr.current() is not None:
        c = scr.current()
        scr.submit(value=truth[c["item_id"]])
    assert audit.tier2_queue().empty

    f = audit.final_labels()
    assert (f.status.isin(["cardiologist", "tier 1 confirmed"])).all()
    adjudicated = f[f.status == "cardiologist"]
    assert all(v == truth[i] for i, v in zip(adjudicated.item_id, adjudicated.final_value))

    rep = audit.report(min_cell=1)
    r = rep.iloc[0]
    assert r.tier1_reviews == len(sample)
    assert r.disagreements_auto_upheld == 0                                # DS were right every time
    assert r.spot_checks_done == counts.get("spot-check", 0)
    assert 0 <= r.spot_check_agreement <= 1

    from annotate.audit import phenotype_validation
    val = phenotype_validation(audit, sample, min_cell=1)
    ppv = val[val.metric == "PPV"].iloc[0]
    true_ppv = (sample[sample.algo_label == 1].truth == 1).mean()
    assert abs(float(ppv.value) - true_ppv) <= len(missed) / (sample.algo_label == 1).sum() + 1e-9

    labels = pd.DataFrame({"person_id": sample.person_id, "y_SCAD": sample.algo_label.astype(float),
                           "y_fmd_trigger": sample.algo_label.astype(float)})
    new, changed = apply_to_labels(audit, labels)
    assert changed >= len(wrong) - len(missed)
    assert new.y_fmd_trigger.equals(new.y_SCAD)


def test_suppression_in_report(pheno):
    audit, sample, _ = pheno
    r = sample.person_id.astype(str).iloc[0]
    audit.submit_tier1(r, "phenotype:SCAD", "agree", "ds_ana")
    rep = audit.report(min_cell=11)
    assert rep.tier1_reviews.iloc[0] == "<11" and rep.tier1_agreement.iloc[0] == "suppressed"


def test_echo_values_compare_with_frame_tolerance():
    a = echo_value("A4C", 3, 11)
    assert same(ECHO_TASK, a, echo_value("A4C", 5, 9))
    assert not same(ECHO_TASK, a, echo_value("A4C", 7, 11))
    assert not same(ECHO_TASK, a, echo_value("A2C", 3, 11))
    assert same(ECHO_TASK, echo_value("OTHER"), echo_value("OTHER", 1, 2))


def test_echo_audit_end_to_end(tmp_path):
    from tests.test_annotate import make_clip, write_dicom
    rows, truth = [], {}
    for k, v in enumerate(["PLAX", "PSAX", "A4C", "A2C"] * 3):
        fr, ed = make_clip(v, n=32, size=96, phase0=k % 5, seed=k)
        p = tmp_path / f"c{k}.dcm"
        write_dicom(p, fr, desc={"A4C": "Apical 4CH", "A2C": "A2C", "PLAX": "PLAX", "PSAX": "PSAX"}[v])
        rows.append({"person_id": k, "record_date": "2020-01-01", "path": str(p)})
        truth[E.item_id(str(p))] = (v, ed)
    man = pd.DataFrame(rows)
    store = AnnotationStore(tmp_path / "echo.csv")
    E.run_auto(man, store, tmp_path / "cache")
    items = echo_audit_items(store, man)
    assert len(items) == 12 and (items.task == ECHO_TASK).all()

    audit = LabelAudit(items, AnnotationStore(tmp_path / "audit.csv"), DS, CARD, spot_check_min=2, seed=0)
    scr = AuditScreen(audit, "ds_ana", tier=1, echo_cache=tmp_path / "cache")
    first = scr.current()
    assert json.loads(first["auto_value"])["view"] in E.VIEWS and first["frames"].ndim == 3
    n = 0
    while (c := scr.current()) is not None:
        if n == 0:                       # pretend the rule mislabelled this clip: DS corrects to PSAX
            scr.submit(verdict="disagree", view="PSAX", ed=1, es=8)
        elif n == 1:
            scr.submit(verdict="uncertain")
        else:
            scr.submit(verdict="agree")
        n += 1
    q2 = audit.tier2_queue("dr_cole")
    assert set(q2.reason) == {"disagreement", "uncertain", "spot-check"}
    card = AuditScreen(audit, "dr_cole", tier=2, echo_cache=tmp_path / "cache")
    assert "auto_value" not in card.current()
    while (c := card.current()) is not None:
        v, ed = truth[c["item_id"]]
        card.submit(view=v, ed=ed, es=ed + 8)
    written = apply_to_echo_store(audit, store)
    assert written >= len(q2)
    fin = E.finalize(man, store)
    assert all(truth[E.item_id(p)][0] == v for p, v in zip(fin.path, fin.view))   # audited views win
    card.show(); AuditScreen(audit, "ds_ben", tier=1, echo_cache=tmp_path / "cache").show()


def test_from_config(tmp_path):
    from pathlib import Path
    from mcp_adapter.config import StudyConfig
    cfg = StudyConfig.load(Path(__file__).resolve().parents[1] / "config/study.yaml")
    items = pd.DataFrame({"item_id": ["1"], "task": ["phenotype:SCAD"], "auto_value": ["yes"]})
    with pytest.raises(ValueError, match="audit.data_scientists"):
        LabelAudit.from_config(items, AnnotationStore(tmp_path / "a.csv"), cfg)   # roles not filled yet
    cfg.audit.update(data_scientists=["ds_ana"], cardiologists=["dr_cole"], tier1_per_item=2)
    a = LabelAudit.from_config(items, AnnotationStore(tmp_path / "a.csv"), cfg)
    assert a.tier1_per_item == 2 and a.spot_check_min == 20
