"""Reviewer training on the synthetic kit: answer key, practice feedback, mock audit, qualification."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from annotate.audit import ECHO_TASK, LabelAudit
from annotate.store import AnnotationStore
from annotate.synthetic_cases import KEY_AUTHOR, TEMPLATES, build_kit, main, sizes_from_config
from annotate.training import (TrainingScreen, load_training_sets, marks_from_config, pair_kappa,
                               qualification_items, qualification_report, reference_key, refresher_items,
                               refresher_report)


@pytest.fixture(scope="module")
def kit(tmp_path_factory):
    out = tmp_path_factory.mktemp("kit")
    r = build_kit(out, n_reference=36, n_practice=18, n_audit=36, n_echo_reference=10,
                  n_echo_practice=5, n_echo_audit=10)
    return {**r, "sets": load_training_sets(r["sets"]), "ctx": pd.read_pickle(r["context"])}


def answer(scr, key, wrong_every=0):
    """Answer every item in the screen's queue from the key, getting every `wrong_every`-th wrong."""
    n = 0
    while (c := scr.current()) is not None:
        v, _ = key[(c["item_id"], c["task"])]
        bad = wrong_every and n % wrong_every == 0
        if c["kind"] == "echo":
            x = json.loads(v)
            if bad:
                x["view"] = "OTHER" if x["view"] != "OTHER" else "A4C"
                x["ed"], x["es"] = (2, 10) if x["view"] != "OTHER" else (None, None)
            scr.submit(view=x["view"], ed=x["ed"], es=x["es"])
        else:
            scr.submit(value={"yes": "no", "no": "yes", "unsure": "yes"}[v] if bad else v)
        n += 1


def test_kit_contents(kit):
    s, ctx = kit["sets"], kit["ctx"]
    assert set(s.set) == {"practice", "reference", "audit"}
    ref = s[s.set == "reference"]
    for part in "AB":
        p = ref[ref.part == part]
        assert {f"phenotype:{c}" for c in TEMPLATES} <= set(p.task) and (p.task == ECHO_TASK).any()
    assert len(ref[ref.part == "A"]) == len(ref[ref.part == "B"])
    key = reference_key(AnnotationStore(kit["store"]))
    assert all((i, t) in key for i, t in zip(s.item_id, s.task))
    answers = pd.Series([key[(i, t)][0] for i, t in zip(s.item_id, s.task) if t != ECHO_TASK])
    assert set(answers) == {"yes", "no", "unsure"}
    assert all(note for _, note in key.values())                                     # every answer has a lesson
    assert ctx["notes"].text.str.contains("SYNTHETIC|spontaneous|LVEF", regex=True).any()
    # the iatrogenic SCAD trap is in the kit: coded positive, answer No
    audit = pd.read_csv(kit["audit_items"], dtype={"item_id": str})
    scad = audit[audit.task == "phenotype:SCAD"]
    assert any(key[(i, "phenotype:SCAD")][0] == "no" and a == "yes" for i, a in zip(scad.item_id, scad.auto_value))


def test_practice_qualify_and_score(kit, tmp_path):
    import shutil
    store_path = tmp_path / "training.csv"; shutil.copy(kit["store"], store_path)
    store, s = AnnotationStore(store_path), kit["sets"]
    kw = dict(phenotype_context=kit["ctx"], echo_cache=kit["cache"])

    # the clinical lead overrides one generated answer
    lead = TrainingScreen(s, store, "dr_lead", "reference", **kw)
    first = lead.current()
    assert lead.remaining() == len(s[s.set != "audit"])
    if first["kind"] == "phenotype":
        lead.submit(value="unsure", note="lead: borderline")
        assert reference_key(store, "dr_lead")[(first["item_id"], first["task"])][0] == "unsure"
    key = reference_key(store, "dr_lead")

    pr = TrainingScreen(s, store, "ds_ana", "practice", lead="dr_lead", **kw)
    c = pr.current()
    v, _ = key[(c["item_id"], c["task"])]
    if c["kind"] == "echo":
        pr.submit(view="OTHER" if json.loads(v)["view"] != "OTHER" else "A4C", ed=1, es=9)
    else:
        pr.submit(value="yes" if v != "yes" else "no")
    f = pr.feedback()
    assert f["match"] is False and f["lead_note"]

    who = {"ds_ana": "data_scientist", "ds_ben": "data_scientist", "dr_cole": "cardiologist", "dr_dee": "cardiologist"}
    for name in who:
        answer(TrainingScreen(s, store, name, "qualification", attempt=1, lead="dr_lead", **kw), key,
               wrong_every=3 if name == "ds_ben" else 0)
    rep = qualification_report(store, s, who, attempt=1, lead="dr_lead").set_index("trainee")
    assert rep.at["ds_ana", "passed"] and rep.at["dr_cole", "passed"] and not rep.at["ds_ben", "passed"]
    assert rep.at["ds_ana", "records_unsure"] == 0          # correct Unsure answers are not penalised
    assert pair_kappa(store, s, "dr_cole", "dr_dee")["kappa"] == 1.0
    answer(TrainingScreen(s, store, "ds_ben", "qualification", attempt=2, lead="dr_lead", **kw), key)
    assert qualification_report(store, s, {"ds_ben": "data_scientist"}, attempt=2, lead="dr_lead").passed.iloc[0]
    q1, q2 = qualification_items(s, 1), qualification_items(s, 2)
    assert not set(zip(q1.item_id, q1.task)) & set(zip(q2.item_id, q2.task))

    items = refresher_items(s, 1)
    answer(TrainingScreen(s, store, "ds_ana", "refresher", items=items, month=1, lead="dr_lead", **kw), key)
    rr = refresher_report(store, s, 1, ["ds_ana"], lead="dr_lead")
    assert rr.agreement.iloc[0] == 1.0
    TrainingScreen(s, store, "ds_ben", "practice", lead="dr_lead", **kw).show()


def test_mock_audit_rehearsal(kit, tmp_path):
    """Trainees rehearse the real audit on the kit: automatic audit in triage mode, tier 1, tier 2."""
    from annotate.audit_ui import AuditScreen
    from annotate.auto_audit import audit_echo, audit_phenotypes
    items = pd.read_csv(kit["audit_items"], dtype={"item_id": str})
    key = reference_key(AnnotationStore(kit["store"]))
    audit = LabelAudit(items, AnnotationStore(tmp_path / "mock_audit.csv"), ["ds_ana"], ["dr_cole"],
                       spot_check_min=3, auto_tier1="triage")
    pv = audit_phenotypes(items, kit["ctx"])
    ev = audit_echo(items[items.task == ECHO_TASK], AnnotationStore(kit["echo_store"]), kit["cache"])
    audit.submit_auto(pd.concat([pv, ev], ignore_index=True))
    ph = items[items.task != ECHO_TASK]
    wrong = {i for i, t, a in zip(ph.item_id, ph.task, ph.auto_value) if key[(i, t)][0] != a}
    flagged = set(pv[pv.verdict != "agree"].item_id)
    assert len(wrong & flagged) / len(wrong) >= 0.7          # the automatic audit flags most planted errors
    q = audit.tier1_queue("ds_ana")
    assert 0 < len(q) < len(items)
    scr = AuditScreen(audit, "ds_ana", tier=1, phenotype_context=kit["ctx"], echo_cache=kit["cache"])
    assert scr.current().get("auto_audit")
    while (c := scr.current()) is not None:
        gold, _ = key[(c["item_id"], c["task"])]
        if c["kind"] == "echo":
            ok = json.loads(gold)["view"] == json.loads(c["auto_value"])["view"]
            scr.submit(verdict="agree") if ok else scr.submit(verdict="uncertain")
        elif gold == "unsure":
            scr.submit(verdict="uncertain")
        else:
            scr.submit(verdict="agree" if gold == c["auto_value"] else "disagree")
    card = AuditScreen(audit, "dr_cole", tier=2, phenotype_context=kit["ctx"], echo_cache=kit["cache"])
    while (c := card.current()) is not None:
        gold, _ = key[(c["item_id"], c["task"])]
        if c["kind"] == "echo":
            g = json.loads(gold); card.submit(view=g["view"], ed=g["ed"], es=g["es"])
        else:
            card.submit(value=gold)
    rep = audit.report(min_cell=1)
    assert set(rep.tier1_by) == {"auto", "human"}


def test_cli_and_config(tmp_path):
    from pathlib import Path
    from mcp_adapter.config import StudyConfig
    cfg = StudyConfig.load(Path(__file__).resolve().parents[1] / "config/study.yaml")
    assert sizes_from_config(cfg)["n_reference"] == 40 and marks_from_config(cfg)["data_scientist"]["echo_frames"] == 17
    main(["--out", str(tmp_path / "k"), "--seed", "1"])
    assert (tmp_path / "k" / "sets.csv").exists() and (tmp_path / "k" / "README.txt").read_text().startswith("SYNTHETIC")
    assert KEY_AUTHOR in set(pd.read_csv(tmp_path / "k" / "training.csv").annotator)
