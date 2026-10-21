"""Reviewer training: reference set, practice feedback, qualification scoring."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from annotate import echo as E
from annotate.audit import ECHO_TASK
from annotate.store import AnnotationStore
from annotate.training import (TrainingScreen, draw_training_sets, exclude_training, load_training_sets,
                               marks_from_config, pair_kappa, qualification_items, qualification_report,
                               refresher_items, save_training_sets)

PERIOD = 16


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    from tests.test_annotate import make_clip, write_dicom
    tmp = tmp_path_factory.mktemp("train")
    rng = np.random.default_rng(0)
    n = 300
    conds = ["SCAD", "PPCM", "mace"]
    labels = pd.DataFrame({"person_id": np.arange(n)})
    for c in conds:
        labels[f"y_{c}"] = rng.integers(0, 2, n).astype(float)
    labels["m_mace"] = 1.0
    ev = pd.DataFrame({"person_id": rng.choice(n, 80), "grp": rng.choice(["cond_SCAD", "cond_PPCM", "out_mi"], 80)})
    ctx = {k: pd.DataFrame({"person_id": np.arange(n), "rel_day": 5, **cols})
           for k, cols in {"diagnoses": {"code": "X", "name": "x"}, "procedures": {"code": "X", "name": "x"},
                           "labs": {"name": "bnp", "value": 1.0, "unit": "pg/mL"},
                           "notes": {"title": "note", "text": "text"}}.items()}
    rows, truth = [], {}
    for k in range(24):
        v = ["PLAX", "PSAX", "A4C", "A2C"][k % 4]
        fr, ed = make_clip(v, n=40, size=64, phase0=k % 5, seed=k)
        p = tmp / f"c{k}.dcm"; write_dicom(p, fr, desc=v)
        rows.append({"person_id": k, "record_date": "2020-01-01", "path": str(p)})
        truth[E.item_id(str(p))] = (v, ed)
    man = pd.DataFrame(rows)
    echo_store = AnnotationStore(tmp / "echo.csv")
    E.run_auto(man, echo_store, tmp / "cache")
    sets = draw_training_sets(labels, ev, echo_store, man, n_reference=24, n_practice=12,
                              n_echo_reference=16, n_echo_practice=8, conditions=conds)
    return {"tmp": tmp, "labels": labels, "ev": ev, "ctx": ctx, "sets": sets, "truth": truth,
            "cache": tmp / "cache", "man": man}


def answer_key(world, item_id, task):
    """The 'true' answer used by the simulated lead."""
    if task == ECHO_TASK:
        v, ed = world["truth"][item_id]
        return {"view": v, "ed": ed, "es": ed + PERIOD // 2}
    lab = world["labels"].set_index("person_id").loc[int(item_id), "y_" + task.split(":")[1]]
    return {"value": "yes" if lab == 1 else "no"}


def work(scr, world, wrong_every=0, unsure_every=0):
    n = 0
    while (c := scr.current()) is not None:
        a = answer_key(world, c["item_id"], c["task"])
        if c["kind"] == "echo":
            if wrong_every and n % wrong_every == 0:
                a = {**a, "view": "OTHER" if a["view"] != "OTHER" else "A4C"}
            scr.submit(view=a["view"], ed=a.get("ed"), es=a.get("es"), note="lead note" if scr.mode == "reference" else None)
        else:
            v = a["value"]
            if unsure_every and n % unsure_every == 1:
                v = "unsure"
            elif wrong_every and n % wrong_every == 0:
                v = {"yes": "no", "no": "yes"}[v]
            scr.submit(value=v, note="lead note" if scr.mode == "reference" else None)
        n += 1


def test_sets_are_balanced_and_excluded(world, tmp_path):
    s = world["sets"]
    assert (s.set == "reference").sum() == 24 + 16 and (s.set == "practice").sum() == 12 + 8
    ref = s[s.set == "reference"]
    for part in "AB":
        p = ref[ref.part == part]
        assert set(p.task) >= {"phenotype:SCAD", "phenotype:PPCM", "phenotype:mace", ECHO_TASK}
    save_training_sets(s, tmp_path / "sets.csv")
    assert load_training_sets(tmp_path / "sets.csv").equals(s.astype({"item_id": str}))
    from annotate.phenotype import sample_for_review
    live = sample_for_review(world["labels"], world["ev"], ["SCAD", "PPCM", "mace"], n_pos=60, n_neg=60)
    left = exclude_training(live, s)
    assert not set(left.person_id.astype(str)) & set(s.item_id)
    from annotate.audit import echo_audit_items
    items = echo_audit_items(AnnotationStore(world["tmp"] / "echo.csv"), world["man"])
    assert not set(exclude_training(items, s).item_id) & set(s.item_id)


def test_full_training_flow(world, tmp_path):
    s, store = world["sets"], AnnotationStore(tmp_path / "training.csv")
    kw = dict(phenotype_context=world["ctx"], echo_cache=world["cache"], lead="dr_lead")
    with pytest.raises(ValueError, match="reference answer"):
        TrainingScreen(s, store, "ds_ana", "practice", **kw)            # lead has not written the key

    work(TrainingScreen(s, store, "dr_lead", "reference", **kw), world)
    assert TrainingScreen(s, store, "dr_lead", "reference", **kw).remaining() == 0

    # practice gives feedback after each answer
    pr = TrainingScreen(s, store, "ds_ana", "practice", **kw)
    c = pr.current()
    a = answer_key(world, c["item_id"], c["task"])
    if c["kind"] == "echo":
        pr.submit(view="OTHER")
    else:
        pr.submit(value={"yes": "no", "no": "yes"}[a["value"]])
    f = pr.feedback()
    assert f["match"] is False and f["lead_note"] == "lead note"
    with pytest.raises(ValueError):
        pr.submit(value="maybe") if pr.current()["kind"] == "phenotype" else pr.submit(view="A4C", ed=5, es=2)

    # qualification: one data scientist passes, one fails, then retakes part B
    work(TrainingScreen(s, store, "ds_ana", "qualification", attempt=1, **kw), world)
    work(TrainingScreen(s, store, "ds_ben", "qualification", attempt=1, **kw), world, wrong_every=3)
    work(TrainingScreen(s, store, "dr_cole", "qualification", attempt=1, **kw), world)
    work(TrainingScreen(s, store, "dr_dee", "qualification", attempt=1, **kw), world)
    rep = qualification_report(store, s, {"ds_ana": "data_scientist", "ds_ben": "data_scientist",
                                          "dr_cole": "cardiologist", "dr_dee": "cardiologist"},
                               attempt=1, lead="dr_lead").set_index("trainee")
    q1 = qualification_items(s, 1)
    assert rep.at["ds_ana", "items"] == len(q1) and rep.at["ds_ana", "complete"]
    assert rep.at["ds_ana", "passed"] and rep.at["dr_cole", "passed"]
    assert not rep.at["ds_ben", "passed"]
    k = pair_kappa(store, s, "dr_cole", "dr_dee", attempt=1)
    assert k["kappa"] == 1.0 and k["items"] == len(q1)

    work(TrainingScreen(s, store, "ds_ben", "qualification", attempt=2, **kw), world)
    rep2 = qualification_report(store, s, {"ds_ben": "data_scientist"}, attempt=2, lead="dr_lead")
    assert rep2.passed.iloc[0]
    q2 = qualification_items(s, 2)
    assert not set(zip(q2.item_id, q2.task)) & set(zip(q1.item_id, q1.task))     # the retake is a fresh set

    # monthly refresher: 10 items, different months differ
    r1, r2 = refresher_items(s, 1), refresher_items(s, 2)
    assert len(r1) == 10 and not r1.equals(r2)
    scr = TrainingScreen(s, store, "ds_ana", "refresher", items=r1, month=1, **kw)
    assert scr.remaining() == 10 and scr.feedback() is None
    work(scr, world)
    from annotate.training import refresher_report
    rr = refresher_report(store, s, 1, ["ds_ana", "ds_ben"], lead="dr_lead").set_index("reviewer")
    assert rr.at["ds_ana", "agreement"] == 1.0 and rr.at["ds_ben", "answered"] == 0
    TrainingScreen(s, store, "ds_ana", "practice", **kw).show()                 # widgets build


def test_marks_from_config():
    from pathlib import Path
    from mcp_adapter.config import StudyConfig
    cfg = StudyConfig.load(Path(__file__).resolve().parents[1] / "config/study.yaml")
    m = marks_from_config(cfg)
    assert m["data_scientist"]["echo_frames"] == 17 and m["cardiologist_pair_kappa"] == 0.80
    assert cfg.training["reference_size"]["records"] == 40
