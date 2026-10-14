"""Tests for the annotation tools, on synthetic DICOM echo clips and the synthetic OMOP study."""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from annotate import echo as E
from annotate.store import AnnotationStore

ROOT = Path(__file__).resolve().parents[1]
PERIOD = 16           # frames per cardiac cycle in the synthetic clips


def make_clip(view: str, n: int = 40, size: int = 128, phase0: int = 0, seed: int = 0) -> tuple[np.ndarray, int]:
    """Synthetic cine with a dark 'cavity' whose area peaks every PERIOD frames.
    Returns (frames uint8 (n, size, size), first end-diastole frame)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    fr = np.zeros((n, size, size), np.uint8)
    shapes = {"PLAX": [(0.55, 0.5, 0.12, 0.30)], "PSAX": [(0.55, 0.5, 0.20, 0.20)],
              "A4C": [(0.55, 0.38, 0.25, 0.10), (0.55, 0.62, 0.25, 0.10)],
              "A2C": [(0.55, 0.5, 0.30, 0.10)], "OTHER": []}[view]
    for t in range(n):
        img = 150 + 40 * rng.standard_normal((size, size))
        scale = 0.75 + 0.25 * np.cos(2 * np.pi * (t - phase0) / PERIOD)    # max at ED
        for cy, cx, ry, rx in shapes:
            m = ((yy - cy * size) / (ry * size * scale)) ** 2 + ((xx - cx * size) / (rx * size * scale)) ** 2 <= 1
            img[m] = 20 + 5 * rng.standard_normal(m.sum())
        if view == "OTHER":
            img = 255 * rng.random((size, size))
        fr[t] = np.clip(img, 1, 255)
    return fr, phase0


def write_dicom(path: Path, frames: np.ndarray, desc: str = "", frame_time: float = 33.3,
                r_waves: list[float] | None = None, color: int = 0) -> None:
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.3.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset(); ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID; ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "US"; ds.SeriesDescription = desc
    ds.NumberOfFrames = len(frames); ds.Rows, ds.Columns = frames.shape[1:]
    ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8; ds.BitsStored = 8; ds.HighBit = 7; ds.PixelRepresentation = 0
    ds.FrameTime = frame_time; ds.UltrasoundColorDataPresent = color
    if r_waves:
        ds.RWaveTimeVector = r_waves
    ds.PixelData = frames.tobytes()
    ds.save_as(path, enforce_file_format=True)


# ------------------------------------------------------------------ unit
@pytest.mark.parametrize("text,expected", [
    ("A4C", "A4C"), ("Apical 4 chamber", "A4C"), ("AP2 LV", "A2C"), ("PLAX zoom", "PLAX"),
    ("PSAX MV level", "PSAX"), ("Apical long axis", "OTHER"), ("Subcostal IVC", "OTHER"),
    ("A4C / A2C", None), ("", None), ("misc", None)])
def test_view_rules(text, expected):
    assert E.view_from_metadata({"text": text}, 30)[0] == expected


def test_color_and_single_frame_are_other():
    assert E.view_from_metadata({"text": "A4C", "color": 1}, 30)[0] == "OTHER"
    assert E.view_from_metadata({"text": "A4C"}, 1)[0] == "OTHER"


@pytest.mark.parametrize("phase0", [0, 3, 7])
def test_phase_from_image(phase0):
    fr, ed = make_clip("A4C", phase0=phase0, seed=phase0)
    ph = E.detect_phases(fr.astype(np.float32) / 255, {"frame_time_ms": 33.3})
    assert abs(ph["ed"] - ed) <= 1
    assert abs(ph["es"] - (ed + PERIOD // 2)) <= 2                  # cavity smallest half a cycle later
    assert ph["confidence"] > 0.5


def test_phase_uses_ecg_r_wave():
    fr, ed = make_clip("A2C", phase0=5, seed=1)
    ph = E.detect_phases(fr.astype(np.float32) / 255, {"frame_time_ms": 33.3, "r_wave_ms": [5 * 33.3, 21 * 33.3]})
    assert ph["ed"] == 5 and ph["method"] == "ecg+image" and ph["confidence"] >= 0.95


def test_store_human_beats_auto(tmp_path):
    s = AnnotationStore(tmp_path / "a.csv")
    s.add([{"item_id": "x", "task": "t", "value": "auto", "confidence": 0.9, "source": "auto:rules"}])
    s.add_human("x", "t", "human", "rev")
    s.add([{"item_id": "x", "task": "t", "value": "auto2", "confidence": 0.99, "source": "auto:model"}])
    assert s.latest("t").value.tolist() == ["human"]
    assert len(s.read()) == 3                                          # nothing overwritten


# ------------------------------------------------------------------ end to end: echo loop
def test_echo_annotation_loop(tmp_path):
    torch = pytest.importorskip("torch")
    from annotate.review_ui import EchoReviewer
    rng = np.random.default_rng(0)
    rows, truth = [], {}
    names = {"PLAX": "PLAX", "PSAX": "PSAX MV", "A4C": "Apical 4CH", "A2C": "A2C"}
    for k in range(80):
        v = ["PLAX", "PSAX", "A4C", "A2C", "OTHER"][k % 5]
        fr, ed = make_clip(v, phase0=int(rng.integers(0, 8)), seed=k)
        p = tmp_path / f"clip{k}.dcm"
        labelled = k < 40 and v != "OTHER"                              # half carry a view name
        write_dicom(p, fr, desc=names[v] if labelled else "", color=int(v == "OTHER" and k % 2 == 0))
        rows.append({"person_id": k, "record_date": "2020-01-01", "path": str(p)})
        truth[E.item_id(str(p))] = (v, ed)
    man = pd.DataFrame(rows)
    store = AnnotationStore(tmp_path / "annot" / "echo.csv")
    cache = tmp_path / "annot" / "cache"

    # pass 1: rules only -> the unlabelled clips land in the review queue
    E.run_auto(man, store, cache, workers=2)
    q = E.review_queue(store, qa_fraction=0.0)
    assert len(q) >= 30
    rule_view = store.latest(E.T_VIEW).set_index("item_id").value
    named = [i for i, (v, _) in truth.items() if rule_view.get(i) in E.VIEWS]
    assert all(rule_view[i] == truth[i][0] for i in named)               # rules never mislabel

    # reviewers label part of the queue (simulated with the true answer)
    rv = EchoReviewer(store, cache, annotator="tester", qa_fraction=0.0)
    for _ in range(20):
        c = rv.current()
        v, ed = truth[c["item_id"]]
        rv.submit(v, ed, (ed + PERIOD // 2) if v != "OTHER" else None)
    with pytest.raises(ValueError):
        rv.submit("NOT_A_VIEW")

    # pass 2: train the view model on reviewed + rule labels, re-run auto with it
    summary = E.train_view_model(store, cache, tmp_path / "annot" / "view_model.pt", epochs=25, min_per_class=3)
    assert summary["n_train"] >= 40
    out = E.run_auto(man, store, cache, view_model=tmp_path / "annot" / "view_model.pt")
    model_rows = out[out.source == "auto:model"]
    acc = np.mean([truth[i][0] == v for i, v in zip(model_rows.item_id, model_rows.view)])
    assert acc >= 0.8, acc

    # finalize -> manifest the pipeline reads
    fin = E.finalize(man, store)
    assert {"view", "ed_frame", "es_frame"} <= set(fin.columns)
    got = fin.dropna(subset=["view"])
    assert np.mean([truth[E.item_id(p)][0] == v for p, v in zip(got.path, got.view)]) >= 0.9
    ok = got[got.view != "OTHER"].dropna(subset=["ed_frame"])
    assert np.mean([abs(truth[E.item_id(p)][1] - e) <= 1 for p, e in zip(ok.path, ok.ed_frame)]) >= 0.9

    qa = E.qa_report(store)
    assert "view agreement (auto vs human)" in set(qa.metric)
    rv.show()                                                         # widget builds without error


# ------------------------------------------------------------------ end to end: phenotype review
def test_phenotype_review(tmp_path):
    from annotate.phenotype import fetch_context, sample_for_review, task_name, validation_report
    from annotate.review_ui import PhenotypeReviewer
    from mcp_adapter.backend import DuckDBBackend
    from mcp_adapter.config import StudyConfig
    from mcp_adapter.pipeline import build_study
    from tests.synthetic_omop import load_into_duckdb

    con = duckdb.connect(); load_into_duckdb(con)
    con.execute("CREATE TABLE omop_cdm.note AS SELECT person_id, condition_start_date AS note_date, "
                "'Cardiology consult' AS note_title, 'Spontaneous dissection of the LAD, no instrumentation.' AS note_text "
                "FROM omop_cdm.condition_occurrence LIMIT 200")
    be = DuckDBBackend(con)
    cfg = StudyConfig.load(ROOT / "config/study.yaml", dialect="duckdb", report_dir=str(tmp_path / "rep"),
                           imaging_required=False)
    t = build_study(be, cfg, tmp_path / "study")
    assert (tmp_path / "study" / "code_events.pkl").exists()

    s = sample_for_review(t["labels"], t["code_events"], ["SCAD", "TAKOTSUBO"], n_pos=15, n_neg=15)
    assert set(s.stratum) >= {"positive", "negative"}
    ctx = fetch_context(be, cfg, t["cohort"], s.person_id)
    assert len(ctx["diagnoses"]) and len(ctx["notes"])

    store = AnnotationStore(tmp_path / "pheno.csv")
    rv = PhenotypeReviewer(s, ctx, store, annotator="tester")
    first = rv.current()
    assert "algo_label" not in first and "stratum" not in first        # reviewer is blinded
    assert "<h3>" in rv.render(first)
    truth = s.set_index(["person_id", "condition"]).algo_label
    n = 0
    while (c := rv.current()) is not None:
        lab = truth[(c["person_id"], c["condition"])]
        rv.submit("yes" if (lab == 1) != (n % 10 == 0) else "no")      # reviewer disagrees 1 in 10
        n += 1
    rep = validation_report(s, store, min_cell=11)
    ppv = rep[(rep.condition == "SCAD") & (rep.metric == "PPV")].iloc[0]
    assert ppv.n_reviewed == 15 and 0.8 <= float(ppv.value) <= 1.0
    assert len(store.latest(task_name("SCAD"))) == 30
    PhenotypeReviewer(s, ctx, store, annotator="tester").show()        # queue now empty, still renders
