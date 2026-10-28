"""Automatic echo annotation: view label (PLAX / PSAX / A4C / A2C / OTHER) and cardiac phase
(end-diastole and end-systole frame) for every clip in the echo manifest.

HCMAN needs a `view` for each clip, and uses `ed_frame` / `es_frame` when present. Since this is
the first training and there are no existing labels, annotation is bootstrapped in a loop:

  1. Rules   — DICOM metadata (view name, series description, protocol). Colour-Doppler and
               single-frame images become OTHER.                    -> high confidence where found
  2. Review  — clips without a confident label go to the review queue (annotate.review_ui);
               reviewers label a seed set, about 100 clips per view.
  3. Model   — a small CNN trained on reviewed + rule labels (train-views), then `auto` runs again
               and only low-confidence clips (plus a random QA sample) go back to review.
  4. Phase   — ED from the ECG R-wave times stored in the DICOM when present; otherwise, and for
               ES always, from the cavity signal (fraction of dark blood-pool pixels per frame).
  5. Finalize — writes the annotated manifest the pipeline reads (view, ed_frame, es_frame).

    python -m annotate.echo auto      --config config/study.yaml --store $WB/annot/echo.csv \
                                      --cache $WB/annot/cache [--view-model $WB/annot/view_model.pt]
    python -m annotate.echo train-views --store $WB/annot/echo.csv --cache $WB/annot/cache \
                                      --out $WB/annot/view_model.pt
    python -m annotate.echo finalize  --config config/study.yaml --store $WB/annot/echo.csv \
                                      --out $WB/annot/echo_manifest_annotated.parquet
    python -m annotate.echo qa-report --config config/study.yaml --store $WB/annot/echo.csv
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from .store import AnnotationStore

log = logging.getLogger("annotate.echo")

VIEWS = ["PLAX", "PSAX", "A4C", "A2C", "OTHER"]
T_VIEW, T_ED, T_ES = "echo_view", "echo_ed", "echo_es"
T_RULE = "echo_view_rule"   # raw metadata-rule result, kept separately so re-runs can re-combine
THUMB = 112          # cached frame size for the view model and the review tool
CACHE_FRAMES = 16    # frames cached per clip

# Ordered: OTHER patterns are checked first so "apical long axis" does not match PLAX.
VIEW_RULES: list[tuple[str, list[str]]] = [
    ("OTHER", [r"\bA3C\b", r"\bAPLAX\b", r"APICAL\s*(3|THREE|LONG)", r"\bA5C\b", r"5\s*CH",
               r"SUBCOSTAL", r"SUBXIPH", r"\bSC\b", r"\bIVC\b", r"SUPRASTERNAL", r"\bSSN\b",
               r"DOPPLER", r"\bCW\b", r"\bPW\b", r"M[\s-]?MODE", r"COLOU?R", r"\bTDI\b"]),
    ("PLAX", [r"\bPLAX\b", r"PARASTERNAL\s*LONG", r"\bPSLAX\b"]),
    ("PSAX", [r"\bPSAX\b", r"PARASTERNAL\s*SHORT", r"\bSAX\b"]),
    ("A4C", [r"\bA4C\b", r"\bAP4\b", r"APICAL\s*(4|FOUR)", r"\b4\s*-?\s*CH", r"FOUR\s*CHAMBER"]),
    ("A2C", [r"\bA2C\b", r"\bAP2\b", r"APICAL\s*(2|TWO)", r"\b2\s*-?\s*CH", r"TWO\s*CHAMBER"]),
]


# ------------------------------------------------------------------ reading
def read_clip(path: str) -> tuple[np.ndarray, dict]:
    """-> (frames (T, H, W) float32 in [0, 1], metadata dict). Accepts DICOM or .npy (T, H, W)."""
    if str(path).endswith(".npy"):
        fr = np.load(path).astype(np.float32)
        meta = {}
    else:
        import pydicom
        ds = pydicom.dcmread(path)
        fr = ds.pixel_array.astype(np.float32)
        nframes = int(getattr(ds, "NumberOfFrames", 1) or 1)
        if nframes == 1:
            fr = fr[None]
        if fr.ndim == 4:                                   # (T, H, W, C) colour -> luminance
            fr = fr[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)
        texts = [str(getattr(ds, k, "") or "") for k in
                 ("ViewName", "SeriesDescription", "ProtocolName", "ImageComments")]
        for item in getattr(ds, "ViewCodeSequence", []) or []:
            texts.append(str(getattr(item, "CodeMeaning", "")))
        ftv = getattr(ds, "FrameTimeVector", None)
        frame_time = float(getattr(ds, "FrameTime", 0) or 0) or (
            float(np.mean([float(x) for x in ftv[1:]])) if ftv is not None and len(ftv) > 1 else 0.0)
        meta = {"text": " | ".join(t for t in texts if t),
                "color": int(getattr(ds, "UltrasoundColorDataPresent", 0) or 0),
                "frame_time_ms": frame_time,
                "r_wave_ms": [float(x) for x in (getattr(ds, "RWaveTimeVector", None) or [])],
                "heart_rate": float(getattr(ds, "HeartRate", 0) or 0)}
    if fr.ndim == 2:
        fr = fr[None]
    hi = float(fr.max()) or 1.0
    return fr / hi, meta


def thumbnails(frames: np.ndarray, n: int = CACHE_FRAMES, size: int = THUMB) -> np.ndarray:
    """(T, H, W) -> (n, size, size) uint8, frames sampled uniformly, aspect kept, zero-padded."""
    from PIL import Image
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    out = np.zeros((n, size, size), np.uint8)
    for k, i in enumerate(idx):
        im = Image.fromarray((frames[i] * 255).astype(np.uint8))
        im.thumbnail((size, size))
        a = np.asarray(im)
        y0, x0 = (size - a.shape[0]) // 2, (size - a.shape[1]) // 2
        out[k, y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = a
    return out


def item_id(path: str) -> str:
    return hashlib.sha1(str(path).encode()).hexdigest()[:16]


def cache_path(cache_dir: str | Path, iid: str) -> Path:
    return Path(cache_dir) / f"{iid}.npz"


# ------------------------------------------------------------------ view from metadata
def view_from_metadata(meta: dict, n_frames: int) -> tuple[str | None, float]:
    if n_frames < 2:
        return "OTHER", 0.99
    if meta.get("color"):
        return "OTHER", 0.99
    text = (meta.get("text") or "").upper()
    if not text:
        return None, 0.0
    hits = [v for v, pats in VIEW_RULES if any(re.search(p, text) for p in pats)]
    if not hits:
        return None, 0.0
    if hits[0] == "OTHER":
        return "OTHER", 0.95
    return (hits[0], 0.97) if len(set(hits)) == 1 else (None, 0.0)


# ------------------------------------------------------------------ cardiac phase
def cavity_signal(frames: np.ndarray) -> np.ndarray:
    """Fraction of dark (blood-pool) pixels in the central sector per frame. Ventricular cavity
    area is largest at end-diastole and smallest at end-systole, so this tracks the cycle."""
    T, H, W = frames.shape
    yy, xx = np.mgrid[0:H, 0:W]
    mask = ((yy - H * 0.55) / (H * 0.35)) ** 2 + ((xx - W * 0.5) / (W * 0.3)) ** 2 <= 1
    centre = frames[:, mask]
    tissue = centre[centre > 0.02]
    thr = np.quantile(tissue, 0.35) if tissue.size else 0.1
    return ((centre > 0.005) & (centre < thr)).mean(axis=1)


def _period(sig: np.ndarray, lo: int, hi: int) -> int | None:
    x = sig - sig.mean()
    if np.allclose(x, 0):
        return None
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac = ac / ac[0]
    hi = min(hi, len(x) - 1)
    if hi <= lo:
        return None
    peaks, _ = signal.find_peaks(ac[lo:hi + 1])
    if not len(peaks):
        return None
    best = peaks[np.argmax(ac[lo:hi + 1][peaks])] + lo
    return int(best) if ac[best] > 0.2 else None


def detect_phases(frames: np.ndarray, meta: dict) -> dict:
    """-> {ed, es, confidence, method}. ED and ES are frame indices into the clip."""
    T = len(frames)
    if T < 8:
        return {"ed": None, "es": None, "confidence": 0.0, "method": "too_short"}
    ft = meta.get("frame_time_ms") or 0.0
    # plausible cycle length in frames: 40-180 bpm when frame time is known
    lo, hi = (max(3, int(60000 / 180 / ft)), int(60000 / 40 / ft)) if ft else (4, T - 1)
    raw = cavity_signal(frames)
    sm = signal.savgol_filter(raw, 5 if T >= 7 else 3, 2) if T >= 5 else raw
    period = _period(sm, lo, hi) or None

    ed_ecg = None
    if ft and meta.get("r_wave_ms"):
        cand = [int(round(r / ft)) for r in meta["r_wave_ms"] if 0 <= r / ft < T - 2]
        ed_ecg = cand[0] if cand else None

    span = period or T
    ed_img = int(np.argmax(sm[:span]))
    ed = ed_ecg if ed_ecg is not None else ed_img
    win_end = min(T, ed + max(3, int(0.6 * span)))
    es = int(ed + 1 + np.argmin(sm[ed + 1:win_end])) if win_end > ed + 1 else None

    amp = float(sm.max() - sm.min())
    noise = float(np.std(raw - sm)) + 1e-6
    conf = amp / (amp + 6 * noise) if amp > 0 else 0.0
    if period is None:
        conf *= 0.5
    method = "image"
    if ed_ecg is not None:
        method = "ecg+image"
        conf = max(conf, 0.95) if abs(ed_ecg - ed_img) <= 2 or period is None else min(conf, 0.5)
    return {"ed": ed, "es": es, "confidence": round(float(np.clip(conf, 0, 1)), 3), "method": method}


# ------------------------------------------------------------------ view model
def _view_net():
    import torch.nn as nn

    class ViewNet(nn.Module):
        """Small per-frame CNN; logits are averaged over the sampled frames of a clip."""
        def __init__(self, n_classes: int = len(VIEWS)):
            super().__init__()
            def blk(i, o):
                return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                     nn.MaxPool2d(2))
            self.f = nn.Sequential(blk(1, 16), blk(16, 32), blk(32, 64), blk(64, 64),
                                   nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                                   nn.Linear(64, n_classes))

        def forward(self, x):                    # x: (B, F, H, W) uint8-scaled floats
            B, Fr, H, W = x.shape
            return self.f(x.reshape(B * Fr, 1, H, W)).reshape(B, Fr, -1).mean(1)
    return ViewNet()


def _frames_for_model(thumbs: np.ndarray, k: int = 4, rng=None) -> np.ndarray:
    n = len(thumbs)
    idx = (np.sort(rng.choice(n, k, replace=False)) if rng is not None
           else np.linspace(0, n - 1, k).round().astype(int))
    return thumbs[idx].astype(np.float32) / 255.0


def train_view_model(store: AnnotationStore, cache_dir: str | Path, out: str | Path,
                     epochs: int = 30, use_rule_labels: bool = True, seed: int = 0,
                     min_per_class: int = 5, min_steps: int = 400) -> dict:
    """Train the view CNN on human labels (and confident rule labels), hold out 20% of human
    labels for an accuracy check. Returns the evaluation summary."""
    import torch
    import torch.nn.functional as F
    lab = store.latest(T_VIEW)
    lab = lab[(lab.source == "human") & lab.value.isin(VIEWS)]
    if use_rule_labels:
        rules = store.latest(T_RULE)
        rules = rules[rules.value.isin(VIEWS) & (pd.to_numeric(rules.confidence) >= 0.95)
                      & ~rules.item_id.isin(lab.item_id)]
        lab = pd.concat([lab, rules], ignore_index=True)
    lab = lab[[cache_path(cache_dir, i).exists() for i in lab.item_id]]
    counts = lab.value.value_counts()
    if (counts.reindex(VIEWS[:4]).fillna(0) < min_per_class).any():
        raise ValueError(f"need >= {min_per_class} labelled clips per view, have {counts.to_dict()}")
    rng = np.random.default_rng(seed)
    human = lab[lab.source == "human"].item_id.tolist()
    rng.shuffle(human)
    held = set(human[: max(1, len(human) // 5)]) if len(human) >= 10 else set()
    tr, ev = lab[~lab.item_id.isin(held)], lab[lab.item_id.isin(held)]
    thumbs = {i: np.load(cache_path(cache_dir, i))["thumbs"] for i in lab.item_id}
    y_of = {v: k for k, v in enumerate(VIEWS)}

    torch.manual_seed(seed)
    net = _view_net()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    w = torch.tensor([1.0 / max(counts.get(v, 1), 1) for v in VIEWS], dtype=torch.float32)
    w = w / w.sum() * len(VIEWS)
    steps_per_epoch = max(1, -(-len(tr) // 32))
    epochs = max(epochs, -(-min_steps // steps_per_epoch))     # small seed sets still get enough updates
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * steps_per_epoch)
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(tr))
        for s in range(0, len(order), 32):
            rows = tr.iloc[order[s:s + 32]]
            x = np.stack([_frames_for_model(thumbs[i], rng=rng) for i in rows.item_id])
            x = x * rng.uniform(0.8, 1.2, (len(x), 1, 1, 1)) + rng.uniform(-0.1, 0.1, (len(x), 1, 1, 1))
            shift = rng.integers(-6, 7, 2)
            x = np.roll(x, tuple(shift), axis=(2, 3))
            y = torch.tensor([y_of[v] for v in rows.value])
            loss = F.cross_entropy(net(torch.tensor(x, dtype=torch.float32)), y, weight=w)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    summary = {"n_train": len(tr), "n_heldout_human": len(ev), "epochs": epochs, "train_counts": tr.value.value_counts().to_dict()}
    if len(ev):
        probs = predict_views(net, [thumbs[i] for i in ev.item_id])
        pred = [VIEWS[k] for k in probs.argmax(1)]
        summary["heldout_accuracy"] = float(np.mean(np.array(pred) == ev.value.values))
    torch.save({"state_dict": net.state_dict(), "views": VIEWS, "summary": summary}, out)
    return summary


def load_view_model(path: str | Path):
    import torch
    ck = torch.load(path, map_location="cpu")
    net = _view_net(); net.load_state_dict(ck["state_dict"]); net.eval()
    return net


def predict_views(net, thumbs_list: list[np.ndarray]) -> np.ndarray:
    import torch
    net.eval()
    with torch.no_grad():
        x = torch.tensor(np.stack([_frames_for_model(t) for t in thumbs_list]), dtype=torch.float32)
        return torch.softmax(net(x), -1).numpy()


# ------------------------------------------------------------------ automatic pass
def _process_clip(args) -> dict:
    path, cache_dir = args
    iid = item_id(path)
    try:
        frames, meta = read_clip(path)
    except Exception as e:                                  # unreadable clip -> review / exclude
        return {"item_id": iid, "path": path, "error": type(e).__name__}
    th = thumbnails(frames)
    np.savez_compressed(cache_path(cache_dir, iid), thumbs=th, n_frames=len(frames))
    view, vconf = view_from_metadata(meta, len(frames))
    ph = detect_phases(frames, meta) if view != "OTHER" else {"ed": None, "es": None, "confidence": 1.0, "method": "n/a"}
    return {"item_id": iid, "path": path, "n_frames": len(frames), "rule_view": view,
            "rule_conf": vconf, **{f"phase_{k}": v for k, v in ph.items()}}


def run_auto(manifest: pd.DataFrame, store: AnnotationStore, cache_dir: str | Path,
             view_model: str | Path | None = None, workers: int = 1) -> pd.DataFrame:
    """Annotate every clip in the manifest and append auto labels to the store.

    Re-runnable: clips already decoded (cached, with a rule result in the store) are not decoded
    again; only the view decision is re-made, e.g. after a new view model is trained."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    paths = manifest.path.astype(str).unique().tolist()
    prev_rules = store.latest(T_RULE).set_index("item_id") if len(store.read()) else pd.DataFrame()
    todo = [p for p in paths if not (item_id(p) in prev_rules.index and cache_path(cache_dir, item_id(p)).exists())]
    jobs = [(p, str(cache_dir)) for p in todo]
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_process_clip, jobs, chunksize=16))
    else:
        res = [_process_clip(j) for j in jobs]
    log.info("decoded %d new clips, %d already cached", len(res), len(paths) - len(res))

    rows = []
    for r in res:
        if r.get("error"):
            rows += [{"item_id": r["item_id"], "task": t, "value": "UNREADABLE", "confidence": 1.0,
                      "source": "auto:read_error", "annotator": "auto", "note": r["error"]}
                     for t in (T_RULE, T_VIEW)]
            continue
        rows.append({"item_id": r["item_id"], "task": T_RULE, "value": r["rule_view"] or "",
                     "confidence": r["rule_conf"], "source": "auto:rules", "annotator": "auto"})
        if r["rule_view"] != "OTHER":
            for t, v in ((T_ED, r["phase_ed"]), (T_ES, r["phase_es"])):
                rows.append({"item_id": r["item_id"], "task": t, "value": "" if v is None else str(int(v)),
                             "confidence": r["phase_confidence"], "source": f"auto:{r['phase_method']}",
                             "annotator": "auto"})
    store.add(rows)

    rules = store.latest(T_RULE).set_index("item_id")
    ids = [item_id(p) for p in paths if item_id(p) in rules.index]
    ids = [i for i in ids if rules.at[i, "value"] != "UNREADABLE"]
    model_probs = {}
    if view_model and ids:
        net = load_view_model(view_model)
        for s in range(0, len(ids), 256):
            chunk = ids[s:s + 256]
            p = predict_views(net, [np.load(cache_path(cache_dir, i))["thumbs"] for i in chunk])
            model_probs.update(dict(zip(chunk, p)))

    out, view_rows = [], []
    for i in ids:
        rv = rules.at[i, "value"] if isinstance(rules.at[i, "value"], str) else ""
        view, conf, src = (rv or None), float(rules.at[i, "confidence"] or 0), "auto:rules"
        if i in model_probs:
            p = model_probs[i]
            mv, mc = VIEWS[int(p.argmax())], float(p.max())
            if view is None:
                view, conf, src = mv, mc, "auto:model"
            elif mv != view and view != "OTHER":
                conf, src = min(conf, 0.5), "auto:rules!=model"
        view_rows.append({"item_id": i, "task": T_VIEW, "value": view or "", "confidence": round(conf, 3),
                          "source": src, "annotator": "auto"})
        out.append({"item_id": i, "view": view, "confidence": conf, "source": src})
    store.add(view_rows)
    pd.DataFrame({"item_id": [item_id(p) for p in paths], "path": paths}).to_csv(
        Path(cache_dir) / "index.csv", index=False)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ queue, finalize, QA
def _qa_pick(iid: str, fraction: float, seed: int = 0) -> bool:
    h = int(hashlib.sha1(f"{seed}:{iid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < fraction


def review_queue(store: AnnotationStore, threshold: float = 0.9, qa_fraction: float = 0.05,
                 seed: int = 0) -> pd.DataFrame:
    """Clips needing a human look: low-confidence view or phase, plus a fixed random QA sample
    of confidently auto-labelled clips (to measure automatic accuracy). Lowest confidence first."""
    lat = store.latest()
    lat = lat[lat.task.isin([T_VIEW, T_ED, T_ES])]
    done = store.human_items(T_VIEW)
    auto = lat[~lat.source.eq("human") & ~lat.item_id.isin(done)]
    if auto.empty:
        return pd.DataFrame(columns=["item_id", "reason", "min_conf"])
    unreadable = set(auto[(auto.task == T_VIEW) & (auto.value == "UNREADABLE")].item_id)
    auto = auto[~auto.item_id.isin(unreadable)]
    conf = auto.assign(c=pd.to_numeric(auto.confidence, errors="coerce").fillna(0))
    conf.loc[(conf.task == T_VIEW) & (conf.value.fillna("") == ""), "c"] = 0.0
    per = conf.groupby("item_id").c.min().rename("min_conf").reset_index()
    per["reason"] = np.where(per.min_conf < threshold, "low confidence", "")
    qa = per.item_id.map(lambda i: _qa_pick(i, qa_fraction, seed)) & (per.reason == "")
    per.loc[qa, "reason"] = "QA sample"
    return per[per.reason != ""].sort_values("min_conf").reset_index(drop=True)


def finalize(manifest: pd.DataFrame, store: AnnotationStore, threshold: float = 0.9) -> pd.DataFrame:
    """Manifest + view, ed_frame, es_frame, label_source. A view below threshold that no human
    reviewed is left empty (the clip is excluded) rather than trained on as a guess."""
    lat = store.latest()
    piv = {}
    for t in (T_VIEW, T_ED, T_ES):
        sub = lat[lat.task == t].set_index("item_id")
        piv[t] = sub
    m = manifest.copy()
    m["_iid"] = m.path.astype(str).map(item_id)
    v = piv[T_VIEW]
    m["view"] = m._iid.map(v.value)
    m["view_conf"] = pd.to_numeric(m._iid.map(v.confidence), errors="coerce")
    m["label_source"] = m._iid.map(v.source)
    keep = (m.label_source == "human") | (m.view_conf >= threshold)
    m.loc[~keep | ~m.view.isin(VIEWS), "view"] = None
    for t, col in ((T_ED, "ed_frame"), (T_ES, "es_frame")):
        s = piv[t]
        val = pd.to_numeric(m._iid.map(s.value), errors="coerce")
        c = pd.to_numeric(m._iid.map(s.confidence), errors="coerce")
        ok = (m._iid.map(s.source) == "human") | (c >= threshold)
        m[col] = val.where(ok)
    return m.drop(columns=["_iid"])        # ED/ES are frame numbers in the original clip


def qa_report(store: AnnotationStore) -> pd.DataFrame:
    """Agreement between automatic and human labels on every clip a human reviewed."""
    df = store.read()
    df = df[df.task.isin([T_VIEW, T_ED, T_ES])]
    hum = df[df.source == "human"].sort_values("ts").groupby(["item_id", "task"]).tail(1)
    aut = df[df.source != "human"].sort_values("ts").groupby(["item_id", "task"]).tail(1)
    j = hum.merge(aut, on=["item_id", "task"], suffixes=("_h", "_a"))
    rows = []
    jv = j[j.task == T_VIEW]
    if len(jv):
        rows.append({"metric": "view agreement (auto vs human)", "n_clips": len(jv),
                     "value": round(float((jv.value_h == jv.value_a).mean()), 3)})
    for t in (T_ED, T_ES):
        jt = j[j.task == t]
        a, h = pd.to_numeric(jt.value_a, errors="coerce"), pd.to_numeric(jt.value_h, errors="coerce")
        ok = a.notna() & h.notna()
        if ok.any():
            err = (a[ok] - h[ok]).abs()
            rows.append({"metric": f"{t} frames within 2 of human", "n_clips": int(ok.sum()),
                         "value": round(float((err <= 2).mean()), 3)})
    return pd.DataFrame(rows, columns=["metric", "n_clips", "value"])


# ------------------------------------------------------------------ CLI
def main(argv=None):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mcp_adapter.config import StudyConfig
    from mcp_adapter.privacy import suppress, write_report

    ap = argparse.ArgumentParser(prog="python -m annotate.echo")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("auto", "train-views", "finalize", "qa-report"):
        p = sub.add_parser(name)
        p.add_argument("--store", required=True)
        if name != "train-views":
            p.add_argument("--config", required=True)
        if name in ("auto", "train-views"):
            p.add_argument("--cache", required=True)
        if name == "auto":
            p.add_argument("--view-model"); p.add_argument("--workers", type=int, default=1)
            p.add_argument("--manifest", help="override the echo manifest in the config")
        if name == "train-views":
            p.add_argument("--out", required=True); p.add_argument("--epochs", type=int, default=30)
        if name == "finalize":
            p.add_argument("--out", required=True); p.add_argument("--threshold", type=float, default=0.9)
            p.add_argument("--manifest", help="override the echo manifest in the config")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    store = AnnotationStore(a.store)

    def manifest(cfg):
        from mcp_adapter.pipeline import _load_manifest
        src = cfg.echo_source
        if getattr(a, "manifest", None):
            src.manifest = a.manifest
        be = None
        if str(src.manifest).startswith("sql:"):
            from mcp_adapter.backend import make_backend
            be = make_backend(cfg)
        return _load_manifest(src, be)

    if a.cmd == "train-views":
        print(train_view_model(store, a.cache, a.out, epochs=a.epochs))
        return
    cfg = StudyConfig.load(a.config)
    if a.cmd == "auto":
        df = run_auto(manifest(cfg), store, a.cache, a.view_model, a.workers)
        q = review_queue(store)
        print(f"annotated {len(df)} clips; {len(q)} in the review queue")
    elif a.cmd == "finalize":
        m = finalize(manifest(cfg), store, a.threshold)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        m.to_parquet(a.out, index=False) if a.out.endswith(".parquet") else m.to_csv(a.out, index=False)
        counts = m.view.fillna("excluded").value_counts().rename_axis("view").reset_index(name="n_clips")
        write_report(suppress(counts, ["n_clips"], cfg.min_cell_size), "09a_echo_views", cfg.report_dir)
        print(counts.to_string(index=False))
    elif a.cmd == "qa-report":
        r = qa_report(store)
        write_report(suppress(r, ["n_clips"], cfg.min_cell_size), "09b_echo_annotation_qa", cfg.report_dir)
        print(r.to_string(index=False))


if __name__ == "__main__":
    main()
