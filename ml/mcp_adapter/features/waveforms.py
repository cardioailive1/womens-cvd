"""Streams 1 and 2: 12-lead ECG waveforms and echocardiography cine frames.

Imaging is available on MCP under the Cardio AI contract. Loading goes through a small source
interface so the exact storage layout only touches config: a manifest (a platform table
"sql:schema.table" or a file) with person_id, record_date, path (and view for echo), plus a
reader for the file format. Supported: GE MUSE XML, DICOM 12-lead ECG waveforms, NumPy arrays,
and DICOM echo cine.

Preprocessing follows the Model Details Form exactly:
  ECG  — 0.5 Hz Butterworth high-pass, 60 Hz notch, per-lead z-score, 500 Hz x 10 s = 5,000
         samples, 50-sample patches -> 100 patches x 12 leads.
  Echo — 224 x 224 grayscale frames replicated to 3 channels, T frames sampled uniformly across
         one cine loop (ED / ES indices used when the manifest provides them).
"""
from __future__ import annotations

import base64
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS, SECONDS = 500, 10
N_SAMPLES = FS * SECONDS
ECHO_VIEWS = ["PLAX", "PSAX", "A4C", "A2C"]


# ------------------------------------------------------------------ ECG
def preprocess_ecg(x: np.ndarray, fs: int = FS) -> np.ndarray:
    """x: (12, n) in mV at `fs` Hz -> (12, 5000) float32, filtered and z-scored per lead."""
    x = np.asarray(x, dtype=np.float64)
    if fs != FS:
        x = signal.resample(x, int(round(x.shape[1] * FS / fs)), axis=1)
    x = x[:, :N_SAMPLES]
    if x.shape[1] < N_SAMPLES:
        x = np.pad(x, ((0, 0), (0, N_SAMPLES - x.shape[1])))
    b, a = signal.butter(2, 0.5, btype="highpass", fs=FS)
    x = signal.filtfilt(b, a, x, axis=1)
    bn, an = signal.iirnotch(60.0, Q=30.0, fs=FS)
    x = signal.filtfilt(bn, an, x, axis=1)
    mu, sd = x.mean(axis=1, keepdims=True), x.std(axis=1, keepdims=True)
    return ((x - mu) / np.where(sd < 1e-6, 1.0, sd)).astype(np.float32)


def read_muse_xml(path: str | Path) -> tuple[np.ndarray, int]:
    """GE MUSE XML 'Rhythm' waveform -> (12, n) mV. MUSE stores I, II, V1–V6; the four limb
    leads III, aVR, aVL, aVF are derived (Einthoven / Goldberger)."""
    root = ET.parse(path).getroot()
    for wf in root.iter("Waveform"):
        if (wf.findtext("WaveformType") or "").strip() != "Rhythm":
            continue
        fs = int(wf.findtext("SampleBase") or FS)
        leads: dict[str, np.ndarray] = {}
        for ld in wf.iter("LeadData"):
            lid = (ld.findtext("LeadID") or "").strip()
            raw = base64.b64decode(ld.findtext("WaveFormData") or "")
            uv_per_bit = float(ld.findtext("LeadAmplitudeUnitsPerBit") or 4.88)
            leads[lid] = np.frombuffer(raw, dtype="<i2").astype(np.float64) * uv_per_bit / 1000.0
        I, II = leads["I"], leads["II"]
        leads.setdefault("III", II - I)
        leads.setdefault("aVR", -(I + II) / 2)
        leads.setdefault("aVL", I - II / 2)
        leads.setdefault("aVF", II - I / 2)
        n = min(len(v) for v in leads.values())
        return np.stack([leads[l][:n] for l in LEADS]), fs
    raise ValueError(f"no Rhythm waveform in {path}")


def read_dicom_ecg(path: str | Path) -> tuple[np.ndarray, int]:
    """DICOM 12-lead ECG waveform (WaveformSequence) -> (12, n) mV, leads reordered by label."""
    import pydicom
    ds = pydicom.dcmread(path)
    wf = ds.WaveformSequence[0]
    fs = int(round(float(wf.SamplingFrequency)))
    arr = ds.waveform_array(0).T.astype(np.float64)              # (channels, n), physical units
    names = []
    for ch in wf.ChannelDefinitionSequence:
        label = str(ch.ChannelSourceSequence[0].CodeMeaning)
        names.append(label.replace("Lead", "").strip())
        unit = str(getattr(ch.ChannelSensitivityUnitsSequence[0], "CodeValue", "mV")) \
            if "ChannelSensitivityUnitsSequence" in ch else "mV"
        if unit.lower() in ("uv", "µv"):
            arr[len(names) - 1] /= 1000.0
    idx = {n.upper(): i for i, n in enumerate(names)}
    return np.stack([arr[idx[l.upper()]] for l in LEADS]), fs


def load_ecg(path: str, kind: str) -> np.ndarray:
    if kind == "npy_dir":
        arr = np.load(path)
        return preprocess_ecg(arr if arr.shape[0] == 12 else arr.T)
    if kind == "muse_xml_dir":
        x, fs = read_muse_xml(path)
        return preprocess_ecg(x, fs)
    if kind == "dicom_ecg":
        x, fs = read_dicom_ecg(path)
        return preprocess_ecg(x, fs)
    raise ValueError(f"unsupported ECG source kind {kind}")


# ------------------------------------------------------------------ Echo
def load_echo(path: str, n_frames: int = 16, ed: int | None = None, es: int | None = None
              ) -> np.ndarray:
    """DICOM cine (or .npy (T, H, W)) -> (n_frames, 3, 224, 224) float32 in [0, 1]-normalized
    ImageNet stats. If ED/ES frame indices are known, the sample spans ED -> ES -> next ED."""
    if path.endswith(".npy"):
        frames = np.load(path)
    else:
        import pydicom
        frames = pydicom.dcmread(path).pixel_array
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim == 4:                      # (T, H, W, C) colour -> luminance
        frames = frames[..., :3].mean(-1)
    if frames.ndim == 2:
        frames = frames[None]
    T = frames.shape[0]
    if ed is not None and es is not None:
        idx = np.linspace(min(ed, es), max(ed, es), n_frames // 2).round().astype(int)
        idx = np.concatenate([idx, idx[::-1]])
    else:
        idx = np.linspace(0, T - 1, n_frames).round().astype(int)
    frames = frames[np.clip(idx, 0, T - 1)]
    out = np.stack([_resize(f, 224) for f in frames])
    out = out / max(float(out.max()), 1.0)
    out = (out - 0.449) / 0.226               # single-channel ImageNet stats
    return np.repeat(out[:, None], 3, axis=1).astype(np.float32)


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape
    s = size / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    r = signal.resample(signal.resample(img, nh, axis=0), nw, axis=1)
    out = np.zeros((size, size), np.float32)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r
    return out


# ------------------------------------------------------------------ manifest selection
def nearest_record(manifest: pd.DataFrame, cohort: pd.DataFrame, window: tuple[int, int],
                   by: tuple[str, ...] = ("person_id",)) -> pd.DataFrame:
    """Pick the record closest to index within `window` days (before index preferred) per
    group in `by` — ("person_id",) for ECG, ("person_id", "view") for echo (one clip per view).
    Manifest columns: person_id, path, record_date (and optionally view, ed_frame, es_frame)."""
    m = manifest.merge(cohort[["person_id", "index_date"]], on="person_id")
    m["rel_day"] = (pd.to_datetime(m.record_date) - pd.to_datetime(m.index_date)).dt.days
    m = m[m.rel_day.between(*window)]
    m["rank"] = m.rel_day.abs() + (m.rel_day > 0) * 0.5
    return m.sort_values([*by, "rank"]).groupby(list(by)).head(1)
