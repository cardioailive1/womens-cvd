"""HCMAN inference service for the CardioAI app (ONNX Runtime, standard library HTTP server).

    python -m hcman.serve --bundle /path/to/hcman_export --port 8080

The bundle is what hcman.export writes plus preprocessors.pkl and thresholds.json from the
training run. POST /predict takes the JSON built by the app's hcmanAdapter.ts and returns
HcmanPrediction. The app has no ECG waveforms, echo cine or ICD/ATC history yet, so those
streams are sent as missing; the response lists them in `missingModalities`.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

from mcp_adapter import concepts as C
from mcp_adapter.features.labs_ehr import harmonize_units
from mcp_adapter.features.repro import FEATURE_NAMES, MENO_STATUS, PREG_STATUS
from mcp_adapter.features.waveforms import ECHO_VIEWS, N_SAMPLES

from .data import Preprocessors
from .export import pad_to

LOINC_TO_ITEM = {code: item for item, spec in C.LABS.items() for code in spec["loinc"]}
LOINC_TO_ITEM.update({code: item for item, codes in C.VITALS.items() for code in codes})


def encode_repro(r: dict, age: float | None) -> np.ndarray:
    """Same encoding as mcp_adapter.features.repro.build_repro, from the app's raw values."""
    f = dict.fromkeys(FEATURE_NAMES, 0.0)
    status = r.get("pregnancy_status") or "not_pregnant"
    f[f"pregnancy_status={status if status in PREG_STATUS else 'not_pregnant'}"] = 1.0
    gw = r.get("gestational_age_weeks")
    f["gestational_age_weeks"] = (gw / 40.0) if gw is not None and status == "pregnant" else 0.0
    f["gestational_age_missing"] = 0.0 if gw is not None and status == "pregnant" else 1.0
    f["gravida_para"] = min(r.get("gravida_para") or 0, 6) / 6.0
    for k in ("preeclampsia_history", "gestational_diabetes", "hellp_history", "pcos_diagnosis",
              "endometriosis", "autoimmune_disease_sle_ra", "migraine_with_aura", "fmd_history",
              "breast_cancer_cardiotoxic_treatment", "emotional_stress_trigger_takotsubo"):
        f[k] = 1.0 if r.get(k) else 0.0
    meno = r.get("menopausal_status")
    inferred = meno not in MENO_STATUS
    if inferred:
        a = age or 0
        meno = "post" if a >= 52 else "peri" if a >= 45 else "pre"
    if status != "not_pregnant":
        meno = "pre"
    f[f"menopausal_status={meno}"] = 1.0
    f["menopausal_status_inferred"] = 1.0 if inferred else 0.0
    am = r.get("age_at_menopause")
    f["age_at_menopause"] = am / 60.0 if am is not None else 0.0
    f["age_at_menopause_missing"] = 0.0 if am is not None else 1.0
    return np.array([f[n] for n in FEATURE_NAMES], np.float32)


class Predictor:
    def __init__(self, bundle: str | Path):
        import onnxruntime as ort
        b = Path(bundle)
        self.sess = ort.InferenceSession(str(b / "hcman.onnx"), providers=["CPUExecutionProvider"])
        self.contract = json.loads((b / "contract.json").read_text())
        self.pre = Preprocessors.load(b / "preprocessors.pkl")
        t = b / "thresholds.json"
        self.thresholds = json.loads(t.read_text()) if t.exists() else {}
        self.version = self.contract.get("model_version", b.name)
        self.T = self.contract["echo_frames_per_view"]

    def _measurements(self, obs: list[dict]) -> pd.DataFrame:
        rows = [{"person_id": 0, "index_date": pd.Timestamp("2000-01-01"), "item": LOINC_TO_ITEM[o["loinc"]],
                 "event_date": pd.Timestamp("2000-01-01") - pd.Timedelta(days=int(o["daysBeforeIndex"])),
                 "value_as_number": float(o["value"]), "unit": o.get("unit")}
                for o in obs if o.get("loinc") in LOINC_TO_ITEM]
        cols = ["person_id", "index_date", "item", "event_date", "value_as_number", "unit"]
        return harmonize_units(pd.DataFrame(rows, columns=cols))

    def predict(self, req: dict) -> dict:
        meas = self._measurements(req.get("observations", []))
        empty_tokens = pd.DataFrame(columns=["person_id", "ttype", "code", "rel_day"])
        ehr = self.pre.ehr.transform(empty_tokens, meas, [0])[0]
        labs = self.pre.labs.transform(meas, [0])[0]
        if len(labs) == 0:
            labs_arr, labs_mask = np.zeros((1, 1, 3), np.float32), np.zeros((1, 1), bool)
        else:
            labs_arr, labs_mask = labs[None].astype(np.float32), np.ones((1, len(labs)), bool)
        labs_arr, labs_mask = pad_to(labs_arr, labs_mask, self.contract["labs_len"])
        ehr_arr, ehr_mask = pad_to(ehr[None].astype(np.float32), np.ones((1, len(ehr)), bool),
                                   self.contract["ehr_len"])
        N = len(ECHO_VIEWS) * self.T
        feeds = {
            "ecg": np.zeros((1, 12, N_SAMPLES), np.float32), "ecg_present": np.zeros(1, bool),
            "echo": np.zeros((1, N, 3, 224, 224), np.float32), "echo_mask": np.zeros((1, N), bool),
            "ehr": ehr_arr, "ehr_mask": ehr_mask,
            "labs": labs_arr, "labs_mask": labs_mask,
            "repro": encode_repro(req.get("repro", {}), req.get("age"))[None],
        }
        cond, mace, pp, tier, atyp, fmd = self.sess.run(None, feeds)
        postpartum = req.get("repro", {}).get("pregnancy_status") == "postpartum"
        return {
            "modelVersion": self.version,
            "conditions": {c: float(p) for c, p in zip(self.contract["conditions"], cond[0])},
            "mace12m": float(mace[0]), "postpartumCvd": float(pp[0]) if postpartum else None,
            "riskTier": [float(x) for x in tier[0]], "atypicalPresentation": float(atyp[0]),
            "fmdScreeningTrigger": float(fmd[0]), "thresholds": self.thresholds,
            "missingModalities": ["ECG waveform", "echocardiogram", "diagnosis/medication history"],
        }


def make_handler(pred: Predictor):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code); self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            self._send(200, {"status": "ok", "modelVersion": pred.version}) if self.path == "/health" \
                else self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/predict":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
                self._send(200, pred.predict(req))
            except Exception as e:                       # never echo request content back
                self._send(400, {"error": type(e).__name__})

        def log_message(self, *a):                       # no request logging: payload is PHI
            pass
    return H


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True); ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args(argv)
    ThreadingHTTPServer((a.host, a.port), make_handler(Predictor(a.bundle))).serve_forever()


if __name__ == "__main__":
    main()
