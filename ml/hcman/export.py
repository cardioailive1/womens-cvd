"""Export the trained model for the CardioAI app: ONNX (EHR integration), TorchScript (web
API), and an INT8 dynamic-quantized variant for point-of-care use.

Weights leave MCP only after the Model Details Form review. The export bundle must contain
only weights, the model config and the fitted preprocessors (vocabulary of codes seen by >= min_count women, lab bin edges,
ELM weights), and nothing row-level.

    python -m hcman.export --run /workbench/hcman_run1 --out /workbench/hcman_export
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from mcp_adapter.concepts import CONDITION_ORDER, LAB_ORDER
from mcp_adapter.features.repro import FEATURE_NAMES
from mcp_adapter.features.waveforms import ECHO_VIEWS, N_SAMPLES

from .model import HCMAN, HCMANConfig

INPUTS = ["ecg", "ecg_present", "echo", "echo_mask", "ehr", "ehr_mask", "labs", "labs_mask", "repro"]
OUTPUTS = ["p_condition", "p_mace", "p_postpartum", "p_risk_tier", "p_atypical", "p_fmd_trigger"]


class InferenceWrapper(nn.Module):
    """Positional tensors in, probabilities out — the contract the app's adapter consumes."""

    def __init__(self, model: HCMAN):
        super().__init__()
        self.m = model.set_export_mode(True).eval()
        self.multilabel = model.c.condition_head == "multilabel"

    def forward(self, ecg, ecg_present, echo, echo_mask, ehr, ehr_mask, labs, labs_mask, repro):
        o = self.m({"ecg": ecg, "ecg_present": ecg_present, "echo": echo, "echo_mask": echo_mask,
                    "ehr": ehr, "ehr_mask": ehr_mask, "labs": labs, "labs_mask": labs_mask,
                    "repro": repro}, grl_lambda=0.0)
        cond = torch.sigmoid(o["condition"]) if self.multilabel else torch.softmax(o["condition"], -1)
        return (cond, torch.sigmoid(o["mace"]), torch.sigmoid(o["postpartum"]),
                torch.softmax(o["risk_tier"], -1), torch.sigmoid(o["atypical"]),
                torch.sigmoid(o["fmd_trigger"]))


# Sequence inputs are exported at a FIXED padded length (the training maxima) and only the batch
# axis is dynamic: the legacy ONNX exporter bakes traced sequence lengths into attention
# reshapes. Padding is exact, not an approximation — padded positions are masked as keys
# everywhere, so outputs equal the unpadded model's (tests/test_pipeline.py checks this).
EHR_LEN, LABS_LEN = 512, 64


def pad_to(arr, mask, n):
    """Right-pad (B, L, W) and its (B, L) mask to length n (truncating keeps the latest)."""
    import numpy as np
    B, L, W = arr.shape
    if L >= n:
        return arr[:, -n:], mask[:, -n:]
    a = np.zeros((B, n, W), arr.dtype); m = np.zeros((B, n), bool)
    a[:, :L], m[:, :L] = arr, mask
    return a, m


def example_inputs(c: HCMANConfig, B: int = 1, L: int = EHR_LEN, E: int = LABS_LEN):
    N = len(ECHO_VIEWS) * c.echo_frames
    ehr = torch.zeros(B, L, 4); ehr[..., 0] = 1
    return (torch.zeros(B, 12, N_SAMPLES), torch.zeros(B, dtype=torch.bool),
            torch.zeros(B, N, 3, 224, 224), torch.zeros(B, N, dtype=torch.bool),
            ehr, torch.ones(B, L, dtype=torch.bool),
            torch.zeros(B, E, 3), torch.ones(B, E, dtype=torch.bool), torch.zeros(B, len(FEATURE_NAMES)))


def export(model: HCMAN, out: Path, opset: int = 17) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    w = InferenceWrapper(model)
    x = example_inputs(model.c)
    dyn = {"ecg": {0: "B"}, "ecg_present": {0: "B"}, "echo": {0: "B"}, "echo_mask": {0: "B"},
           "ehr": {0: "B"}, "ehr_mask": {0: "B"}, "labs": {0: "B"},
           "labs_mask": {0: "B"}, "repro": {0: "B"}, **{o: {0: "B"} for o in OUTPUTS}}
    torch.onnx.export(w, x, out / "hcman.onnx", input_names=INPUTS, output_names=OUTPUTS,
                      dynamic_axes=dyn, opset_version=opset, dynamo=False)
    ts = torch.jit.trace(w, x, check_trace=False, strict=False)
    ts.save(str(out / "hcman.torchscript.pt"))
    q = torch.ao.quantization.quantize_dynamic(InferenceWrapper(model), {nn.Linear}, dtype=torch.qint8)
    torch.save(q.state_dict(), out / "hcman_int8_state_dict.pt")
    contract = {"inputs": INPUTS, "outputs": OUTPUTS, "conditions": CONDITION_ORDER,
                "risk_tiers": ["LOW", "MODERATE", "HIGH", "CRITICAL"], "repro_features": FEATURE_NAMES,
                "labs": LAB_ORDER, "ehr_len": EHR_LEN, "labs_len": LABS_LEN, "echo_views": ECHO_VIEWS, "echo_frames_per_view": model.c.echo_frames,
                "ecg": {"leads": 12, "fs_hz": 500, "samples": N_SAMPLES},
                "condition_head": model.c.condition_head, "opset": opset}
    (out / "contract.json").write_text(json.dumps(contract, indent=2))
    return contract


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ck = torch.load(Path(a.run) / "hcman_best.pt", map_location="cpu")
    c = HCMANConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["model_config"].items()})
    m = HCMAN(c); m.load_state_dict(ck["state_dict"])
    out = Path(a.out)
    export(m, out)
    import shutil
    for f in ("preprocessors.pkl", "thresholds.json"):
        if (Path(a.run) / f).exists():
            shutil.copy(Path(a.run) / f, out / f)


if __name__ == "__main__":
    main()
