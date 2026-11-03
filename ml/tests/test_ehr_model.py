"""Tabular EHR model (stream 3) and the MobileNet1D ECG encoder (stream 1)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from hcman.ehr_model import BINARY_TARGETS, N_SCORES, EHRModel, feature_key, featurize
from mcp_adapter.concepts import CONDITION_ORDER


def _cohort(n=400, seed=0):
    rng = np.random.default_rng(seed)
    pid = np.arange(1, n + 1)
    htn = rng.random(n) < 0.4                      # I10 raises MACE risk; beta blocker marks it too
    rows = pd.DataFrame({"person_id": pid})
    for c in CONDITION_ORDER:
        rows[f"y_{c}"] = (rng.random(n) < 0.15).astype(float)
    rows["y_mace"] = ((htn & (rng.random(n) < 0.6)) | (rng.random(n) < 0.05)).astype(float)
    rows["m_mace"] = 1.0
    for k in ("postpartum_cvd", "atypical", "fmd_trigger"):
        rows[f"y_{k}"] = (rng.random(n) < 0.2).astype(float); rows[f"m_{k}"] = (rng.random(n) < 0.5).astype(float)
    rows["y_risk_tier"] = np.where(htn, 2, rng.integers(0, 2, n)).astype(float); rows["m_risk_tier"] = 1.0
    tok = []
    for p, h in zip(pid, htn):
        if h:
            tok += [(p, "DX", "I10", -30), (p, "RX", "C07AB02", -20)]
        tok += [(p, "DX", "I25.42", -10)]          # SCAD code: a label code, never a feature
        tok += [(p, "DX", "Z00.00", 5)]            # after index: never a feature
        if p % 3 == 0:
            tok += [(p, "DX", "E78.5", -100)]
    return rows, pd.DataFrame(tok, columns=["person_id", "ttype", "code", "rel_day"])


def test_feature_keys_and_leakage_guard():
    assert feature_key("DX", "I10") == "DX:I10" and feature_key("DX", "E78.5") == "DX:E78"
    assert feature_key("RX", "C07AB02") == "RX:C07AB" and feature_key("DX", "I25.42") is None
    rows, tok = _cohort(30)
    f = featurize(tok, rows.person_id, 365)
    assert all(k.startswith(("DX:", "RX:")) for s in f.values() for k in s)
    assert not any("I25" in k or "Z00" in k for s in f.values() for k in s)   # label code, after index


@pytest.mark.parametrize("kind", ["logistic", "xgboost"])
def test_ehr_model_learns_and_round_trips(kind):
    if kind == "xgboost":
        pytest.importorskip("xgboost")
    rows, tok = _cohort()
    f = featurize(tok, rows.person_id, 365)
    tr, te = rows.iloc[:300], rows.iloc[300:]
    m = EHRModel(kind=kind, min_count=11, seed=1).fit_vocab(f, tr.person_id)
    assert m.features == ["DX:E78", "DX:I10", "RX:C07AB"]
    X_tr, X_te = m.matrix(f, tr.person_id.tolist()), m.matrix(f, te.person_id.tolist())
    oof = m.oof_scores(X_tr, tr)
    m.fit(X_tr, tr)
    S = m.scores(X_te)
    assert oof.shape == (300, N_SCORES) and S.shape == (100, N_SCORES) and np.isfinite(S).all()
    j = BINARY_TARGETS.index("mace")
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(te.y_mace, S[:, j]) > 0.75 and roc_auc_score(tr.y_mace, oof[:, j]) > 0.75
    tier = np.exp(S[:, len(BINARY_TARGETS):])
    assert np.allclose(tier.sum(1), 1, atol=1e-4)
    m2 = EHRModel.from_json(m.to_json())                      # what goes in the export bundle
    assert np.allclose(m2.scores(X_te), S, atol=1e-5)
    d = json.loads(m.to_json())
    assert set(d) == {"kind", "features", "score_names", "min_count", "models"}   # numbers only, no rows


def test_rare_codes_are_not_features():
    rows, tok = _cohort(200)
    tok = pd.concat([tok, pd.DataFrame([(p, "DX", "Q99.9", -5) for p in range(1, 11)],
                                       columns=tok.columns)])           # 10 women: below 11
    m = EHRModel(min_count=11).fit_vocab(featurize(tok, rows.person_id, 365), rows.person_id)
    assert "DX:Q99" not in m.features


def test_mobilenet1d_ecg_encoder():
    from hcman.model import HCMAN, HCMANConfig, MobileNet1D
    c = HCMANConfig()
    enc = MobileNet1D(c)
    out = enc(torch.randn(2, 12, 5000))
    assert out.shape == (2, 1 + c.ecg_segments, c.ecg_dim)
    n_ecg = sum(p.numel() for p in enc.parameters()) / 1e6
    assert n_ecg < 2.0                                          # ~1.0M, vs ~85M for the transformer
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in enc.modules())
    n = sum(p.numel() for p in HCMAN(c).parameters()) / 1e6
    assert 45 < n < 60                                          # in line with the form's ~52M
