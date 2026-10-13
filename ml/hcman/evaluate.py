"""Evaluation: discrimination, calibration, operating points, subgroup fairness, bootstrap CIs.

Sensitivity targets in the Model Details Form (e.g. SCAD >= 92%) are only meaningful with the
specificity paid for them, so thresholds are chosen on VALIDATION to hit the target sensitivity
and then applied unchanged to TEST, reporting both sensitivity and specificity.
Every count in the output passes through small-cell suppression before it can be exported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from mcp_adapter.concepts import CONDITION_ORDER

# Form targets: (metric, value). Sensitivity targets set the validation threshold.
TARGETS = {
    "SCAD": ("sensitivity", 0.92), "PREECLAMPSIA_CVD": ("sensitivity", 0.94),
    "PPCM": ("sensitivity", 0.89), "mace": ("auroc", 0.84), "postpartum": ("accuracy", 0.85),
    "atypical": ("sensitivity", 0.80), "fmd_trigger": ("sensitivity", 0.95),
}


@torch.no_grad()
def predict(model, loader, device="cpu", condition_head="multilabel") -> pd.DataFrame:
    model.eval()
    rows = []
    for b in loader:
        bb = {k: v.to(device) for k, v in b.items()}
        o = model(bb, grl_lambda=0.0)
        pc = torch.sigmoid(o["condition"]) if condition_head == "multilabel" else torch.softmax(o["condition"], -1)
        df = pd.DataFrame(pc.cpu().numpy(), columns=[f"p_{c}" for c in CONDITION_ORDER])
        df["person_id"] = b["person_id"].numpy()
        for c_i, c in enumerate(CONDITION_ORDER):
            df[f"y_{c}"] = b["y_condition"][:, c_i].numpy()
        for k, head in (("mace", "mace"), ("postpartum_cvd", "postpartum"), ("atypical", "atypical"),
                        ("fmd_trigger", "fmd_trigger")):
            df[f"p_{head}"] = torch.sigmoid(o[head]).cpu().numpy()
            df[f"y_{head}"] = b[f"y_{k}"].numpy(); df[f"m_{head}"] = b[f"m_{k}"].numpy()
        tier = torch.softmax(o["risk_tier"], -1).cpu().numpy()
        df["p_tier"] = list(tier); df["y_tier"] = b["y_risk_tier"].numpy(); df["m_tier"] = b["m_risk_tier"].numpy()
        df["fair_group"] = b["fair_group"].numpy(); df["site_id"] = b["site_id"].numpy()
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def threshold_for_sensitivity(y, p, target):
    pos = np.sort(p[y > 0.5])
    if len(pos) == 0:
        return 0.5
    k = int(np.floor((1 - target) * len(pos)))
    return float(pos[min(k, len(pos) - 1)])


def binary_metrics(y, p, thr) -> dict:
    y = y.astype(int); pred = (p >= thr).astype(int)
    tp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 1)).sum())
    tn, fp = int(((pred == 0) & (y == 0)).sum()), int(((pred == 1) & (y == 0)).sum())
    two = len(np.unique(y)) == 2
    return {"n": len(y), "n_pos": int(y.sum()), "threshold": thr,
            "auroc": roc_auc_score(y, p) if two else np.nan,
            "auprc": average_precision_score(y, p) if two else np.nan,
            "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "ppv": tp / max(tp + fp, 1), "accuracy": (tp + tn) / max(len(y), 1),
            "brier": brier_score_loss(y, p) if len(y) else np.nan, "ece": ece(y, p) if len(y) else np.nan}


def _tasks(df):
    for c in CONDITION_ORDER:
        yield c, df[f"y_{c}"].values, df[f"p_{c}"].values, np.ones(len(df), bool)
    for h in ("mace", "postpartum", "atypical", "fmd_trigger"):
        yield h, df[f"y_{h}"].values, df[f"p_{h}"].values, df[f"m_{h}"].values > 0


def evaluate(val: pd.DataFrame, test: pd.DataFrame, n_boot: int = 1000, seed: int = 0,
             min_cell: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (name, yv, pv, mv), (_, yt, pt, mt) in zip(_tasks(val), _tasks(test)):
        metric, target = TARGETS.get(name, ("auroc", None))
        thr = threshold_for_sensitivity(yv[mv], pv[mv], target) if metric == "sensitivity" else 0.5
        y, p = yt[mt], pt[mt]
        r = {"task": name, "target_metric": metric, "target": target, **binary_metrics(y, p, thr)}
        # bootstrap CI for AUROC and sensitivity
        aucs, sens = [], []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) == 2:
                aucs.append(roc_auc_score(y[i], p[i]))
                sens.append(((p[i] >= thr) & (y[i] == 1)).sum() / max((y[i] == 1).sum(), 1))
        if aucs:
            r["auroc_ci"] = f"{np.percentile(aucs, 2.5):.3f}-{np.percentile(aucs, 97.5):.3f}"
            r["sensitivity_ci"] = f"{np.percentile(sens, 2.5):.3f}-{np.percentile(sens, 97.5):.3f}"
        if target is not None and not np.isnan(r.get(metric, np.nan)):
            r["meets_target"] = bool(r[metric] >= target)
        rows.append(r)
    out = pd.DataFrame(rows)
    # metrics computed on fewer than min_cell positives are not reported
    small = out.n_pos < min_cell
    for c in ("auroc", "auprc", "sensitivity", "specificity", "ppv", "auroc_ci", "sensitivity_ci"):
        if c in out:
            out[c] = out[c].astype(object); out.loc[small, c] = "suppressed (<%d positives)" % min_cell
    out["n_pos"] = out.n_pos.astype(object); out.loc[small, "n_pos"] = f"<{min_cell}"
    return out


def fairness_table(val: pd.DataFrame, test: pd.DataFrame, min_cell: int = 11) -> pd.DataFrame:
    """PPCM FNR and MACE AUROC by Black vs non-Black women — the form's equity constraint."""
    thr = threshold_for_sensitivity(val.y_PPCM.values, val.p_PPCM.values, TARGETS["PPCM"][1])
    rows = []
    for g, name in ((1, "Black"), (0, "non-Black"), (-1, "race unknown")):
        d = test[test.fair_group == g]
        pos = d[d.y_PPCM > 0.5]
        fnr = float((pos.p_PPCM < thr).mean()) if len(pos) else np.nan
        dm = d[d.m_mace > 0]
        auc = roc_auc_score(dm.y_mace, dm.p_mace) if dm.y_mace.nunique() == 2 else np.nan
        rows.append({"group": name, "n": len(d), "ppcm_pos": len(pos), "ppcm_fnr": fnr, "mace_auroc": auc})
    out = pd.DataFrame(rows).astype(object)
    for i, r in out.iterrows():
        if r.ppcm_pos < min_cell:
            out.at[i, "ppcm_fnr"] = "suppressed"; out.at[i, "ppcm_pos"] = f"<{min_cell}"
        if r.n < min_cell:
            out.at[i, "n"] = f"<{min_cell}"; out.at[i, "mace_auroc"] = "suppressed"
    return out
