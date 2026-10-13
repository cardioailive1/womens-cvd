"""Training losses: masked multi-task loss, focal loss, adversarial terms, FNR-parity penalty.

total = task_loss + lambda_site * site_adv_loss + lambda_fair * (fair_adv_loss + fnr_gap)
(the form's formula; the GRL inside the model turns the adversary CE terms into invariance).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mcp_adapter.concepts import CONDITION_ORDER

PPCM = CONDITION_ORDER.index("PPCM")


def focal_bce(logits, y, gamma=2.0, alpha=None, reduction="none"):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    pt = p * y + (1 - p) * (1 - y)
    loss = ce * (1 - pt) ** gamma
    if alpha is not None:
        loss = loss * (alpha * y + (1 - alpha) * (1 - y))
    return loss


def masked_mean(x, m):
    return (x * m).sum() / m.sum().clamp(min=1.0)


@dataclass
class LossWeights:
    condition: float = 1.0
    mace: float = 1.0
    postpartum: float = 0.5
    risk_tier: float = 0.5
    atypical: float = 0.5
    fmd_trigger: float = 0.5
    site_adv: float = 0.1          # lambda_site_adv
    fair: float = 0.3              # lambda_fair (form: 0.3)
    gamma: float = 2.0             # focal gamma (form: 2)


def fnr_gap(prob, y, group):
    """Soft FNR difference for PPCM between Black (1) and non-Black (0) women in the batch."""
    pos = y > 0.5
    fnr = []
    for g in (0, 1):
        sel = pos & (group == g)
        if sel.sum() == 0:
            return prob.new_zeros(())
        fnr.append((1 - prob[sel]).mean())
    return (fnr[0] - fnr[1]).abs()


def hcman_loss(out, b, w: LossWeights, condition_head: str = "multilabel"):
    parts = {}
    yc = b["y_condition"]
    if condition_head == "multilabel":
        parts["condition"] = focal_bce(out["condition"], yc, w.gamma).mean()
    else:  # softmax as in the form: only women with exactly one condition contribute
        one = yc.sum(1) == 1
        lp = F.log_softmax(out["condition"], -1)
        ce = -(lp * yc).sum(1)
        pt = torch.exp(-ce)
        parts["condition"] = masked_mean(ce * (1 - pt) ** w.gamma, one.float())

    for key, head in (("mace", "mace"), ("postpartum_cvd", "postpartum"),
                      ("atypical", "atypical"), ("fmd_trigger", "fmd_trigger")):
        parts[head] = masked_mean(focal_bce(out[head], b[f"y_{key}"], w.gamma), b[f"m_{key}"])
    tier_ce = F.cross_entropy(out["risk_tier"], b["y_risk_tier"].long(), reduction="none")
    parts["risk_tier"] = masked_mean(tier_ce, b["m_risk_tier"])

    task = sum(getattr(w, k) * v for k, v in parts.items())

    site_ok = (b["site_id"] >= 0).float()
    site = masked_mean(F.cross_entropy(out["site_logits"], b["site_id"].clamp(min=0), reduction="none"), site_ok)
    fair_ok = (b["fair_group"] >= 0).float()
    fair_adv = masked_mean(F.cross_entropy(out["fair_logits"], b["fair_group"].clamp(min=0), reduction="none"), fair_ok)
    if condition_head == "multilabel":
        p_ppcm = torch.sigmoid(out["condition"][:, PPCM])
    else:
        p_ppcm = torch.softmax(out["condition"], -1)[:, PPCM]
    gap = fnr_gap(p_ppcm, yc[:, PPCM], b["fair_group"])

    total = task + w.site_adv * site + w.fair * (fair_adv + gap)
    parts.update(site_adv=site, fair_adv=fair_adv, fnr_gap=gap, task=task, total=total)
    return total, {k: float(v.detach()) for k, v in parts.items()}
