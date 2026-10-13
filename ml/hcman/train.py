"""Phase 2 fine-tuning of HCMAN inside the MCP workbench.

    python -m hcman.train --config config/study.yaml --study-dir /workbench/hcman_study \
        --out /workbench/hcman_run1 \
        --init-weights /workbench/hcman_pretrain/encoders.pt \
        --preprocessors /workbench/hcman_pretrain/preprocessors.pkl [--size small]

First-time training: run hcman.pretrain first (from scratch, inside MCP) and pass both of its
outputs here. Without --init-weights every encoder starts from random weights.

Optimizer and schedule follow the Model Details Form: AdamW with lr 1e-4 (encoders) /
3e-4 (fusion) / 5e-4 (heads), weight decay 0.01, 500-step warmup + cosine, batch 32, mixed
precision on GPU, gradient clipping 1.0, early stopping with patience 15. Early stopping uses
validation macro-AUROC over the six conditions instead of accuracy: with prevalences of 2–17%,
accuracy rewards predicting "no condition".
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from mcp_adapter.concepts import CONDITION_ORDER
from mcp_adapter.config import StudyConfig
from mcp_adapter.privacy import write_report

from .data import COND_Y, StudyData, collate
from .elm import MultiLayerELM
from .evaluate import TARGETS, evaluate, fairness_table, predict, threshold_for_sensitivity
from .losses import LossWeights, hcman_loss
from .model import HCMAN, HCMANConfig

log = logging.getLogger("hcman.train")


def balanced_sampler(rows) -> WeightedRandomSampler:
    """Up-weight women with rare conditions (inverse prevalence), the form's class-imbalance aim
    without SMOTE (which cannot synthesize a matching ECG / echo / EHR history)."""
    Y = rows[COND_Y].to_numpy()
    prev = Y.mean(0).clip(1e-3)
    w = 1.0 + (Y / prev).max(1) / (1 / prev).mean()
    return WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(w), replacement=True)


def macro_auroc(df) -> float:
    s = [roc_auc_score(df[f"y_{c}"], df[f"p_{c}"]) for c in CONDITION_ORDER if df[f"y_{c}"].nunique() == 2]
    return float(np.mean(s)) if s else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--study-dir", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--init-weights")
    ap.add_argument("--preprocessors", help="preprocessors.pkl from hcman.pretrain (required with --init-weights)")
    ap.add_argument("--size", choices=["full", "small"]); ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32); ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=500); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-imaging", action="store_true", help="skip ECG/echo loading (EHR-only ablation)")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = StudyConfig.load(a.config)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    if a.init_weights and not a.preprocessors:
        ap.error("--init-weights needs --preprocessors from the same pre-training run (token ids must match)")
    pre = None
    if a.preprocessors:
        from .data import Preprocessors
        pre = Preprocessors.load(a.preprocessors)
        co = pd.read_pickle(Path(a.study_dir) / "cohort.pkl")
        pre.sites = sorted(co.loc[co.split == "train", "site"].astype(str).unique())
    data = StudyData(a.study_dir, cfg, pre)
    size = a.size or cfg.model_size
    mk = HCMANConfig.small if size == "small" else HCMANConfig
    mc = mk(ehr_vocab=data.pre.ehr.vocab_size, n_sites=max(len(data.pre.sites), 1),
            condition_head=cfg.condition_head)
    elm = MultiLayerELM(mc.repro_in, mc.elm_hidden, seed=cfg.seed).fit(data.repro_matrix("train"))
    model = HCMAN(mc, elm).to(dev)
    if a.init_weights:      # encoders pre-trained from scratch inside MCP by hcman.pretrain
        enc = torch.load(a.init_weights, map_location=dev)
        missing, unexpected = model.load_state_dict(enc, strict=False)
        if unexpected:
            raise ValueError(f"init weights do not match this model size: {unexpected[:5]}")
        log.info("loaded %d pre-trained tensors (%s)", len(enc), sorted({k.split('.')[0] for k in enc}))
    else:
        log.warning("no --init-weights: all encoders start from random initialization")
    n_params = sum(p.numel() for p in model.parameters())
    log.info("HCMAN %s: %.1fM parameters", size, n_params / 1e6)

    T = mc.echo_frames
    tr = data.split("train", echo_frames=T, load_imaging=not a.no_imaging)
    va = data.split("val", echo_frames=T, load_imaging=not a.no_imaging)
    te = data.split("test", echo_frames=T, load_imaging=not a.no_imaging)
    kw = dict(batch_size=a.batch_size, collate_fn=collate, num_workers=a.workers)
    dl_tr = DataLoader(tr, sampler=balanced_sampler(tr.rows), drop_last=len(tr) > a.batch_size, **kw)
    dl_va, dl_te = DataLoader(va, **kw), DataLoader(te, **kw)

    opt = torch.optim.AdamW(model.param_groups())
    total_steps = a.epochs * max(len(dl_tr), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(s / total_steps, 1.0))))
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    w = LossWeights()
    best, bad, step, history = -1.0, 0, 0, []

    for epoch in range(a.epochs):
        model.set_imaging_frozen(epoch < cfg.freeze_imaging_backbones_epochs)
        model.train(); losses = []
        for b in dl_tr:
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            p = step / total_steps
            lam = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0          # DANN GRL schedule
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                o = model(b, grl_lambda=lam)
            loss, parts = hcman_loss({k: v.float() for k, v in o.items()}, b, w, mc.condition_head)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            losses.append(parts)
        pv = predict(model, dl_va, dev, mc.condition_head)
        score = macro_auroc(pv)
        mean = {k: float(np.mean([l[k] for l in losses])) for k in losses[0]} if losses else {}
        history.append({"epoch": epoch, "val_macro_auroc": score, **mean})
        log.info("epoch %d  loss %.4f  val macro-AUROC %.4f", epoch, mean.get("total", float("nan")), score)
        s = 0.0 if math.isnan(score) else score
        if s > best:
            best, bad = s, 0
            torch.save({"state_dict": model.state_dict(), "model_config": asdict(mc)}, out / "hcman_best.pt")
            data.pre.save(out / "preprocessors.pkl")
        else:
            bad += 1
            if bad >= a.patience:
                log.info("early stop at epoch %d", epoch); break

    ck = torch.load(out / "hcman_best.pt", map_location=dev)
    model.load_state_dict(ck["state_dict"])
    pv, pt = predict(model, dl_va, dev, mc.condition_head), predict(model, dl_te, dev, mc.condition_head)
    # operating points chosen on validation, shipped with the model for the app
    thr = {}
    for name, (metric, target) in TARGETS.items():
        col = name
        y, p = pv.get(f"y_{col}"), pv.get(f"p_{col}")
        if y is None:
            continue
        m = pv[f"m_{col}"] > 0 if f"m_{col}" in pv else slice(None)
        thr[name] = (threshold_for_sensitivity(y[m].values, p[m].values, target)
                     if metric == "sensitivity" else 0.5)
    (out / "thresholds.json").write_text(json.dumps(thr, indent=2))
    rep = Path(cfg.report_dir)
    write_report(evaluate(pv, pt, n_boot=a.n_boot, min_cell=cfg.min_cell_size), "10_test_metrics", rep)
    write_report(fairness_table(pv, pt, cfg.min_cell_size), "11_fairness", rep)
    write_report({"params_millions": round(n_params / 1e6, 2), "best_val_macro_auroc": best,
                  "epochs_run": len(history), "history": history}, "12_training_summary", rep)
    (out / "model_config.json").write_text(json.dumps(asdict(mc), indent=2))
    return best


if __name__ == "__main__":
    main()
