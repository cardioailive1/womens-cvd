"""Phase 1 pre-training from scratch, inside MCP (first-time training: no prior weights exist).

    python -m hcman.pretrain --config config/study.yaml --pool-dir /workbench/hcman_pool \
        --out /workbench/hcman_pretrain [--size full|small] [--stages ecg,echo,ehr,labs]

The pool comes from mcp_adapter.pipeline.build_pretraining_pool: women with cardiac evaluations,
EXCLUDING the cohort's validation and test women. No labels are used.

  ECG   SimCLR (NT-Xent) on 12-lead waveforms — shift, amplitude, noise, lead dropout, baseline wander
  Echo  SimCLR on single echo frames — random resized crop, brightness/contrast, noise
  EHR   masked-token modelling, 15% of tokens (as in the form)
  Labs  masked-step prediction — 15% of lab events have their value hidden and predicted

Outputs `encoders.pt` (keys prefixed ecg./echo./ehr./labs. to load into HCMAN with
`--init-weights`) and `preprocessors.pkl` (EHR vocabulary, lab bins and z-scores fit on the
pool), which fine-tuning must reuse via `--preprocessors` so token ids line up.
"""
from __future__ import annotations

import argparse
import logging
import math
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mcp_adapter.config import StudyConfig
from mcp_adapter.features.labs_ehr import EHRTokenizer, LabEncoder
from mcp_adapter.features.waveforms import load_echo, load_ecg
from mcp_adapter.privacy import write_report

from .data import Preprocessors
from .model import CardioTransformer, ECGTransformer, EchoEncoder, HCMANConfig, TemporalTransformer

log = logging.getLogger("hcman.pretrain")


# ------------------------------------------------------------------ datasets
class ECGSet(Dataset):
    def __init__(self, records: pd.DataFrame, kind: str):
        self.paths, self.kind = records.path.tolist(), kind
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        try:
            return torch.from_numpy(load_ecg(self.paths[i], self.kind)), True
        except Exception:
            return torch.zeros(12, 5000), False


class EchoFrameSet(Dataset):
    """One random frame per clip per epoch."""
    def __init__(self, records: pd.DataFrame, frames: int = 8):
        self.paths, self.frames = records.path.tolist(), frames
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        try:
            clip = load_echo(self.paths[i], self.frames)
            return torch.from_numpy(clip[np.random.randint(len(clip))]), True
        except Exception:
            return torch.zeros(3, 224, 224), False


class SeqSet(Dataset):
    def __init__(self, seqs: list[np.ndarray]): self.seqs = seqs
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): return self.seqs[i]


def pad_collate(items, width):
    L = max(1, max(len(x) for x in items))
    a = np.zeros((len(items), L, width), np.float32); m = np.zeros((len(items), L), bool)
    for i, x in enumerate(items):
        a[i, :len(x)], m[i, :len(x)] = x, True
    return torch.from_numpy(a), torch.from_numpy(m)


def ok_collate(items):
    x = torch.stack([a for a, _ in items]); ok = torch.tensor([b for _, b in items])
    return x[ok]


# ------------------------------------------------------------------ augmentations
def aug_ecg(x: torch.Tensor) -> torch.Tensor:
    B, L, T = x.shape
    shift = torch.randint(-250, 251, (B,))
    x = torch.stack([torch.roll(x[i], int(shift[i]), -1) for i in range(B)])
    x = x * torch.empty(B, L, 1, device=x.device).uniform_(0.8, 1.2)
    t = torch.arange(T, device=x.device) / 500.0
    f = torch.empty(B, 1, 1, device=x.device).uniform_(0.1, 0.5)
    x = x + 0.1 * torch.sin(2 * math.pi * f * t)
    x = x + 0.05 * torch.randn_like(x)
    keep = (torch.rand(B, L, 1, device=x.device) > 0.2).float()
    return x * keep


def aug_frame(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    out = []
    for i in range(B):
        s = float(np.random.uniform(0.6, 1.0)); h, w = int(H * s), int(W * s)
        y0, x0 = np.random.randint(0, H - h + 1), np.random.randint(0, W - w + 1)
        crop = x[i:i + 1, :, y0:y0 + h, x0:x0 + w]
        out.append(F.interpolate(crop, size=(H, W), mode="bilinear", align_corners=False))
    x = torch.cat(out)
    x = x * torch.empty(B, 1, 1, 1, device=x.device).uniform_(0.8, 1.2) \
        + torch.empty(B, 1, 1, 1, device=x.device).uniform_(-0.2, 0.2)
    return x + 0.05 * torch.randn_like(x)


def nt_xent(z1, z2, tau=0.1):
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.T / tau
    n = z1.shape[0]
    sim.fill_diagonal_(float("-inf"))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


def proj(d):  # SimCLR projection head (discarded after pre-training)
    return nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 128))


def _fit(params, loader, step_fn, epochs, lr, dev, name):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    total = max(1, epochs * len(loader)); s = 0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(k / total, 1))))
    hist = []
    for ep in range(epochs):
        losses = []
        for batch in loader:
            loss = step_fn(batch)
            if loss is None:
                continue
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); s += 1
            losses.append(float(loss.detach()))
        m = float(np.mean(losses)) if losses else float("nan")
        hist.append(m); log.info("%s epoch %d loss %.4f", name, ep, m)
    return hist


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--size", choices=["full", "small"])
    ap.add_argument("--stages", default="ecg,echo,ehr,labs"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, help="override every stage's epochs (smoke tests)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = StudyConfig.load(a.config); P = cfg.pretrain
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d = Path(a.pool_dir)
    t = {n: pd.read_pickle(d / f"{n}.pkl") for n in ("pool", "measurements", "ehr_tokens", "ecg_records", "echo_records")}
    ids = t["pool"].person_id.tolist()

    pre = Preprocessors(ehr=EHRTokenizer(lookback_days=cfg.lookback_days).fit(t["ehr_tokens"], t["measurements"], ids),
                        labs=LabEncoder(window_days=cfg.labs_window_days).fit(t["measurements"], ids), sites=[])
    pre.save(out / "preprocessors.pkl")
    size = a.size or cfg.model_size
    mc = (HCMANConfig.small if size == "small" else HCMANConfig)(ehr_vocab=pre.ehr.vocab_size)
    bs = P["batch_size"]; ep = lambda k: a.epochs if a.epochs is not None else P[k]
    kw = dict(num_workers=a.workers, drop_last=False)
    stages = set(a.stages.split(","))
    state, summary = {}, {"pool_women": len(ids), "model_size": size}

    if "ecg" in stages and len(t["ecg_records"]):
        enc, head = ECGTransformer(mc).to(dev), proj(mc.ecg_dim).to(dev)
        dl = DataLoader(ECGSet(t["ecg_records"], cfg.ecg_source.kind), batch_size=bs, shuffle=True, collate_fn=ok_collate, **kw)
        def step(x):
            if len(x) < 2: return None
            x = x.to(dev)
            return nt_xent(head(enc(aug_ecg(x))[:, 0]), head(enc(aug_ecg(x))[:, 0]))
        summary["ecg_loss"] = _fit(list(enc.parameters()) + list(head.parameters()), dl, step, ep("ecg_epochs"), 3e-4, dev, "ecg")
        state.update({f"ecg.{k}": v.cpu() for k, v in enc.state_dict().items()})

    if "echo" in stages and len(t["echo_records"]):
        enc = EchoEncoder(mc).to(dev); head = proj(mc.echo_dim).to(dev)
        frame = lambda x: enc.fc(enc.backbone(x))
        dl = DataLoader(EchoFrameSet(t["echo_records"], mc.echo_frames), batch_size=bs, shuffle=True, collate_fn=ok_collate, **kw)
        def step(x):
            if len(x) < 2: return None
            x = x.to(dev)
            return nt_xent(head(frame(aug_frame(x))), head(frame(aug_frame(x))))
        params = list(enc.backbone.parameters()) + list(enc.fc.parameters()) + list(head.parameters())
        summary["echo_loss"] = _fit(params, dl, step, ep("echo_epochs"), 3e-4, dev, "echo")
        state.update({f"echo.{k}": v.cpu() for k, v in enc.state_dict().items() if k.startswith(("backbone", "fc"))})

    if "ehr" in stages:
        enc = CardioTransformer(mc).to(dev)
        seqs = list(pre.ehr.transform(t["ehr_tokens"], t["measurements"], ids).values())
        dl = DataLoader(SeqSet(seqs), batch_size=bs, shuffle=True, collate_fn=partial(pad_collate, width=4), **kw)
        mask_id = pre.ehr.vocab["[MASK]"]
        def step(batch):
            x, m = (v.to(dev) for v in batch)
            cand = m.clone(); cand[:, 0] = False                       # never mask CLS
            sel = cand & (torch.rand_like(x[..., 0]) < 0.15)
            if not sel.any(): return None
            target = x[..., 0].long()[sel]
            xm = x.clone(); xm[..., 0] = torch.where(sel, torch.full_like(x[..., 0], mask_id), x[..., 0])
            xm[..., 2] = torch.where(sel, torch.zeros_like(x[..., 2]), x[..., 2])
            return F.cross_entropy(enc.mlm(enc(xm, m))[sel], target)
        summary["ehr_loss"] = _fit(list(enc.parameters()), dl, step, ep("ehr_epochs"), 1e-4, dev, "ehr")
        state.update({f"ehr.{k}": v.cpu() for k, v in enc.state_dict().items()})

    if "labs" in stages:
        enc = TemporalTransformer(mc).to(dev)
        seqs = [s for s in pre.labs.transform(t["measurements"], ids).values() if len(s) >= 2]
        if seqs:
            dl = DataLoader(SeqSet(seqs), batch_size=bs, shuffle=True, collate_fn=partial(pad_collate, width=3), **kw)
            def step(batch):
                x, m = (v.to(dev) for v in batch)
                sel = m & (torch.rand_like(x[..., 0]) < 0.15)
                if not sel.any(): return None
                target = x[..., 1][sel]
                xm = x.clone(); xm[..., 1] = torch.where(sel, torch.zeros_like(x[..., 1]), x[..., 1])
                tok, _ = enc(xm, m)
                return F.mse_loss(enc.head(tok).squeeze(-1)[sel], target)
            summary["labs_loss"] = _fit(list(enc.parameters()), dl, step, ep("labs_epochs"), 1e-4, dev, "labs")
            state.update({f"labs.{k}": v.cpu() for k, v in enc.state_dict().items()})

    torch.save(state, out / "encoders.pt")
    summary["stages_done"] = sorted({k.split(".")[0] for k in state})
    write_report(summary, "08_pretraining_summary", cfg.report_dir)
    log.info("saved %d tensors for stages %s", len(state), summary["stages_done"])
    return summary


if __name__ == "__main__":
    main()
