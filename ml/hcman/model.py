"""HCMAN — Hierarchical Cross-Modal Attention Network (PyTorch), per the Model Details Form.

Two deliberate changes from the form, both explained in the protocol:
1. Encoders hand TOKEN SEQUENCES (not single pooled vectors) to the fusion stages. Cross-
   attention from one vector onto one vector is a softmax over a single key — an identity
   map — so F1–F3 would not actually attend to anything.
2. Every stream can be missing. Each cross-attention appends a learned "null" key/value so a
   woman with no echo (say) attends to the null token instead of producing NaNs, and missing
   query streams are replaced by learned null queries.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from mcp_adapter.concepts import CONDITION_ORDER, LAB_ORDER
from mcp_adapter.features.repro import REPRO_DIM

from .elm import MultiLayerELM


@dataclass
class HCMANConfig:
    # stream 1 — ECG transformer (form: 12L-12H, d=768)
    ecg_dim: int = 768; ecg_layers: int = 12; ecg_heads: int = 12
    ecg_patch: int = 50; ecg_samples: int = 5000; ecg_leads: int = 12
    # stream 2 — echo (form: MobileNetV3-Large -> 960 -> FC 512)
    echo_dim: int = 512; echo_views: int = 4; echo_frames: int = 8
    echo_backbone: str = "mobilenet_v3_large"      # or "tiny" for tests
    # stream 3 — EHR BERT (form: 6L-8H, d=256)
    ehr_dim: int = 256; ehr_layers: int = 6; ehr_heads: int = 8; ehr_vocab: int = 20000
    # stream 4 — ELM (form: 15 -> 128 -> 256 -> 128)
    repro_in: int = REPRO_DIM; elm_hidden: tuple[int, ...] = (128, 256, 128); repro_tokens: int = 4
    # stream 5 — labs causal decoder (form: 4L-4H, d=128)
    labs_dim: int = 128; labs_layers: int = 4; labs_heads: int = 4
    # fusion (form: d=512)
    d: int = 512; f1_layers: int = 2; f2_layers: int = 4; f3_layers: int = 2
    fusion_heads: int = 8; f3_heads: int = 4; ffn: int = 2048; dropout: float = 0.1
    # heads / adversaries
    n_conditions: int = len(CONDITION_ORDER); condition_head: str = "multilabel"
    n_sites: int = 4; n_fair_groups: int = 2

    @classmethod
    def small(cls, **kw) -> "HCMANConfig":
        base = dict(ecg_dim=64, ecg_layers=2, ecg_heads=4, echo_dim=64, echo_backbone="tiny",
                    echo_frames=4, ehr_dim=64, ehr_layers=2, ehr_heads=4, labs_dim=32,
                    labs_layers=1, labs_heads=2, d=64, f1_layers=1, f2_layers=1, f3_layers=1,
                    fusion_heads=4, f3_heads=2, ffn=128, elm_hidden=(32, 64, 32))
        base.update(kw)
        return cls(**base)


# ============================================================ utilities
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x: torch.Tensor, lam: float) -> torch.Tensor:
    return GradReverse.apply(x, lam)


def time_encoding(days: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal encoding of (relative) days; works for negative values."""
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=days.device) / half)
    ang = days.unsqueeze(-1).float() * freq
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


def masked_mean(x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    w = keep.float().unsqueeze(-1)
    return (x * w).sum(1) / w.sum(1).clamp(min=1.0)


def encoder(d: int, layers: int, heads: int, ffn: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(d, heads, ffn, dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


# ============================================================ stream 1 — ECG
class CBAM1D(nn.Module):
    """CBAM adapted to patch tokens: channel attention over the embedding dimension (shared
    MLP on avg- and max-pooled tokens), then token ("spatial") attention from a 1-D conv over
    the per-token avg/max — lets the model up-weight ST-segment and P-wave patches."""

    def __init__(self, d: int, reduction: int = 16, k: int = 7):
        super().__init__()
        h = max(d // reduction, 4)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, d))
        self.conv = nn.Conv1d(2, 1, k, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x: (B, N, d)
        ca = torch.sigmoid(self.mlp(x.mean(1)) + self.mlp(x.amax(1)))
        x = x * ca.unsqueeze(1)
        s = torch.stack([x.mean(-1), x.amax(-1)], dim=1)           # (B, 2, N)
        sa = torch.sigmoid(self.conv(s)).transpose(1, 2)           # (B, N, 1)
        return x * sa


class ECGTransformer(nn.Module):
    def __init__(self, c: HCMANConfig):
        super().__init__()
        self.c = c
        n_p = c.ecg_samples // c.ecg_patch
        self.patch = nn.Linear(c.ecg_patch, c.ecg_dim)
        self.lead_emb = nn.Parameter(torch.zeros(c.ecg_leads, c.ecg_dim))
        self.pos_emb = nn.Parameter(torch.zeros(n_p, c.ecg_dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, c.ecg_dim))
        nn.init.trunc_normal_(self.lead_emb, std=.02); nn.init.trunc_normal_(self.pos_emb, std=.02)
        self.cbam = CBAM1D(c.ecg_dim)
        self.enc = encoder(c.ecg_dim, c.ecg_layers, c.ecg_heads, 4 * c.ecg_dim, c.dropout)
        self.norm = nn.LayerNorm(c.ecg_dim)

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        """ecg (B, 12, 5000) -> tokens (B, 13, ecg_dim): [CLS, one summary token per lead]."""
        B, L, T = ecg.shape
        p = ecg.reshape(B, L, T // self.c.ecg_patch, self.c.ecg_patch)
        x = self.patch(p) + self.lead_emb[None, :, None] + self.pos_emb[None, None]
        x = self.cbam(x.reshape(B, -1, self.c.ecg_dim))            # 1,200 patch tokens
        x = self.norm(self.enc(torch.cat([self.cls.expand(B, -1, -1), x], 1)))
        per_lead = x[:, 1:].reshape(B, L, -1, self.c.ecg_dim).mean(2)
        return torch.cat([x[:, :1], per_lead], 1)


# ============================================================ stream 2 — Echo
class EchoEncoder(nn.Module):
    def __init__(self, c: HCMANConfig):
        super().__init__()
        if c.echo_backbone == "mobilenet_v3_large":
            from torchvision.models import mobilenet_v3_large      # SE ratio .25 + h-swish built in
            m = mobilenet_v3_large(weights=None)
            self.backbone = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
            feat = 960
        else:                                                       # tiny CNN for smoke tests
            self.backbone = nn.Sequential(nn.Conv2d(3, 16, 5, 4, 2), nn.Hardswish(),
                                          nn.Conv2d(16, 32, 3, 2, 1), nn.Hardswish(),
                                          nn.AdaptiveAvgPool2d(1), nn.Flatten())
            feat = 32
        self.fc = nn.Linear(feat, c.echo_dim)
        self.view_emb = nn.Embedding(c.echo_views, c.echo_dim)
        self.frame_emb = nn.Embedding(c.echo_frames, c.echo_dim)
        self.temporal = encoder(c.echo_dim, 1, 8 if c.echo_dim % 8 == 0 else 4, 2 * c.echo_dim, c.dropout)
        self.c = c
        self.export_mode = False

    def forward(self, echo: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        """echo (B, V*T, 3, 224, 224), frame_mask (B, V*T) True = present -> (B, V*T, echo_dim).
        Only present frames go through the CNN (saves most of the compute)."""
        B, N = frame_mask.shape
        if self.export_mode:       # static graph for ONNX: run every frame, zero the missing ones
            out = self.fc(self.backbone(echo.flatten(0, 1))).reshape(B, N, -1)
            out = out * frame_mask.unsqueeze(-1).to(out.dtype)
        else:
            out = echo.new_zeros(B, N, self.c.echo_dim)
            if frame_mask.any():
                out[frame_mask] = self.fc(self.backbone(echo[frame_mask]))
        V, T = self.c.echo_views, self.c.echo_frames
        ids = torch.arange(N, device=echo.device)
        out = out + self.view_emb(ids // T) + self.frame_emb(ids % T)
        # temporal attention across frames/views; all-missing rows keep one dummy key
        pad = ~frame_mask
        pad = torch.cat([pad[:, :1] & frame_mask.any(1, keepdim=True), pad[:, 1:]], 1)
        return self.temporal(out, src_key_padding_mask=pad)


# ============================================================ stream 3 — EHR
class CardioTransformer(nn.Module):
    def __init__(self, c: HCMANConfig):
        super().__init__()
        self.tok = nn.Embedding(c.ehr_vocab, c.ehr_dim, padding_idx=0)
        self.typ = nn.Embedding(8, c.ehr_dim)
        self.val = nn.Linear(1, c.ehr_dim)
        self.enc = encoder(c.ehr_dim, c.ehr_layers, c.ehr_heads, 4 * c.ehr_dim, c.dropout)
        self.norm = nn.LayerNorm(c.ehr_dim)
        self.mlm = nn.Linear(c.ehr_dim, c.ehr_vocab)                # pre-training head
        self.dim = c.ehr_dim

    def forward(self, ehr: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ehr (B, L, 4) [token, type, value, rel_day]; mask (B, L) True = real token."""
        x = (self.tok(ehr[..., 0].long()) + self.typ(ehr[..., 1].long())
             + self.val(ehr[..., 2:3]) + time_encoding(ehr[..., 3], self.dim))
        return self.norm(self.enc(x, src_key_padding_mask=~mask))


# ============================================================ stream 5 — labs
class TemporalTransformer(nn.Module):
    def __init__(self, c: HCMANConfig):
        super().__init__()
        self.lab = nn.Embedding(len(LAB_ORDER), c.labs_dim)
        self.val = nn.Linear(1, c.labs_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, c.labs_dim))
        self.enc = encoder(c.labs_dim, c.labs_layers, c.labs_heads, 4 * c.labs_dim, c.dropout)
        self.head = nn.Linear(c.labs_dim, 1)                       # masked-step pre-training
        self.dim = c.labs_dim

    def forward(self, labs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """labs (B, E, 3) [lab idx, z, t]; mask (B, E). Returns (tokens, key mask). Women with no
        labs get the learned mask token as their single token."""
        B, E, _ = labs.shape
        x = self.lab(labs[..., 0].long()) + self.val(labs[..., 1:2]) + time_encoding(labs[..., 2] * 90, self.dim)
        empty = ~mask.any(1)
        x = torch.where(empty[:, None, None] & (torch.arange(E, device=x.device) == 0)[None, :, None],
                        self.mask_token.expand(B, E, -1), x)
        keep = torch.cat([mask[:, :1] | empty.unsqueeze(1), mask[:, 1:]], 1)   # no in-place (ONNX)
        causal = torch.triu(torch.ones(E, E, dtype=torch.bool, device=x.device), 1)
        return self.enc(x, mask=causal, src_key_padding_mask=~keep), keep


# ============================================================ fusion
class CrossBlock(nn.Module):
    def __init__(self, d: int, kv_dim: int, heads: int, ffn: int, dropout: float):
        super().__init__()
        self.nq, self.nkv = nn.LayerNorm(d), nn.LayerNorm(kv_dim)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, kdim=kv_dim, vdim=kv_dim,
                                          batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ffn), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(ffn, d))
        self.null = nn.Parameter(torch.zeros(1, 1, kv_dim))

    def forward(self, q, kv, kv_keep):
        B = q.shape[0]
        kv = torch.cat([self.null.expand(B, -1, -1), kv], 1)      # always one valid key
        keep = torch.cat([kv_keep.new_ones(B, 1), kv_keep], 1)
        kvn = self.nkv(kv)
        q = q + self.attn(self.nq(q), kvn, kvn, key_padding_mask=~keep, need_weights=False)[0]
        return q + self.ff(q)


class HCMAN(nn.Module):
    def __init__(self, c: HCMANConfig, elm: MultiLayerELM | None = None):
        super().__init__()
        self.c = c
        self.ecg = ECGTransformer(c)
        self.echo = EchoEncoder(c)
        self.ehr = CardioTransformer(c)
        self.elm = elm or MultiLayerELM(c.repro_in, c.elm_hidden)
        self.labs = TemporalTransformer(c)
        d = c.d
        # modality projections into the shared d=512 space
        self.p_ecg, self.p_echo = nn.Linear(c.ecg_dim, d), nn.Linear(c.echo_dim, d)
        self.p_ehr = nn.Linear(c.ehr_dim, d)
        self.repro_tok = nn.Linear(c.elm_hidden[-1], c.repro_tokens * c.elm_hidden[-1])
        self.null_ecg = nn.Parameter(torch.zeros(1, 1 + c.ecg_leads, d))
        # F1 imaging pre-fusion: ECG queries echo
        self.f1 = nn.ModuleList(CrossBlock(d, d, c.fusion_heads, c.ffn, c.dropout) for _ in range(c.f1_layers))
        # F2 clinical integration: bidirectional imaging <-> EHR
        self.f2_img = nn.ModuleList(CrossBlock(d, d, c.fusion_heads, c.ffn, c.dropout) for _ in range(c.f2_layers))
        self.f2_ehr = nn.ModuleList(CrossBlock(d, d, c.fusion_heads, c.ffn, c.dropout) for _ in range(c.f2_layers))
        # F3 women's context: parallel cross-attention to repro and labs, then merge
        r = c.elm_hidden[-1]
        self.f3_repro = nn.ModuleList(CrossBlock(d, r, c.f3_heads, c.ffn, c.dropout) for _ in range(c.f3_layers))
        self.f3_labs = nn.ModuleList(CrossBlock(d, c.labs_dim, c.f3_heads, c.ffn, c.dropout) for _ in range(c.f3_layers))
        self.f3_merge = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d))
        # F4 causal-invariant projection
        self.f4 = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(c.dropout), nn.Linear(d, d), nn.LayerNorm(d))
        # adversaries (through gradient reversal)
        self.site_adv = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, c.n_sites))
        self.fair_adv = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, c.n_fair_groups))
        # six output heads, exactly as in the form
        nc = c.n_conditions
        self.h_cond = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, nc))
        self.h_mace = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 1))
        self.h_pp = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))
        self.h_tier = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 4))
        self.h_atyp = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 1))
        self.h_fmd = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 1))
        self.export_mode = False

    def set_export_mode(self, on: bool = True) -> "HCMAN":
        self.export_mode = self.echo.export_mode = on
        return self

    def forward(self, b: dict[str, torch.Tensor], grl_lambda: float = 1.0) -> dict[str, torch.Tensor]:
        B = b["ehr"].shape[0]
        # ---- encoders
        ecg_present = b["ecg_present"]
        if self.export_mode or ecg_present.any():
            e = self.p_ecg(self.ecg(b["ecg"]))
            q = torch.where(ecg_present[:, None, None], e, self.null_ecg.expand(B, -1, -1))
        else:
            q = self.null_ecg.expand(B, -1, -1)
        echo_tok = self.p_echo(self.echo(b["echo"], b["echo_mask"]))
        ehr_tok = self.p_ehr(self.ehr(b["ehr"], b["ehr_mask"]))
        r = self.elm(b["repro"])                                      # (B, 128), ELM is frozen
        repro_tok = self.repro_tok(r).reshape(B, self.c.repro_tokens, -1)
        labs_tok, labs_keep = self.labs(b["labs"], b["labs_mask"])

        # ---- F1: ECG queries echo
        for blk in self.f1:
            q = blk(q, echo_tok, b["echo_mask"])
        # ---- F2: imaging <-> EHR (bidirectional)
        ehr_keep = b["ehr_mask"]
        for bi, be in zip(self.f2_img, self.f2_ehr):
            q_new = bi(q, ehr_tok, ehr_keep)
            ehr_tok = be(ehr_tok, q, torch.ones(B, q.shape[1], dtype=torch.bool, device=q.device))
            q = q_new
        f2 = torch.cat([q, ehr_tok[:, :1]], 1)                       # imaging tokens + EHR CLS
        # ---- F3: women's context (parallel)
        a, l = f2, f2
        repro_keep = torch.ones(B, repro_tok.shape[1], dtype=torch.bool, device=f2.device)
        for br, bl in zip(self.f3_repro, self.f3_labs):
            a = br(a, repro_tok, repro_keep)
            l = bl(l, labs_tok, labs_keep)
        f3 = self.f3_merge(torch.cat([a, l], -1)).mean(1)
        # ---- F4
        f4 = self.f4(f3)
        out = {
            "condition": self.h_cond(f4),
            "mace": self.h_mace(f4).squeeze(-1),
            "postpartum": self.h_pp(f4).squeeze(-1),
            "risk_tier": self.h_tier(f4),
            "atypical": self.h_atyp(f4).squeeze(-1),
            "fmd_trigger": self.h_fmd(f4).squeeze(-1),
            "embedding": f4,
        }
        if not self.export_mode:   # adversaries are training-only (and not traceable)
            out["site_logits"] = self.site_adv(grad_reverse(f4, grl_lambda))
            out["fair_logits"] = self.fair_adv(grad_reverse(f3, grl_lambda))
        return out

    # parameter groups with the form's three learning rates
    def param_groups(self, lr_enc=1e-4, lr_fusion=3e-4, lr_heads=5e-4, wd=0.01):
        enc, heads, fusion = [], [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if n.split(".")[0] in {"ecg", "echo", "ehr", "labs"}:
                enc.append(p)
            elif n.startswith(("h_", "site_adv", "fair_adv")):
                heads.append(p)
            else:
                fusion.append(p)
        return [{"params": enc, "lr": lr_enc, "weight_decay": wd},
                {"params": fusion, "lr": lr_fusion, "weight_decay": wd},
                {"params": heads, "lr": lr_heads, "weight_decay": wd}]

    def set_imaging_frozen(self, frozen: bool) -> None:
        for m in (self.ecg, self.echo.backbone):
            for p in m.parameters():
                p.requires_grad_(not frozen)
