"""Stream 4 encoder: multi-layer Extreme Learning Machine (ML-ELM), as in Deep MELM.

The form describes "output weights solved analytically via Moore-Penrose pseudoinverse — no
backpropagation". In a multi-layer ELM that has a precise meaning: each layer is an ELM
autoencoder with random input weights; its output weights beta are the ridge least-squares
solution that reconstructs the layer's input from the random hidden activations, and beta^T
becomes that layer's encoding weights (Kasun et al., 2013). No labels are used, so it can be fit
on the training split once, before end-to-end training, and then kept frozen.

SMOTE is NOT needed here: the ELM is unsupervised, and synthesizing tabular rows would not
synthesize the matching ECG/echo/EHR for the same woman. Class imbalance is handled in the
loss (focal) and the sampler instead.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiLayerELM(nn.Module):
    def __init__(self, d_in: int, hidden: tuple[int, ...] = (128, 256, 128), ridge: float = 1e-2,
                 seed: int = 0):
        super().__init__()
        self.dims = (d_in, *hidden)
        self.ridge = ridge
        g = torch.Generator().manual_seed(seed)
        for i, (a, b) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            self.register_buffer(f"W{i}", torch.zeros(a, b))           # fitted encoder weights
            self.register_buffer(f"b{i}", torch.zeros(b))
            self.register_buffer(f"R{i}", torch.randn(a, b, generator=g) / a ** .5)   # random
            self.register_buffer(f"r{i}", torch.randn(b, generator=g) * .1)
        self.register_buffer("mu", torch.zeros(d_in))
        self.register_buffer("sd", torch.ones(d_in))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, X: torch.Tensor) -> "MultiLayerELM":
        """X: (n, d_in) training rows only. Closed-form, one pass per layer."""
        X = X.float()
        self.mu.copy_(X.mean(0)); self.sd.copy_(X.std(0).clamp(min=1e-3))
        H = (X - self.mu) / self.sd
        for i in range(len(self.dims) - 1):
            R, r = getattr(self, f"R{i}"), getattr(self, f"r{i}")
            Z = torch.sigmoid(H @ R + r)                                   # random features (n, b)
            # beta: (b, a) minimizing ||Z beta - H||^2 + ridge ||beta||^2   (pseudoinverse)
            A = Z.T @ Z + self.ridge * torch.eye(Z.shape[1])
            beta = torch.linalg.solve(A, Z.T @ H)
            getattr(self, f"W{i}").copy_(beta.T)                           # (a, b)
            H = torch.sigmoid(H @ beta.T)
        self.fitted.fill_(True)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = (x.float() - self.mu) / self.sd
        for i in range(len(self.dims) - 1):
            h = torch.sigmoid(h @ getattr(self, f"W{i}") + getattr(self, f"b{i}"))
        return h
