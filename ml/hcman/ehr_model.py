"""Stream 3 — EHR: a simple tabular model on diagnosis and medication history.

Replaces the CardioTransformer (MCP has approved exporting the weights of this model and of the
MobileNet1D ECG encoder). Features are binary: a diagnosis category (ICD-10-CM, first three
characters) or a drug class (ATC level 4) recorded in the lookback window before index. Codes that
define a condition label are never features (C.LEAKY_CODE_PREFIXES), and a feature is kept only if
at least `min_count` training women have it (11 = MCP's minimum cell size). Lab results are not
features here: they have their own stream.

One model per target (logistic regression by default, XGBoost optional). Their scores — a logit per
binary target and log-probabilities for the four risk tiers — are HCMAN's EHR input. Training women
get OUT-OF-FOLD scores (each scored by a model that never saw her label); validation, test and
inference use the model fitted on the whole training split.

The fitted model is plain numbers (coefficients and intercepts, or XGBoost trees) and is written to
`ehr_model.json` in the export bundle.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from mcp_adapter import concepts as C
from mcp_adapter.concepts import CONDITION_ORDER

log = logging.getLogger("hcman.ehr_model")

BINARY_TARGETS = [*CONDITION_ORDER, "mace", "postpartum_cvd", "atypical", "fmd_trigger"]
N_TIERS = 4
SCORE_NAMES = [f"logit_{t}" for t in BINARY_TARGETS] + [f"logp_tier{k}" for k in range(N_TIERS)]
N_SCORES = len(SCORE_NAMES)
CLIP = 10.0


def _is_leaky(code: str) -> bool:
    return any(str(code).startswith(p) for p in C.LEAKY_CODE_PREFIXES)


def feature_key(ttype: str, code: str) -> str | None:
    """DX -> ICD-10-CM category (3 characters); RX -> ATC 4th level (5 characters)."""
    code = str(code)
    if ttype == "DX":
        return None if _is_leaky(code) else f"DX:{code[:3]}"
    if ttype == "RX":
        return f"RX:{code[:5]}"
    return None


def featurize(tokens: pd.DataFrame, person_ids, lookback_days: int = 365) -> dict[int, set[str]]:
    """person_id -> set of feature keys from her EHR tokens on days [-lookback, 0]."""
    out = {int(p): set() for p in person_ids}
    if tokens is None or not len(tokens):
        return out
    t = tokens[tokens.ttype.isin(["DX", "RX"]) & tokens.rel_day.between(-lookback_days, 0)]
    keys = [feature_key(a, b) for a, b in zip(t.ttype, t.code)]
    for pid, k in zip(t.person_id, keys):
        if k is not None and int(pid) in out:
            out[int(pid)].add(k)
    return out


def _labels(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(Y, M) for the binary targets and (y_tier, m_tier) from cohort rows with y_* / m_* columns."""
    Y = np.stack([rows[f"y_{t}"].to_numpy(float) for t in BINARY_TARGETS], 1)
    M = np.stack([rows[f"m_{t}"].to_numpy(float) if f"m_{t}" in rows else np.ones(len(rows))
                  for t in BINARY_TARGETS], 1)
    return Y, M > 0, rows["y_risk_tier"].to_numpy(float), rows["m_risk_tier"].to_numpy(float) > 0


@dataclass
class EHRModel:
    kind: str = "logistic"            # logistic | xgboost
    C: float = 1.0                    # inverse L2 strength (logistic)
    min_count: int = 11               # women per feature; MCP's minimum cell size
    min_class: int = 5                # fewer positives (or negatives) -> constant score
    n_folds: int = 5
    seed: int = 0
    xgb: dict = field(default_factory=lambda: {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
                                               "subsample": 0.8, "colsample_bytree": 0.8})
    features: list[str] = field(default_factory=list)
    models: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- features
    def fit_vocab(self, feats: dict[int, set[str]], train_ids) -> "EHRModel":
        counts: dict[str, int] = {}
        for pid in train_ids:
            for k in feats.get(int(pid), ()):
                counts[k] = counts.get(k, 0) + 1
        self.features = sorted(k for k, n in counts.items() if n >= self.min_count)
        return self

    def matrix(self, feats: dict[int, set[str]], person_ids) -> sparse.csr_matrix:
        col = {k: j for j, k in enumerate(self.features)}
        rows, cols = [], []
        for i, pid in enumerate(person_ids):
            for k in feats.get(int(pid), ()):
                if k in col:
                    rows.append(i); cols.append(col[k])
        data = np.ones(len(rows), np.float32)
        return sparse.csr_matrix((data, (rows, cols)), shape=(len(person_ids), max(len(self.features), 1)))

    @staticmethod
    def present(X: sparse.csr_matrix) -> np.ndarray:
        return np.asarray(X.getnnz(axis=1) > 0)

    # ---------------------------------------------------------------- fitting
    def _binary(self, X, y) -> dict:
        pos, neg = int(y.sum()), int(len(y) - y.sum())
        if pos < self.min_class or neg < self.min_class or X.shape[1] == 0:
            p = (pos + 1) / (len(y) + 2)
            return {"type": "const", "value": float(np.log(p / (1 - p)))}
        if self.kind == "xgboost":
            import xgboost as xgb
            m = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                                  random_state=self.seed, scale_pos_weight=neg / pos, n_jobs=1, **self.xgb)
            m.fit(X, y)
            return {"type": "xgb", "raw": m.get_booster().save_raw("json").decode()}
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=self.C, class_weight="balanced", solver="liblinear", max_iter=2000,
                               random_state=self.seed).fit(X, y)
        return {"type": "lr", "coef": m.coef_[0].astype(float).tolist(), "intercept": float(m.intercept_[0])}

    def _tier(self, X, y) -> dict:
        classes, counts = np.unique(y.astype(int), return_counts=True)
        if len(classes) < 2 or X.shape[1] == 0:
            p = np.full(N_TIERS, 1.0)
            for c, n in zip(classes, counts):
                p[c] += n
            return {"type": "const", "value": np.log(p / p.sum()).tolist()}
        if self.kind == "xgboost":
            import xgboost as xgb
            yy = np.searchsorted(classes, y.astype(int))
            m = xgb.XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", tree_method="hist",
                                  random_state=self.seed, n_jobs=1, **self.xgb)
            m.fit(X, yy)
            return {"type": "xgb_multi", "classes": classes.tolist(), "raw": m.get_booster().save_raw("json").decode()}
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=self.C, class_weight="balanced", max_iter=2000, random_state=self.seed).fit(X, y.astype(int))
        coef, icpt = m.coef_, m.intercept_
        if len(classes) == 2:                       # sklearn returns one row for two classes
            coef, icpt = np.vstack([-coef[0] / 2, coef[0] / 2]), np.array([-icpt[0] / 2, icpt[0] / 2])
        return {"type": "lr_multi", "classes": classes.tolist(), "coef": coef.astype(float).tolist(),
                "intercept": icpt.astype(float).tolist()}

    def _fit_all(self, X, rows: pd.DataFrame) -> dict:
        Y, M, yt, mt = _labels(rows)
        out = {t: self._binary(X[M[:, j]], Y[M[:, j], j]) for j, t in enumerate(BINARY_TARGETS)}
        out["risk_tier"] = self._tier(X[mt], yt[mt])
        return out

    def fit(self, X, rows: pd.DataFrame) -> "EHRModel":
        self.models = self._fit_all(X, rows)
        return self

    # ---------------------------------------------------------------- scoring
    @staticmethod
    def _apply(models: dict, X) -> np.ndarray:
        n = X.shape[0]
        S = np.zeros((n, N_SCORES), np.float32)
        for j, t in enumerate(BINARY_TARGETS):
            S[:, j] = _score_binary(models[t], X)
        S[:, len(BINARY_TARGETS):] = _score_tier(models["risk_tier"], X)
        return np.clip(S, -CLIP, CLIP)

    def scores(self, X) -> np.ndarray:
        if not self.models:
            raise RuntimeError("EHRModel is not fitted")
        return self._apply(self.models, X)

    def oof_scores(self, X, rows: pd.DataFrame) -> np.ndarray:
        """Out-of-fold scores for the training split: each woman scored by a model fit without her."""
        n = X.shape[0]
        if n < 2 * self.n_folds:
            log.warning("only %d training women: out-of-fold scores fall back to the full fit", n)
            return self._apply(self._fit_all(X, rows), X)
        rng = np.random.default_rng(self.seed)
        fold = rng.permutation(np.arange(n) % self.n_folds)
        S = np.zeros((n, N_SCORES), np.float32)
        for k in range(self.n_folds):
            tr, te = fold != k, fold == k
            S[te] = self._apply(self._fit_all(X[tr], rows.iloc[np.flatnonzero(tr)]), X[te])
        return S

    # ---------------------------------------------------------------- export
    def to_json(self) -> str:
        return json.dumps({"kind": self.kind, "features": self.features, "score_names": SCORE_NAMES,
                           "min_count": self.min_count, "models": self.models})

    @classmethod
    def from_json(cls, s: str) -> "EHRModel":
        d = json.loads(s)
        return cls(kind=d["kind"], min_count=d["min_count"], features=d["features"], models=d["models"])


def _xgb_margin(raw: str, X) -> np.ndarray:
    import xgboost as xgb
    b = xgb.Booster(); b.load_model(bytearray(raw.encode()))
    return b.predict(xgb.DMatrix(X), output_margin=True)


def _score_binary(m: dict, X) -> np.ndarray:
    if m["type"] == "const":
        return np.full(X.shape[0], m["value"], np.float32)
    if m["type"] == "lr":
        return np.asarray(X @ np.asarray(m["coef"])).ravel() + m["intercept"]
    return _xgb_margin(m["raw"], X)


def _score_tier(m: dict, X) -> np.ndarray:
    n = X.shape[0]
    if m["type"] == "const":
        return np.tile(np.asarray(m["value"], np.float32), (n, 1))
    z = np.full((n, N_TIERS), -CLIP, np.float64)
    if m["type"] == "lr_multi":
        sub = np.asarray(X @ np.asarray(m["coef"]).T) + np.asarray(m["intercept"])
    else:
        sub = _xgb_margin(m["raw"], X).reshape(n, -1)
    z[:, m["classes"]] = sub
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))          # log-probabilities
