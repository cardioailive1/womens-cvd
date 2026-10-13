"""Stream 3 (structured EHR tokens) and stream 5 (serial labs).

Both need statistics (vocabulary, quantile bins, z-score parameters) that are fit on the
TRAINING split only and then applied to validation / test — call `fit` then `transform`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import concepts as C

LOG_ITEMS = {"bnp", "nt_probnp", "troponin_i", "troponin_t"}


def harmonize_units(meas: pd.DataFrame) -> pd.DataFrame:
    m = meas.copy()
    unit = m.unit.fillna("").astype(str).str.lower()
    mult = np.ones(len(m))
    for item, conv in C.UNIT_CONVERSIONS.items():
        sel = (m.item == item).values
        mult[sel] = unit[sel].map(conv).fillna(1.0).values
    m["value"] = m.value_as_number.astype(float) * mult
    m.loc[m.item.isin(LOG_ITEMS), "value"] = np.log1p(m.loc[m.item.isin(LOG_ITEMS), "value"].clip(lower=0))
    m["event_date"] = pd.to_datetime(m.event_date)
    m["rel_day"] = (m.event_date - pd.to_datetime(m.index_date)).dt.days
    return m


# ------------------------------------------------------------------ stream 5
@dataclass
class LabEncoder:
    window_days: int = 90
    max_events: int = 64
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)

    def fit(self, meas: pd.DataFrame, train_ids) -> "LabEncoder":
        m = meas[meas.person_id.isin(train_ids) & meas.item.isin(C.LAB_ORDER)]
        for item in C.LAB_ORDER:
            v = m.loc[m.item == item, "value"]
            self.mean[item] = float(v.mean()) if len(v) else 0.0
            self.std[item] = float(v.std()) if len(v) > 1 and v.std() > 0 else 1.0
        return self

    def transform(self, meas: pd.DataFrame, person_ids) -> dict[int, np.ndarray]:
        """person_id -> array (n_events, 3): [lab index, z-value, rel_day / window].
        Women with no labs get an empty array; the model substitutes its learned mask token."""
        m = meas[meas.item.isin(C.LAB_ORDER) & meas.rel_day.between(-self.window_days, 0)].copy()
        m["li"] = m.item.map({k: i for i, k in enumerate(C.LAB_ORDER)})
        m["z"] = (m.value - m.item.map(self.mean)) / m.item.map(self.std)
        m["z"] = m.z.clip(-6, 6)
        m["t"] = m.rel_day / self.window_days
        m = m.sort_values(["person_id", "rel_day"])
        out = {pid: np.zeros((0, 3), np.float32) for pid in person_ids}
        for pid, g in m.groupby("person_id"):
            if pid in out:
                out[pid] = g[["li", "z", "t"]].to_numpy(np.float32)[-self.max_events:]
        return out


# ------------------------------------------------------------------ stream 3
SPECIAL = ["[PAD]", "[CLS]", "[UNK]", "[MASK]"]
TTYPES = {"PAD": 0, "CLS": 1, "DX": 2, "RX": 3, "LAB": 4, "VIT": 5}


@dataclass
class EHRTokenizer:
    lookback_days: int = 365
    max_len: int = 512
    min_count: int = 5                # rare codes -> [UNK]; also a privacy floor
    n_bins: int = 10
    vocab: dict[str, int] = field(default_factory=dict)
    bin_edges: dict[str, np.ndarray] = field(default_factory=dict)
    vit_mean: dict[str, float] = field(default_factory=dict)
    vit_std: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def _is_leaky(code: str) -> bool:
        return any(code.startswith(p) for p in C.LEAKY_CODE_PREFIXES)

    def fit(self, tokens: pd.DataFrame, meas: pd.DataFrame, train_ids) -> "EHRTokenizer":
        t = tokens[tokens.person_id.isin(train_ids)]
        t = t[~((t.ttype == "DX") & t.code.map(self._is_leaky))]
        keys = t.ttype + ":" + t.code.astype(str)
        # count women, not rows, toward the min_count floor
        women = (t.assign(k=keys).drop_duplicates(["person_id", "k"]).groupby("k").size())
        keep = women[women >= self.min_count].index
        m = meas[meas.person_id.isin(train_ids)]
        lab_keys = [f"LAB:{i}:q{b}" for i in C.LAB_ORDER for b in range(self.n_bins)]
        vit_keys = [f"VIT:{v}" for v in C.VITALS]
        self.vocab = {s: i for i, s in enumerate(SPECIAL + sorted(keep) + lab_keys + vit_keys)}
        for item in C.LAB_ORDER:
            v = m.loc[m.item == item, "value"].dropna()
            qs = np.quantile(v, np.linspace(0, 1, self.n_bins + 1)[1:-1]) if len(v) else np.zeros(self.n_bins - 1)
            self.bin_edges[item] = qs
        for vt in C.VITALS:
            v = m.loc[m.item == vt, "value"].dropna()
            self.vit_mean[vt] = float(v.mean()) if len(v) else 0.0
            self.vit_std[vt] = float(v.std()) if len(v) > 1 and v.std() > 0 else 1.0
        return self

    def transform(self, tokens: pd.DataFrame, meas: pd.DataFrame, person_ids) -> dict[int, np.ndarray]:
        """person_id -> int/float array (L, 4): [token_id, token_type, value, rel_day]."""
        unk = self.vocab["[UNK]"]
        t = tokens[~((tokens.ttype == "DX") & tokens.code.map(self._is_leaky))].copy()
        t["tid"] = (t.ttype + ":" + t.code.astype(str)).map(self.vocab).fillna(unk).astype(int)
        t["tt"] = t.ttype.map(TTYPES)
        t["val"] = 0.0
        parts = [t[["person_id", "tid", "tt", "val", "rel_day"]]]

        m = meas[meas.rel_day.between(-self.lookback_days, 0)]
        lab = m[m.item.isin(C.LAB_ORDER)].copy()
        if len(lab):
            lab["bin"] = [int(np.searchsorted(self.bin_edges[i], v)) for i, v in zip(lab.item, lab.value)]
            lab["tid"] = (("LAB:" + lab.item + ":q" + lab.bin.astype(str)).map(self.vocab)
                          .fillna(unk).astype(int))
            lab["tt"], lab["val"] = TTYPES["LAB"], 0.0
            parts.append(lab[["person_id", "tid", "tt", "val", "rel_day"]])
        vit = m[m.item.isin(list(C.VITALS))].copy()
        if len(vit):
            vit["tid"] = ("VIT:" + vit.item).map(self.vocab).astype(int)
            vit["tt"] = TTYPES["VIT"]
            vit["val"] = ((vit.value - vit.item.map(self.vit_mean)) / vit.item.map(self.vit_std)).clip(-6, 6)
            parts.append(vit[["person_id", "tid", "tt", "val", "rel_day"]])

        allt = pd.concat(parts).sort_values(["person_id", "rel_day"])
        cls = np.array([[self.vocab["[CLS]"], TTYPES["CLS"], 0.0, 0.0]], np.float32)
        out = {pid: cls.copy() for pid in person_ids}
        for pid, g in allt.groupby("person_id"):
            if pid in out:
                arr = g[["tid", "tt", "val", "rel_day"]].to_numpy(np.float32)[-(self.max_len - 1):]
                out[pid] = np.concatenate([cls, arr])
        return out

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
