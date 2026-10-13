"""Turn the adapter's study tables into model-ready tensors.

All fitted statistics (EHR vocabulary + lab bins, lab z-scores, ELM weights) come from the
TRAINING split only and are saved with the model so inference applies the same transform.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from mcp_adapter import concepts as C
from mcp_adapter.config import StudyConfig
from mcp_adapter.features.labs_ehr import EHRTokenizer, LabEncoder
from mcp_adapter.features.repro import FEATURE_NAMES
from mcp_adapter.features.waveforms import ECHO_VIEWS, N_SAMPLES, load_echo, load_ecg

COND_Y = [f"y_{c}" for c in C.CONDITION_ORDER]


def _opt_int(v):
    return None if v is None or pd.isna(v) else int(v)


@dataclass
class Preprocessors:
    ehr: EHRTokenizer
    labs: LabEncoder
    sites: list[str]
    def save(self, path): Path(path).write_bytes(pickle.dumps(self))
    @staticmethod
    def load(path) -> "Preprocessors": return pickle.loads(Path(path).read_bytes())


class StudyData:
    def __init__(self, study_dir: str | Path, cfg: StudyConfig, pre: Preprocessors | None = None):
        d = Path(study_dir)
        t = {n: pd.read_pickle(d / f"{n}.pkl") for n in
             ("cohort", "labels", "repro", "measurements", "ehr_tokens", "ecg_records", "echo_records")}
        self.cfg = cfg
        self.cohort = t["cohort"].merge(t["labels"], on="person_id").merge(t["repro"], on="person_id")
        ids = self.cohort.person_id.tolist()
        train_ids = self.cohort.loc[self.cohort.split == "train", "person_id"]
        if pre is None:
            pre = Preprocessors(
                ehr=EHRTokenizer(lookback_days=cfg.lookback_days).fit(t["ehr_tokens"], t["measurements"], train_ids),
                labs=LabEncoder(window_days=cfg.labs_window_days).fit(t["measurements"], train_ids),
                sites=sorted(self.cohort.loc[self.cohort.split == "train", "site"].astype(str).unique()))
        self.pre = pre
        self.ehr = pre.ehr.transform(t["ehr_tokens"], t["measurements"], ids)
        self.labs = pre.labs.transform(t["measurements"], ids)
        self.ecg = t["ecg_records"].set_index("person_id").path.to_dict() if len(t["ecg_records"]) else {}
        self.echo = {pid: g for pid, g in t["echo_records"].groupby("person_id")} if len(t["echo_records"]) else {}
        self.cohort["site_id"] = self.cohort.site.astype(str).map({s: i for i, s in enumerate(pre.sites)}).fillna(-1)
        # fairness groups: Black (8516) vs not; unknown race (0 / missing) is masked
        r = self.cohort.race_concept_id.fillna(0)
        self.cohort["fair_group"] = np.where(r == C.BLACK_RACE_CONCEPT_ID, 1, np.where(r == 0, -1, 0))

    def split(self, name: str, **kw) -> "HCMANDataset":
        return HCMANDataset(self, self.cohort[self.cohort.split == name].reset_index(drop=True), **kw)

    def repro_matrix(self, name: str = "train") -> torch.Tensor:
        return torch.tensor(self.cohort.loc[self.cohort.split == name, FEATURE_NAMES].to_numpy(np.float32))


class HCMANDataset(Dataset):
    def __init__(self, data: StudyData, rows: pd.DataFrame, echo_frames: int = 8,
                 load_imaging: bool = True):
        self.d, self.rows, self.T, self.load_imaging = data, rows, echo_frames, load_imaging

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows.iloc[i]
        pid = r.person_id
        item = {"person_id": int(pid)}
        ecg = np.zeros((12, N_SAMPLES), np.float32); has_ecg = False
        src = self.d.cfg.ecg_source
        if self.load_imaging and pid in self.d.ecg:
            try:
                ecg, has_ecg = load_ecg(self.d.ecg[pid], src.kind), True
            except Exception:
                pass
        item["ecg"], item["ecg_present"] = ecg, has_ecg

        V, T = len(ECHO_VIEWS), self.T
        echo = np.zeros((V * T, 3, 224, 224), np.float32) if self.load_imaging else np.zeros((V * T, 3, 1, 1), np.float32)
        fmask = np.zeros(V * T, bool)
        if self.load_imaging and pid in self.d.echo:
            for _, rec in self.d.echo[pid].iterrows():
                v = ECHO_VIEWS.index(rec.view) if rec.get("view") in ECHO_VIEWS else None
                if v is None:
                    continue
                try:
                    echo[v * T:(v + 1) * T] = load_echo(rec.path, T, _opt_int(rec.get("ed_frame")),
                                                         _opt_int(rec.get("es_frame")))
                    fmask[v * T:(v + 1) * T] = True
                except Exception:
                    pass
        item["echo"], item["echo_mask"] = echo, fmask

        item["ehr"] = self.d.ehr[pid]
        item["labs"] = self.d.labs[pid]
        item["repro"] = r[FEATURE_NAMES].to_numpy(np.float32)
        item["y_condition"] = r[COND_Y].to_numpy(np.float32)
        for k in ("mace", "postpartum_cvd", "atypical", "fmd_trigger", "risk_tier"):
            item[f"y_{k}"] = float(r[f"y_{k}"]); item[f"m_{k}"] = float(r[f"m_{k}"])
        item["site_id"] = int(r.site_id); item["fair_group"] = int(r.fair_group)
        return item


def collate(items: list[dict]) -> dict[str, torch.Tensor]:
    B = len(items)
    out: dict[str, torch.Tensor] = {}
    out["person_id"] = torch.tensor([x["person_id"] for x in items])
    out["ecg"] = torch.from_numpy(np.stack([x["ecg"] for x in items]))
    out["ecg_present"] = torch.tensor([x["ecg_present"] for x in items])
    out["echo"] = torch.from_numpy(np.stack([x["echo"] for x in items]))
    out["echo_mask"] = torch.from_numpy(np.stack([x["echo_mask"] for x in items]))
    for key, width in (("ehr", 4), ("labs", 3)):
        L = max(1, max(len(x[key]) for x in items))
        arr = np.zeros((B, L, width), np.float32); m = np.zeros((B, L), bool)
        for i, x in enumerate(items):
            n = len(x[key]); arr[i, :n] = x[key]; m[i, :n] = True
        out[key], out[f"{key}_mask"] = torch.from_numpy(arr), torch.from_numpy(m)
    out["repro"] = torch.from_numpy(np.stack([x["repro"] for x in items]))
    out["y_condition"] = torch.from_numpy(np.stack([x["y_condition"] for x in items]))
    for k in ("mace", "postpartum_cvd", "atypical", "fmd_trigger", "risk_tier"):
        out[f"y_{k}"] = torch.tensor([x[f"y_{k}"] for x in items], dtype=torch.float32)
        out[f"m_{k}"] = torch.tensor([x[f"m_{k}"] for x in items], dtype=torch.float32)
    out["site_id"] = torch.tensor([x["site_id"] for x in items])
    out["fair_group"] = torch.tensor([x["fair_group"] for x in items])
    return out
