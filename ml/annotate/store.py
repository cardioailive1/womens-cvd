"""Append-only annotation store shared by the automatic annotators and the review tool.

Every label is one row: which item, which task, the value, who or what produced it, and when.
Nothing is overwritten, so the history of auto-labels and human corrections is kept for audit.
`latest()` resolves each (item, task) to one value, with human review always beating automatic.

The store lives inside the workbench (it references patient-level items) and never leaves it.
"""
from __future__ import annotations

import getpass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

COLUMNS = ["item_id", "task", "value", "confidence", "source", "annotator", "ts", "note"]
# source: "auto:<method>" for machine labels, "human" for reviewer labels


class AnnotationStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=COLUMNS)
        return pd.read_csv(self.path, dtype={"item_id": str, "value": str, "note": str})

    def add(self, rows: list[dict] | pd.DataFrame) -> int:
        df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows.copy()
        if df.empty:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        df["ts"] = df["ts"].fillna(now) if "ts" in df else now
        for c in COLUMNS:
            if c not in df:
                df[c] = None
        df["item_id"] = df.item_id.astype(str)
        df[COLUMNS].to_csv(self.path, mode="a", header=not self.path.exists(), index=False)
        return len(df)

    def add_human(self, item_id: str, task: str, value, annotator: str | None = None,
                  note: str | None = None) -> None:
        self.add([{"item_id": str(item_id), "task": task, "value": str(value), "confidence": 1.0,
                   "source": "human", "annotator": annotator or getpass.getuser(), "note": note}])

    def latest(self, task: str | None = None) -> pd.DataFrame:
        """One row per (item_id, task): the newest human label if any, else the newest auto label."""
        df = self.read()
        if task:
            df = df[df.task == task]
        if df.empty:
            return df
        df = df.assign(is_human=(df.source == "human").astype(int)).sort_values(["is_human", "ts"], kind="stable")
        return df.groupby(["item_id", "task"]).tail(1).drop(columns="is_human").reset_index(drop=True)

    def human_items(self, task: str) -> set[str]:
        df = self.read()
        return set(df[(df.task == task) & (df.source == "human")].item_id)
