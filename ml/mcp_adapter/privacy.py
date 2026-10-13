"""Aggregate-only reporting with small-cell suppression.

MCP lets only aggregate results leave the workbench, above a de-identification threshold and
after approval. Everything written to the report directory goes through `suppress` so a report
can be submitted for export as-is. The default threshold (11) follows common CMS practice; use
the value MCP gives you.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def suppress(df: pd.DataFrame, count_cols: list[str], k: int) -> pd.DataFrame:
    """Replace counts in (0, k) with '<k' and blank any derived percentage in that row.
    Also applies complementary suppression when one hidden cell in a column could be
    recovered from a total (the column's second-smallest count is hidden too)."""
    out = df.copy().astype(object)
    for c in count_cols:
        v = pd.to_numeric(df[c], errors="coerce")
        small = (v > 0) & (v < k)
        if small.sum() == 1 and (v >= k).sum() > 0:
            small[v[v >= k].idxmin()] = True
        out.loc[small, c] = f"<{k}"
        for pc in [x for x in df.columns if x.startswith(c + "_pct") or x == f"pct_{c}"]:
            out.loc[small, pc] = None
    return out


def write_report(obj: dict | pd.DataFrame, name: str, report_dir: str | Path) -> Path:
    p = Path(report_dir); p.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        path = p / f"{name}.csv"; obj.to_csv(path, index=False)
    else:
        path = p / f"{name}.json"
        path.write_text(json.dumps(obj, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))
    return path
