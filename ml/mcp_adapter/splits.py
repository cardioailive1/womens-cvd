"""Patient-level train / validation / test splits.

Each woman appears in exactly one split. Splits are stratified on (stratum, any-condition) so
rare strata are represented everywhere. Optional extras for external-validity checks:
  - temporal holdout: index dates after a (shifted) day go to `test_temporal`
  - site holdout: one Mayo campus held out entirely as `test_site` (needs >= 3 sites)
Locking (`lock_splits`): the first build saves every woman's split to the project work dataset;
later builds reuse it, so a validation or test woman can never drift into training when the data
are refreshed or the cohort is rebuilt. Only women not seen before get a new split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .concepts import CONDITION_ORDER
from .config import StudyConfig


def make_splits(cohort: pd.DataFrame, labels: pd.DataFrame, cfg: StudyConfig,
                holdout_site: str | None = None) -> pd.Series:
    df = cohort[["person_id", "stratum", "site", "index_date"]].merge(labels, on="person_id")
    any_cond = (df[[f"y_{c}" for c in CONDITION_ORDER]].sum(axis=1) > 0).astype(int)
    split = pd.Series("train", index=df.person_id, name="split")

    pool = df.copy()
    if cfg.temporal_holdout_after_day is not None:
        late = pool.index_date >= pd.Timestamp("1970-01-01") + pd.Timedelta(days=cfg.temporal_holdout_after_day)
        split[pool.person_id[late].values] = "test_temporal"
        pool = pool[~late]
    if holdout_site is not None:
        s = pool.site == holdout_site
        split[pool.person_id[s].values] = "test_site"
        pool = pool[~s]

    strat = pool.stratum + "|" + any_cond.reindex(pool.index).astype(str)
    counts = strat.value_counts()
    strat = strat.where(strat.map(counts) >= 6, pool.stratum)   # merge tiny strata
    tv, test = train_test_split(pool.person_id, test_size=cfg.test_fraction,
                                stratify=strat, random_state=cfg.seed)
    rel_val = cfg.val_fraction / (1 - cfg.test_fraction)
    train, val = train_test_split(tv, test_size=rel_val, stratify=strat.loc[tv.index],
                                  random_state=cfg.seed)
    split[test.values] = "test"
    split[val.values] = "val"
    return split


def lock_splits(be, cfg: StudyConfig, split: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Reuse the locked split for women already assigned; save the union back.

    split: person_id, split (freshly computed). Returns (split, counts). Disabled when
    cfg.split_lock_table is empty. The table lives in the work dataset (MCP, 28 Oct 2026)."""
    table = cfg.split_lock_table
    if not table:
        return split, {}
    try:
        locked = be.sql(f"SELECT person_id, split FROM {table}")
    except Exception:                       # first build: nothing locked yet
        locked = pd.DataFrame(columns=["person_id", "split"])
    locked = locked.drop_duplicates("person_id")
    prev = split.person_id.map(locked.set_index("person_id").split)
    out = split.assign(split=prev.fillna(split.split))
    moved = int((prev.notna() & (prev != split.split)).sum())
    new = out[~out.person_id.isin(locked.person_id)]
    keep = pd.concat([locked, new[["person_id", "split"]]], ignore_index=True)
    keep["person_id"] = keep.person_id.astype("int64")
    (getattr(be, "save_table", None) or be.register)(keep, table)
    counts = {"reused from the locked split": int(prev.notna().sum()), "newly assigned": len(new),
              "kept locked where the new draw differed": moved}
    return out, counts
