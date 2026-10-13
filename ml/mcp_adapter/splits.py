"""Patient-level train / validation / test splits.

Each woman appears in exactly one split. Splits are stratified on (stratum, any-condition) so
rare strata are represented everywhere. Optional extras for external-validity checks:
  - temporal holdout: index dates after a (shifted) day go to `test_temporal`
  - site holdout: one Mayo campus held out entirely as `test_site` (needs >= 3 sites)
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
