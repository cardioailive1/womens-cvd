"""Two-tier audit of the automatic labels (echo annotations and condition labels).

  Tier 1 — data scientists audit EVERY item in the audit set. They see the automatic label and
           answer agree / disagree (with the corrected value) / uncertain. Optionally more than
           one data scientist per item (`tier1_per_item`).
  Tier 2 — a cardiologist adjudicates, blinded to both the automatic and the data-scientist label:
             * every disagreement (data scientist vs automatic label, or between data scientists)
             * every item a data scientist marked uncertain
             * a blinded spot-check: a random share of the items tier 1 accepted, which measures
               how many wrong labels slip through tier 1 (automation bias included).
           The three kinds are shuffled together so the cardiologist cannot tell which is which.

Final label per item: the cardiologist's when there is one; otherwise the automatic label if
tier 1 agreed; otherwise pending (not used for training until adjudicated).

Roles are enforced: only listed data scientists can submit tier 1, only listed cardiologists
tier 2, and nobody adjudicates an item they audited at tier 1. Everything is appended to the
same append-only AnnotationStore used by the other tools.

    from annotate.audit import LabelAudit, echo_audit_items, phenotype_audit_items
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .echo import T_ED, T_ES, T_VIEW, VIEWS, item_id as clip_id
from .phenotype import task_name, wilson
from .store import AnnotationStore

VERDICTS = ("agree", "disagree", "uncertain")
ECHO_TASK = "echo"
FRAME_TOLERANCE = 2          # ED / ES within this many frames count as the same


# ------------------------------------------------------------------ values
def echo_value(view, ed=None, es=None) -> str:
    clean = lambda v: None if v is None or (isinstance(v, float) and math.isnan(v)) or v == "" else int(float(v))
    return json.dumps({"view": view or None, "ed": clean(ed), "es": clean(es)}, sort_keys=True)


def same(task: str, a, b) -> bool:
    """Whether two labels for the same item count as agreeing."""
    if a is None or b is None or (isinstance(a, float) and pd.isna(a)) or (isinstance(b, float) and pd.isna(b)):
        return False
    if task != ECHO_TASK:
        return str(a).strip().lower() == str(b).strip().lower()
    x, y = json.loads(a), json.loads(b)
    if x.get("view") != y.get("view"):
        return False
    if x.get("view") not in VIEWS[:4]:
        return True
    for k in ("ed", "es"):
        if x.get(k) is not None and y.get(k) is not None and abs(x[k] - y[k]) > FRAME_TOLERANCE:
            return False
    return True


def _hash(key: str, seed: int) -> float:
    return int(hashlib.sha1(f"{seed}:{key}".encode()).hexdigest()[:12], 16) / float(16 ** 12)


# ------------------------------------------------------------------ audit sets
def phenotype_audit_items(sample: pd.DataFrame) -> pd.DataFrame:
    """From annotate.phenotype.sample_for_review: the automatic yes/no per (woman, condition)."""
    return pd.DataFrame({"item_id": sample.person_id.astype(str),
                         "task": [task_name(c) for c in sample.condition],
                         "auto_value": np.where(sample.algo_label == 1, "yes", "no")})


def echo_audit_items(store: AnnotationStore, manifest: pd.DataFrame | None = None,
                     sample: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Every automatically annotated echo clip (or a random `sample` of them): view + ED + ES."""
    lat = store.latest()
    lat = lat[lat.task.isin([T_VIEW, T_ED, T_ES]) & ~lat.source.eq("human")]
    if lat.empty:
        return pd.DataFrame(columns=["item_id", "task", "auto_value"])
    piv = lat.pivot_table(index="item_id", columns="task", values="value", aggfunc="last")
    piv = piv[piv.get(T_VIEW, pd.Series(dtype=str)).fillna("").isin(VIEWS)]
    if manifest is not None:
        piv = piv[piv.index.isin({clip_id(p) for p in manifest.path.astype(str)})]
    items = pd.DataFrame({"item_id": piv.index, "task": ECHO_TASK,
                          "auto_value": [echo_value(r.get(T_VIEW), r.get(T_ED), r.get(T_ES))
                                         for _, r in piv.iterrows()]})
    if sample is not None and sample < len(items):
        items = items.sample(sample, random_state=seed)
    return items.reset_index(drop=True)


# ------------------------------------------------------------------ the audit
@dataclass
class LabelAudit:
    items: pd.DataFrame                       # item_id, task, auto_value
    store: AnnotationStore
    data_scientists: list[str]
    cardiologists: list[str]
    spot_check_fraction: float = 0.10         # of tier-1-accepted items
    spot_check_min: int = 20
    tier1_per_item: int = 1
    seed: int = 0
    _key: pd.Index = field(init=False, repr=False)

    def __post_init__(self):
        if isinstance(self.store, str):
            self.store = AnnotationStore(self.store)
        overlap = set(self.data_scientists) & set(self.cardiologists)
        if overlap:
            raise ValueError(f"{sorted(overlap)} listed as both data scientist and cardiologist")
        self.items = self.items.drop_duplicates(["item_id", "task"]).reset_index(drop=True)
        self.items["item_id"] = self.items.item_id.astype(str)

    @classmethod
    def from_config(cls, items: pd.DataFrame, store, cfg) -> "LabelAudit":
        a = cfg.audit or {}
        if not a.get("data_scientists") or not a.get("cardiologists"):
            raise ValueError("list audit.data_scientists and audit.cardiologists in config/study.yaml")
        return cls(items, store, list(a["data_scientists"]), list(a["cardiologists"]),
                   spot_check_fraction=float(a.get("spot_check_fraction", 0.10)),
                   spot_check_min=int(a.get("spot_check_min", 20)),
                   tier1_per_item=int(a.get("tier1_per_item", 1)), seed=cfg.seed)

    # ---- raw reads
    def _tier(self, tier: int) -> pd.DataFrame:
        df = self.store.read()
        df = df[df.task.str.startswith(f"audit{tier}:", na=False) & (df.source == "human")].copy()
        df["task"] = df.task.str.slice(len(f"audit{tier}:"))
        if tier == 1:
            parsed = df.value.map(json.loads)
            df["verdict"] = parsed.map(lambda v: v["verdict"])
            df["corrected"] = parsed.map(lambda v: v.get("value"))
        # one answer per reviewer per item: their latest
        return df.sort_values("ts", kind="stable").groupby(["item_id", "task", "annotator"]).tail(1)

    # ---- tier 1
    def tier1_queue(self, annotator: str) -> pd.DataFrame:
        """Items this data scientist has not audited that still need tier-1 reviews."""
        t1 = self._tier(1)
        mine = set(zip(t1[t1.annotator == annotator].item_id, t1[t1.annotator == annotator].task))
        n = t1.groupby(["item_id", "task"]).size()
        need = [(i, t) not in mine and n.get((i, t), 0) < self.tier1_per_item
                for i, t in zip(self.items.item_id, self.items.task)]
        q = self.items[need]
        return q.assign(_h=[_hash(f"{annotator}|{i}|{t}", self.seed) for i, t in zip(q.item_id, q.task)]
                        ).sort_values("_h").drop(columns="_h").reset_index(drop=True)

    def submit_tier1(self, item_id, task: str, verdict: str, annotator: str,
                     value=None, note: str | None = None) -> None:
        if annotator not in self.data_scientists:
            raise PermissionError(f"{annotator} is not a listed data scientist (tier 1)")
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}")
        auto = self._auto(item_id, task)
        if verdict == "disagree":
            if value is None and task != ECHO_TASK:          # binary: disagreeing flips the label
                value = {"yes": "no", "no": "yes"}.get(str(auto).lower())
            if value is None or same(task, value, auto):
                raise ValueError("a disagreement needs a corrected value that differs from the automatic label")
        payload = {"verdict": verdict, "value": value if verdict == "disagree" else
                   (auto if verdict == "agree" else None)}
        self.store.add_human(str(item_id), f"audit1:{task}", json.dumps(payload), annotator, note)

    # ---- tier 2
    def tier2_queue(self, annotator: str | None = None) -> pd.DataFrame:
        """Shuffled cardiologist queue: disagreements, uncertain items and the blinded spot-check.
        `reason` is for reporting only; the review screen never shows it."""
        t1 = self._tier(1)
        rows = []
        done1 = t1.groupby(["item_id", "task"])
        accepted = []
        for (i, t), g in done1:
            if len(g) < self.tier1_per_item:
                continue
            auto = self._auto(i, t)
            labels = [r.corrected if r.verdict == "disagree" else auto for r in g.itertuples() if r.verdict != "uncertain"]
            if (g.verdict == "uncertain").any():
                rows.append((i, t, "uncertain"))
            elif not all(same(t, x, auto) for x in labels):
                rows.append((i, t, "disagreement"))
            else:
                accepted.append((i, t))
        # blinded spot-check over tier-1-accepted items: hash threshold, topped up to the minimum
        acc = sorted(accepted, key=lambda k: _hash(f"spot|{k[0]}|{k[1]}", self.seed))
        k = min(len(acc), max(self.spot_check_min, math.ceil(self.spot_check_fraction * len(acc))))
        rows += [(i, t, "spot-check") for i, t in acc[:k]]
        q = pd.DataFrame(rows, columns=["item_id", "task", "reason"])
        t2 = self._tier(2)
        done = set(zip(t2.item_id, t2.task))
        q = q[[(i, t) not in done for i, t in zip(q.item_id, q.task)]]
        if annotator is not None:                            # never adjudicate your own tier-1 item
            own = set(zip(t1[t1.annotator == annotator].item_id, t1[t1.annotator == annotator].task))
            q = q[[(i, t) not in own for i, t in zip(q.item_id, q.task)]]
        q = q.assign(_h=[_hash(f"t2|{i}|{t}", self.seed) for i, t in zip(q.item_id, q.task)])
        return q.sort_values("_h").drop(columns="_h").reset_index(drop=True)

    def submit_tier2(self, item_id, task: str, value, annotator: str, note: str | None = None) -> None:
        """The cardiologist's own label. For conditions: yes / no / unsure (unsure stays pending).
        For echo: echo_value(view, ed, es), view may be UNUSABLE."""
        if annotator not in self.cardiologists:
            raise PermissionError(f"{annotator} is not a listed cardiologist (tier 2)")
        t1 = self._tier(1)
        if ((t1.item_id == str(item_id)) & (t1.task == task) & (t1.annotator == annotator)).any():
            raise PermissionError("an item cannot be adjudicated by someone who audited it at tier 1")
        self.store.add_human(str(item_id), f"audit2:{task}", value, annotator, note)

    # ---- results
    def _auto(self, item_id, task):
        m = self.items[(self.items.item_id == str(item_id)) & (self.items.task == task)]
        if m.empty:
            raise KeyError(f"{item_id} / {task} is not in the audit set")
        return m.auto_value.iloc[0]

    def final_labels(self) -> pd.DataFrame:
        """item_id, task, auto_value, final_value, status, reason."""
        t1, t2 = self._tier(1), self._tier(2)
        t2v = t2.set_index(["item_id", "task"]).value if len(t2) else pd.Series(dtype=str)
        q = self.tier2_queue()
        all_t2 = pd.concat([q, self._tier2_reasons_done()], ignore_index=True)
        reason = {(r.item_id, r.task): r.reason for r in all_t2.itertuples()}
        n1 = t1.groupby(["item_id", "task"]).size() if len(t1) else pd.Series(dtype=int)
        out = []
        for r in self.items.itertuples():
            key = (r.item_id, r.task)
            why = reason.get(key, "")
            if key in t2v.index:
                v = t2v[key]
                status = "pending" if str(v).lower() == "unsure" else "cardiologist"
                final = None if status == "pending" else v
            elif n1.get(key, 0) < self.tier1_per_item:
                final, status = None, "awaiting tier 1"
            elif why in ("disagreement", "uncertain"):
                final, status = None, "awaiting tier 2"
            else:
                final, status = r.auto_value, "tier 1 confirmed"
            out.append({"item_id": r.item_id, "task": r.task, "auto_value": r.auto_value,
                        "final_value": final, "status": status, "reason": why})
        return pd.DataFrame(out)

    def _tier2_reasons_done(self) -> pd.DataFrame:
        """Reasons for items already adjudicated (they have left the queue)."""
        t1, t2 = self._tier(1), self._tier(2)
        rows = []
        done = set(zip(t2.item_id, t2.task))
        for (i, t), g in t1.groupby(["item_id", "task"]):
            if (i, t) not in done:
                continue
            auto = self._auto(i, t)
            if (g.verdict == "uncertain").any():
                rows.append((i, t, "uncertain"))
            elif not all(same(t, (r.corrected if r.verdict == "disagree" else auto), auto)
                         for r in g.itertuples()):
                rows.append((i, t, "disagreement"))
            else:
                rows.append((i, t, "spot-check"))
        return pd.DataFrame(rows, columns=["item_id", "task", "reason"])

    def report(self, min_cell: int = 11, spot_check_target: float = 0.95) -> pd.DataFrame:
        """Per task group: tier-1 agreement with the automatic labels, tier-2 workload and outcomes,
        and the spot-check agreement (estimated accuracy of tier-1-accepted labels) with its
        95% Wilson interval. Counts under `min_cell` are suppressed."""
        f = self.final_labels()
        t1 = self._tier(1)
        f["group"] = f.task.str.replace("phenotype:", "", regex=False)
        rows = []
        for gname, g in f.groupby("group"):
            keys = set(zip(g.item_id, g.task))
            t1g = t1[[k in keys for k in zip(t1.item_id, t1.task)]]
            n1 = len(t1g)
            agree1 = int((t1g.verdict == "agree").sum())
            adj = g[g.status == "cardiologist"]
            spot = adj[adj.reason == "spot-check"]
            spot_ok = sum(same(r.task, r.final_value, r.auto_value) for r in spot.itertuples())
            dis = adj[adj.reason == "disagreement"]
            auto_right = sum(same(r.task, r.final_value, r.auto_value) for r in dis.itertuples())
            lo, hi = wilson(spot_ok, len(spot))
            rows.append({
                "group": gname, "items": len(g),
                "tier1_reviews": n1, "tier1_agree": agree1,
                "tier1_agreement": round(agree1 / n1, 3) if n1 else None,
                "tier1_uncertain": int((t1g.verdict == "uncertain").sum()),
                "tier2_disagreements": int((g.reason == "disagreement").sum()),
                "tier2_uncertain": int((g.reason == "uncertain").sum()),
                "tier2_spot_checks": int((g.reason == "spot-check").sum()),
                "disagreements_auto_upheld": auto_right, "disagreements_adjudicated": len(dis),
                "spot_checks_done": len(spot), "spot_checks_agree": spot_ok,
                "spot_check_agreement": round(spot_ok / len(spot), 3) if len(spot) else None,
                "spot_ci_low": lo, "spot_ci_high": hi,
                "spot_check_pass": (bool(lo >= spot_check_target) if len(spot) and not math.isnan(lo) else None),
                "final": int(g.final_value.notna().sum()), "pending": int(g.final_value.isna().sum()),
            })
        out = pd.DataFrame(rows).astype(object)
        counts = [c for c in out.columns if c not in ("group", "tier1_agreement", "spot_check_agreement",
                                                      "spot_ci_low", "spot_ci_high", "spot_check_pass")]
        for c in counts:
            v = pd.to_numeric(out[c], errors="coerce")
            out.loc[(v > 0) & (v < min_cell), c] = f"<{min_cell}"
        small = pd.to_numeric(out.spot_checks_done, errors="coerce").fillna(0) < min_cell
        out.loc[small, ["spot_check_agreement", "spot_ci_low", "spot_ci_high"]] = "suppressed"
        small1 = pd.to_numeric(out.tier1_reviews, errors="coerce").fillna(0) < min_cell
        out.loc[small1, "tier1_agreement"] = "suppressed"
        return out


# ------------------------------------------------------------------ using the audited labels
def apply_to_echo_store(audit: LabelAudit, echo_store: AnnotationStore, annotator: str = "audit") -> int:
    """Write every final echo label back into the echo store as a human label, so
    `annotate.echo finalize` uses the audited views and frames."""
    f = audit.final_labels()
    f = f[(f.task == ECHO_TASK) & f.final_value.notna()]
    rows = []
    for r in f.itertuples():
        v = json.loads(r.final_value)
        rows.append({"item_id": r.item_id, "task": T_VIEW, "value": v["view"], "confidence": 1.0,
                     "source": "human", "annotator": annotator, "note": r.status})
        if v["view"] in VIEWS[:4]:
            for t, k in ((T_ED, "ed"), (T_ES, "es")):
                if v.get(k) is not None:
                    rows.append({"item_id": r.item_id, "task": t, "value": str(v[k]), "confidence": 1.0,
                                 "source": "human", "annotator": annotator, "note": r.status})
    return echo_store.add(rows)


def apply_to_labels(audit: LabelAudit, labels: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Replace automatic condition labels with the audited final labels for the women audited.
    Returns (new labels table, number of labels changed). Pending items are left as they were."""
    f = audit.final_labels()
    f = f[f.task.str.startswith("phenotype:") & f.final_value.notna()]
    out = labels.copy()
    pid = out.person_id.astype(str)
    changed = 0
    for r in f.itertuples():
        col = "y_" + r.task.split(":", 1)[1]
        new = 1.0 if str(r.final_value).lower() == "yes" else 0.0
        m = pid == r.item_id
        changed += int((out.loc[m, col] != new).sum())
        out.loc[m, col] = new
    if "y_SCAD" in out and "y_fmd_trigger" in out:          # FMD screening trigger follows SCAD
        out["y_fmd_trigger"] = out["y_SCAD"]
    return out, changed


def phenotype_validation(audit: LabelAudit, sample: pd.DataFrame, min_cell: int = 11) -> pd.DataFrame:
    """PPV / NPV of the automatic condition labels against the audited final labels (report 09c).
    Pending items are left out, like 'unsure' answers."""
    from .phenotype import validation_report
    f = audit.final_labels()
    f = f[f.task.str.startswith("phenotype:") & f.final_value.notna()]

    class _Finals:                                   # what validation_report reads from a store
        def latest(self, task):
            g = f[f.task == task]
            return pd.DataFrame({"item_id": g.item_id, "value": g.final_value.str.lower()})
    return validation_report(sample, _Finals(), min_cell)
