"""Audit of the automatic labels (echo annotations, condition and outcome labels), in this order:

  Automatic audit — REQUIRED, right after automatic labelling (annotate.auto_audit, then
           `submit_auto`). Every item gets agree / disagree / uncertain from independent evidence.
           Neither tier opens until every item in the audit set has a verdict.
  Tier 1 — data scientists. They see the automatic label and the automatic audit's finding and
           answer agree / disagree (with the corrected value) / uncertain. Optionally more than
           one data scientist per item (`tier1_per_item`). Which items reach them depends on
           `auto_tier1` (below).
  Tier 2 — a cardiologist adjudicates, blinded to both the automatic and the data-scientist label:
             * every disagreement (data scientist vs automatic label, or between data scientists)
             * every item a data scientist marked uncertain
             * a blinded spot-check: a random share of the items tier 1 accepted, which measures
               how many wrong labels slip through tier 1 (automation bias included).
           The three kinds are shuffled together so the cardiologist cannot tell which is which.

How the automatic verdicts route items (`auto_tier1`):
  triage   — default. Items the auditor agrees with are tier-1 accepted without a data scientist;
             the ones it disagrees with or is unsure about go to the data scientists, whose
             answer decides.
  replace  — the auditor is tier 1; its disagreements and uncertain items go straight to tier 2.
The spot-check samples auto-accepted and human-accepted items separately
(`auto_spot_check_fraction` / `spot_check_fraction`, each at least `spot_check_min`), so the
report shows how many wrong labels each route lets through.

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
AUTO_MODES = ("triage", "replace")             # the automatic audit is required; there is no "off"
AUTO_AUDITOR, AUTO_SOURCE = "auto-audit", "auto:audit"
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
    auto_tier1: str = "triage"                # triage | replace (the automatic audit is required)
    auto_spot_check_fraction: float = 0.20    # of auto-accepted items
    _key: pd.Index = field(init=False, repr=False)

    def __post_init__(self):
        if isinstance(self.store, str):
            self.store = AnnotationStore(self.store)
        overlap = set(self.data_scientists) & set(self.cardiologists)
        if overlap:
            raise ValueError(f"{sorted(overlap)} listed as both data scientist and cardiologist")
        if self.auto_tier1 not in AUTO_MODES:
            raise ValueError(f"the automatic audit is required: auto_tier1 must be one of {AUTO_MODES}")
        if AUTO_AUDITOR in set(self.data_scientists) | set(self.cardiologists):
            raise ValueError(f"{AUTO_AUDITOR!r} is reserved for the automatic auditor")
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
                   tier1_per_item=int(a.get("tier1_per_item", 1)), seed=cfg.seed,
                   auto_tier1=str(a.get("auto_tier1", "triage") or "triage"),
                   auto_spot_check_fraction=float(a.get("auto_spot_check_fraction", 0.20)))

    # ---- raw reads
    def _tier(self, tier: int, auto: bool = False) -> pd.DataFrame:
        """Tier answers from people; with auto=True, tier-1 rows of the automatic auditor too."""
        df = self.store.read()
        ok = (df.source == "human") | ((df.source == AUTO_SOURCE) & auto & (tier == 1))
        df = df[df.task.str.startswith(f"audit{tier}:", na=False) & ok].copy()
        df["task"] = df.task.str.slice(len(f"audit{tier}:"))
        if tier == 1:
            parsed = df.value.map(json.loads)
            df["verdict"] = parsed.map(lambda v: v["verdict"])
            df["corrected"] = parsed.map(lambda v: v.get("value"))
        # one answer per reviewer per item: their latest
        return df.sort_values("ts", kind="stable").groupby(["item_id", "task", "annotator"]).tail(1)

    # ---- automatic tier 1
    def submit_auto(self, verdicts: pd.DataFrame) -> int:
        """Store the automatic auditor's verdicts (annotate.auto_audit). Re-running replaces
        earlier automatic verdicts (the latest counts); people's answers are never touched."""
        known = set(zip(self.items.item_id, self.items.task))
        rows = []
        for r in verdicts.itertuples():
            if (str(r.item_id), r.task) not in known or r.verdict not in VERDICTS:
                continue
            val = r.value if r.verdict == "disagree" else (self._auto(r.item_id, r.task) if r.verdict == "agree" else None)
            if r.verdict == "disagree" and (val is None or (isinstance(val, float) and pd.isna(val))
                                            or same(r.task, val, self._auto(r.item_id, r.task))):
                continue                                     # a disagreement without a correction is not usable
            rows.append({"item_id": str(r.item_id), "task": f"audit1:{r.task}",
                         "value": json.dumps({"verdict": r.verdict, "value": val}),
                         "confidence": getattr(r, "confidence", None), "source": AUTO_SOURCE,
                         "annotator": AUTO_AUDITOR, "note": getattr(r, "note", None)})
        return self.store.add(rows)

    def _t1_effective(self) -> dict:
        """(item_id, task) -> (rows that make up tier 1, complete?, 'auto' | 'human')."""
        t1 = self._tier(1, auto=True)
        out = {}
        for key, g in t1.groupby(["item_id", "task"]):
            hum, aut = g[g.annotator != AUTO_AUDITOR], g[g.annotator == AUTO_AUDITOR]
            if self.auto_tier1 == "replace" and len(aut):
                rows, done, by = (aut if hum.empty else g), True, "auto" if hum.empty else "human"
            elif self.auto_tier1 == "triage" and len(aut) and hum.empty and (aut.verdict == "agree").all():
                rows, done, by = aut, True, "auto"
            else:                                            # people decide (auto verdict is a hint)
                rows, done, by = hum, len(hum) >= self.tier1_per_item, "human"
            out[key] = (rows, done, by)
        return out

    def _t1_state(self) -> pd.DataFrame:
        """item_id, task, outcome (accepted | disagreement | uncertain), by (auto | human) for every
        item whose tier 1 is complete."""
        rows = []
        for (i, t), (g, done, by) in self._t1_effective().items():
            if not done or g.empty:
                continue
            auto = self._auto(i, t)
            labels = [r.corrected if r.verdict == "disagree" else auto for r in g.itertuples() if r.verdict != "uncertain"]
            if (g.verdict == "uncertain").any():
                o = "uncertain"
            elif not all(same(t, x, auto) for x in labels):
                o = "disagreement"
            else:
                o = "accepted"
            rows.append((i, t, o, by))
        return pd.DataFrame(rows, columns=["item_id", "task", "outcome", "by"])

    def auto_verdicts(self) -> pd.DataFrame:
        """Latest automatic verdict per item (item_id, task, verdict, corrected, note)."""
        t = self._tier(1, auto=True)
        return t[t.annotator == AUTO_AUDITOR][["item_id", "task", "verdict", "corrected", "confidence", "note"]]

    def automatic_audit_missing(self) -> pd.DataFrame:
        """Items in the audit set that the automatic audit has not given a verdict yet."""
        av = self.auto_verdicts()
        done = set(zip(av.item_id, av.task))
        return self.items[[(i, t) not in done for i, t in zip(self.items.item_id, self.items.task)]]

    def _require_automatic_audit(self) -> None:
        missing = self.automatic_audit_missing()
        if len(missing):
            raise RuntimeError(
                f"{len(missing)} of {len(self.items)} items have no automatic-audit verdict. The automatic "
                "audit is required before the data-scientist and cardiologist audits: run "
                "annotate.auto_audit (audit_phenotypes / audit_echo) and audit.submit_auto(...) first.")

    # ---- tier 1
    def tier1_queue(self, annotator: str) -> pd.DataFrame:
        """Items this data scientist has not audited that still need tier-1 reviews."""
        self._require_automatic_audit()
        t1 = self._tier(1)
        mine = set(zip(t1[t1.annotator == annotator].item_id, t1[t1.annotator == annotator].task))
        eff = self._t1_effective()
        need = [(i, t) not in mine and not eff.get((i, t), (None, False))[1]
                for i, t in zip(self.items.item_id, self.items.task)]
        q = self.items[need]
        flagged = set()
        if self.auto_tier1 == "triage":                     # items the auditor flagged come first
            av = self.auto_verdicts()
            flagged = set(zip(av[av.verdict != "agree"].item_id, av[av.verdict != "agree"].task))
        return q.assign(_f=[(i, t) not in flagged for i, t in zip(q.item_id, q.task)],
                        _h=[_hash(f"{annotator}|{i}|{t}", self.seed) for i, t in zip(q.item_id, q.task)]
                        ).sort_values(["_f", "_h"]).drop(columns=["_f", "_h"]).reset_index(drop=True)

    def submit_tier1(self, item_id, task: str, verdict: str, annotator: str,
                     value=None, note: str | None = None) -> None:
        if annotator not in self.data_scientists:
            raise PermissionError(f"{annotator} is not a listed data scientist (tier 1)")
        self._require_automatic_audit()
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
        self._require_automatic_audit()
        t1 = self._tier(1)
        st = self._t1_state()
        rows = [(r.item_id, r.task, r.outcome) for r in st.itertuples() if r.outcome != "accepted"]
        # blinded spot-check over tier-1-accepted items, sampled separately for items accepted by
        # people and by the automatic auditor: hash threshold, topped up to the minimum
        for by, frac in (("human", self.spot_check_fraction), ("auto", self.auto_spot_check_fraction)):
            acc = st[(st.outcome == "accepted") & (st.by == by)]
            acc = sorted(zip(acc.item_id, acc.task), key=lambda k: _hash(f"spot|{k[0]}|{k[1]}", self.seed))
            k = min(len(acc), max(self.spot_check_min, math.ceil(frac * len(acc))))
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
        self._require_automatic_audit()
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
        """item_id, task, auto_value, final_value, status, reason, tier1_by (auto | human)."""
        t2 = self._tier(2)
        t2v = t2.set_index(["item_id", "task"]).value if len(t2) else pd.Series(dtype=str)
        q = self.tier2_queue()
        all_t2 = pd.concat([q, self._tier2_reasons_done()], ignore_index=True)
        reason = {(r.item_id, r.task): r.reason for r in all_t2.itertuples()}
        eff = self._t1_effective()
        out = []
        for r in self.items.itertuples():
            key = (r.item_id, r.task)
            why = reason.get(key, "")
            if key in t2v.index:
                v = t2v[key]
                status = "pending" if str(v).lower() == "unsure" else "cardiologist"
                final = None if status == "pending" else v
            elif not eff.get(key, (None, False))[1]:
                final, status = None, "awaiting tier 1"
            elif why in ("disagreement", "uncertain"):
                final, status = None, "awaiting tier 2"
            else:
                final, status = r.auto_value, "tier 1 confirmed"
            out.append({"item_id": r.item_id, "task": r.task, "auto_value": r.auto_value,
                        "final_value": final, "status": status, "reason": why,
                        "tier1_by": eff.get(key, (None, False, "human"))[2]})
        return pd.DataFrame(out)

    def _tier2_reasons_done(self) -> pd.DataFrame:
        """Reasons for items already adjudicated (they have left the queue)."""
        t2 = self._tier(2)
        done = set(zip(t2.item_id, t2.task))
        st = self._t1_state()
        st = st[[k in done for k in zip(st.item_id, st.task)]]
        return pd.DataFrame({"item_id": st.item_id, "task": st.task,
                             "reason": st.outcome.replace({"accepted": "spot-check"})})

    def report(self, min_cell: int = 11, spot_check_target: float = 0.95) -> pd.DataFrame:
        """Per task group and tier-1 route (tier1_by: human | auto): tier-1 agreement with the
        automatic labels, tier-2 workload and outcomes, and the spot-check agreement (estimated
        accuracy of tier-1-accepted labels) with its 95% Wilson interval. With automatic tier 1 on,
        the `auto` row's spot-check is what shows whether auto-accepted labels can be trusted.
        Counts under `min_cell` are suppressed."""
        f = self.final_labels()
        eff = self._t1_effective()
        t1 = pd.concat([g for g, _, _ in eff.values()] or [self._tier(1)], ignore_index=True)
        f["group"] = f.task.str.replace("phenotype:", "", regex=False)
        rows = []
        for (gname, by), g in f.groupby(["group", "tier1_by"]):
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
                "group": gname, "tier1_by": by, "items": len(g),
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
        counts = [c for c in out.columns if c not in ("group", "tier1_by", "tier1_agreement", "spot_check_agreement",
                                                      "spot_ci_low", "spot_ci_high", "spot_check_pass")]
        for c in counts:
            v = pd.to_numeric(out[c], errors="coerce")
            out.loc[(v > 0) & (v < min_cell), c] = f"<{min_cell}"
        small = pd.to_numeric(out.spot_checks_done, errors="coerce").fillna(0) < min_cell
        out.loc[small, ["spot_check_agreement", "spot_ci_low", "spot_ci_high"]] = "suppressed"
        small1 = pd.to_numeric(out.tier1_reviews, errors="coerce").fillna(0) < min_cell
        out.loc[small1, "tier1_agreement"] = "suppressed"
        return out


    def auto_calibration(self, min_cell: int = 11) -> pd.DataFrame:
        """How the automatic auditor's verdicts compare with people, per task group and verdict:
        on items a data scientist also audited, how often they gave the same verdict; on items
        the cardiologist labelled, how often the automatic label turned out right. Use it to
        decide whether to move from triage to replace (and to spot a verdict class that is off)."""
        av = self.auto_verdicts()
        if av.empty:
            return pd.DataFrame(columns=["group", "auto_verdict", "items", "ds_reviewed", "ds_same_verdict",
                                         "cardiologist_labelled", "auto_label_right"])
        hum = self._tier(1)
        t2 = self._tier(2)
        t2v = t2.set_index(["item_id", "task"]).value.to_dict() if len(t2) else {}
        rows = []
        av = av.assign(group=av.task.str.replace("phenotype:", "", regex=False))
        for (gname, verdict), g in av.groupby(["group", "verdict"]):
            ds_n = ds_same = c_n = c_right = 0
            for r in g.itertuples():
                h = hum[(hum.item_id == r.item_id) & (hum.task == r.task)]
                if len(h):
                    ds_n += 1
                    ds_same += int((h.verdict == verdict).all())
                v = t2v.get((r.item_id, r.task))
                if v is not None and str(v).lower() != "unsure":
                    c_n += 1
                    c_right += int(same(r.task, v, self._auto(r.item_id, r.task)))
            rows.append({"group": gname, "auto_verdict": verdict, "items": len(g), "ds_reviewed": ds_n,
                         "ds_same_verdict": ds_same, "cardiologist_labelled": c_n, "auto_label_right": c_right})
        out = pd.DataFrame(rows).astype(object)
        for c in out.columns[2:]:
            v = pd.to_numeric(out[c], errors="coerce")
            out.loc[(v > 0) & (v < min_cell), c] = f"<{min_cell}"
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
    """Replace automatic condition and outcome labels with the audited final labels for the women
    audited. Returns (new labels table, number of labels changed). Pending items are left as they
    were. A reviewed outcome sets its mask to 1. Risk tier is not recomputed (see RUNBOOK 4b)."""
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
        mask = "m_" + col[2:]
        if mask in out and mask != "m_condition":          # a reviewed outcome is observed
            out.loc[m, mask] = 1.0
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
