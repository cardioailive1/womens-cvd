"""Reviewer training on the SYNTHETIC kit: practice, qualification and scoring, on a laptop.

Reviewers practise before they get workspace access, on made-up records and cartoon echo clips
with known answers (annotate.synthetic_cases). No patient data is involved, so this runs anywhere
the repository runs (local JupyterLab), never inside the Mayo Clinic Platform workspace.

  1. build   — `python -m annotate.synthetic_cases --out training_kit` writes the kit: the cases,
               the answer key (each answer with a teaching note) and a mock audit set.
  2. check   — the clinical lead reads the key; any answer they relabel in
               `TrainingScreen(mode="reference")` overrides the generated one.
  3. learn   — trainees work the practice set in `TrainingScreen(mode="practice")`; after each
               answer the screen shows the key's answer and its teaching note.
  4. audit   — trainees rehearse the real two-tier audit on the mock audit set (the same
               LabelAudit and AuditScreen used in the workspace, with the automatic audit).
  5. qualify — trainees label half of the reference set (part A; part B is the retake) in
               `TrainingScreen(mode="qualification", attempt=1)`, with no feedback.
  6. score   — `qualification_report` scores each trainee against the pass marks in
               config/study.yaml (`training:`); `pair_kappa` checks two cardiologists agree.

Answers go to the kit's own store (training_kit/training.csv). Tasks are prefixed
ref: / practice: / qual1: / qual2: / refresh<month>:.

    from annotate.training import TrainingScreen, load_training_sets, qualification_report
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import ECHO_TASK, FRAME_TOLERANCE, echo_value, same
from .echo import VIEWS, cavity_signal, read_clip
from .phenotype import QUESTION
from .synthetic_cases import KEY_AUTHOR
from .store import AnnotationStore

MODES = ("reference", "practice", "qualification", "refresher")
PHENO_ANSWERS = ("yes", "no", "unsure")
DEFAULT_MARKS = {
    "data_scientist": {"echo_view": 18, "echo_frames": 17, "phenotype": 18, "phenotype_max_unsure": 2},
    "cardiologist": {"echo": 19, "phenotype": 19},
    "cardiologist_pair_kappa": 0.80,
    "qualification_size": 20,
}


# ------------------------------------------------------------------ sets
def save_training_sets(sets: pd.DataFrame, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sets.to_csv(path, index=False)


def load_training_sets(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"item_id": str, "part": str}).fillna({"part": ""})


# ------------------------------------------------------------------ answers
def _prefix(mode: str, attempt: int | None = None) -> str:
    if mode == "reference":
        return "ref:"
    if mode == "qualification":
        return f"qual{attempt}:"
    return f"{mode}:"


def answers(store: AnnotationStore, mode: str, attempt: int | None = None,
            annotator: str | None = None) -> pd.DataFrame:
    """Latest answer per (annotator, item, task) in one mode: item_id, task, annotator, value, note."""
    df = store.read()
    p = _prefix(mode, attempt)
    df = df[df.task.str.startswith(p, na=False)].copy()
    if annotator is not None:
        df = df[df.annotator == annotator]
    df["task"] = df.task.str.slice(len(p))
    return (df.sort_values("ts", kind="stable").groupby(["annotator", "item_id", "task"]).tail(1)
              [["item_id", "task", "annotator", "value", "note"]].reset_index(drop=True))


def reference_key(store: AnnotationStore, lead: str | None = None) -> dict:
    """(item_id, task) -> (value, note): the kit's generated answer, overridden by any answer the
    clinical lead (`lead`) gave the same item in mode="reference"."""
    ref = answers(store, "reference")
    ref = ref[ref.annotator.isin([KEY_AUTHOR] + ([lead] if lead else []))]
    ref = ref.assign(_lead=(ref.annotator != KEY_AUTHOR).astype(int)).sort_values("_lead", kind="stable")
    return {(r.item_id, r.task): (r.value, r.note) for r in ref.groupby(["item_id", "task"]).tail(1).itertuples()}


def qualification_items(sets: pd.DataFrame, attempt: int) -> pd.DataFrame:
    """Attempt 1 = reference part A, attempt 2 (the retake) = part B."""
    part = {1: "A", 2: "B"}.get(attempt)
    if part is None:
        raise ValueError("attempt is 1 (first try) or 2 (retake)")
    return sets[(sets.set == "reference") & (sets.part == part)].reset_index(drop=True)


def refresher_items(sets: pd.DataFrame, month: int, n: int = 10) -> pd.DataFrame:
    """A different 10 reference items each month (half echo, half records where both exist)."""
    ref = sets[sets.set == "reference"]
    picks = []
    for is_echo in (True, False):
        g = ref[(ref.task == ECHO_TASK) == is_echo]
        if len(g):
            picks.append(g.sample(min(len(g), math.ceil(n / 2)), random_state=1000 + month))
    return pd.concat(picks).head(n).reset_index(drop=True)


# ------------------------------------------------------------------ the screen
class TrainingScreen:
    """Blinded labelling screen for every training mode. Logic lives in current / submit /
    skip / feedback; show() wires them to widgets.

    reference      the clinical lead checks the key; their answer overrides the generated one
    practice       trainees; after each answer `feedback()` returns the reference answer + note
    qualification  trainees, attempt 1 (part A) or 2 (part B); no feedback
    refresher      trainees, a monthly 10 (pass `items=refresher_items(sets, month)`), no feedback
    """

    def __init__(self, sets: pd.DataFrame, store, annotator: str, mode: str, attempt: int | None = None,
                 items: pd.DataFrame | None = None, phenotype_context=None, echo_cache=None,
                 lead: str | None = None, month: int | None = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.store = store if isinstance(store, AnnotationStore) else AnnotationStore(store)
        self.mode, self.annotator, self.lead = mode, annotator, lead
        self.attempt = attempt if mode == "qualification" else (month if mode == "refresher" else None)
        if mode == "reference":
            pool = sets[sets.set != "audit"]
        elif mode == "practice":
            pool = sets[sets.set == "practice"]
        elif mode == "qualification":
            pool = qualification_items(sets, attempt)
        else:
            if items is None or month is None:
                raise ValueError("refresher needs items=refresher_items(sets, month) and month")
            pool = items
        if mode != "reference":
            key = reference_key(self.store, lead)
            missing = [(i, t) for i, t in zip(pool.item_id, pool.task) if (i, t) not in key]
            if missing:
                raise ValueError(f"{len(missing)} items have no answer in the key; "
                                 "rebuild the kit with annotate.synthetic_cases")
        if isinstance(phenotype_context, (str, Path)):
            phenotype_context = pd.read_pickle(phenotype_context)
        self.ctx = phenotype_context
        self.paths = {}
        if echo_cache is not None:
            self.paths = pd.read_csv(Path(echo_cache) / "index.csv").set_index("item_id").path.to_dict()
        self.prefix = (f"refresh{month}:" if mode == "refresher" else _prefix(mode, attempt))
        df = self.store.read()
        mine = df[df.task.str.startswith(self.prefix, na=False) & (df.annotator == annotator)]
        done = set(zip(mine.item_id.astype(str), mine.task.str.slice(len(self.prefix))))
        self.queue = pool[[(i, t) not in done for i, t in zip(pool.item_id, pool.task)]].reset_index(drop=True)
        self.pos, self._clip, self._last = 0, None, None

    def remaining(self) -> int:
        return max(0, len(self.queue) - self.pos)

    def current(self) -> dict | None:
        if self.pos >= len(self.queue):
            return None
        r = self.queue.iloc[self.pos]
        item = {"item_id": r.item_id, "task": r.task, "kind": "echo" if r.task == ECHO_TASK else "phenotype"}
        if item["kind"] == "echo":
            if self._clip is None or self._clip[0] != r.item_id:
                frames, _ = read_clip(self.paths[r.item_id])
                self._clip = (r.item_id, frames, cavity_signal(frames))
            item["frames"], item["signal"] = self._clip[1], self._clip[2]
        else:
            cond = r.task.split(":", 1)[1]
            pid = int(r.item_id)
            pick = lambda k: self.ctx[k][self.ctx[k].person_id == pid].sort_values("rel_day")
            item.update({"person_id": pid, "condition": cond, "question": QUESTION.get(cond, cond),
                         **{k: pick(k) for k in ("diagnoses", "procedures", "labs", "notes")}})
        return item

    def submit(self, *, value=None, view=None, ed=None, es=None, note: str | None = None) -> None:
        """Conditions/outcomes: value = yes | no | unsure. Echo: view (+ ed, es), view may be UNUSABLE."""
        c = self.current()
        if c is None:
            return
        if c["kind"] == "echo":
            if view is None:
                raise ValueError("choose a view")
            if view in VIEWS[:4]:
                n = len(c["frames"])
                for v in (ed, es):
                    if v is None or not 0 <= int(v) < n:
                        raise ValueError(f"mark ED and ES inside the clip (0-{n - 1})")
                if int(es) <= int(ed):
                    raise ValueError("ES must come after ED")
                value = echo_value(view, ed, es)
            else:
                value = echo_value(view)
        elif value not in PHENO_ANSWERS:
            raise ValueError(f"answer one of {PHENO_ANSWERS}")
        self.store.add_human(c["item_id"], self.prefix + c["task"], value, self.annotator, note)
        self._last = (c["item_id"], c["task"], value)
        self.pos += 1

    def skip(self) -> None:
        self.pos += 1

    def feedback(self) -> dict | None:
        """Practice mode only: the last answer against the reference answer and the lead's note."""
        if self.mode != "practice" or self._last is None:
            return None
        i, t, v = self._last
        ref, note = reference_key(self.store, self.lead)[(i, t)]
        return {"item_id": i, "task": t, "yours": v, "reference": ref, "lead_note": note,
                "match": _match(t, v, ref)}

    # ---- widgets
    def show(self):
        import ipywidgets as W
        from IPython.display import display

        from .review_ui import HUMAN_VIEWS, PhenotypeReviewer, _png, _signal_png
        head, body, msg, fb = W.HTML(), W.HTML(), W.HTML(), W.HTML()
        img, sig = W.Image(format="png"), W.Image(format="png")
        frame = W.IntSlider(description="frame", min=0, max=1)
        view = W.ToggleButtons(options=HUMAN_VIEWS, description="view")
        ed, es = W.IntText(description="ED frame"), W.IntText(description="ES frame")
        set_ed, set_es = W.Button(description="ED = this frame"), W.Button(description="ES = this frame")
        note = W.Text(description="note", placeholder="say what you saw")
        b = {v: W.Button(description=v.capitalize(), button_style={"yes": "success", "no": "danger"}.get(v, ""))
             for v in PHENO_ANSWERS}
        b_save, b_skip = W.Button(description="Save and next", button_style="primary"), W.Button(description="Skip")
        box, st = W.VBox(), {}
        titles = {"reference": "Answer key check (clinical lead)", "practice": "Practice: feedback after each answer",
                  "qualification": f"Qualification, attempt {self.attempt}: no feedback",
                  "refresher": "Monthly refresher: no feedback"}

        def draw(*_):
            if st.get("kind") != "echo":
                return
            f = st["frames"]; i = min(frame.value, len(f) - 1)
            img.value = _png(f[i], 1 if f.shape[1] >= 300 else 2)
            sig.value = _signal_png(st["signal"], ed.value, es.value, i)

        def show_feedback():
            f = self.feedback()
            if not f:
                fb.value = ""; return
            colour = "#166534" if f["match"] else "#b91c1c"
            fb.value = (f"<div style='border-left:4px solid {colour};padding:4px 8px'>"
                        f"<b>{'Matches' if f['match'] else 'Differs from'} the reference.</b> "
                        f"Yours: {html.escape(str(f['yours']))} · reference: {html.escape(str(f['reference']))}"
                        f"<br>{html.escape(str(f['lead_note'] or ''))}</div>")

        def load(*_):
            c = self.current(); st.clear(); msg.value = ""; note.value = ""
            if c is None:
                head.value = "<b>Done.</b>"; box.children = [head, fb]; return
            st.update(c)
            head.value = f"<b>{self.remaining()} left</b> · {titles[self.mode]}"
            if c["kind"] == "phenotype":
                body.value = PhenotypeReviewer.render(c)
                box.children = [fb, head, body, note, W.HBox(list(b.values()) + [b_skip]), msg]
            else:
                body.value = "<p>Choose the view and mark ED and ES on the first cycle.</p>"
                view.value = None; ed.value = es.value = 0
                frame.max = len(c["frames"]) - 1; frame.value = 0; draw()
                box.children = [fb, head, body, img, frame, sig, view, W.HBox([ed, set_ed]),
                                W.HBox([es, set_es]), note, W.HBox([b_save, b_skip]), msg]

        def act(**kw):
            try:
                self.submit(note=note.value or None, **kw); show_feedback(); load()
            except ValueError as e:
                msg.value = f"<span style='color:#b91c1c'>{html.escape(str(e))}</span>"

        for v, btn in b.items():
            btn.on_click(lambda _, v=v: act(value=v))
        b_save.on_click(lambda _: act(view=view.value, ed=ed.value, es=es.value))
        b_skip.on_click(lambda _: (self.skip(), load()))
        frame.observe(draw, "value"); ed.observe(draw, "value"); es.observe(draw, "value")
        set_ed.on_click(lambda _: setattr(ed, "value", frame.value))
        set_es.on_click(lambda _: setattr(es, "value", frame.value))
        load()
        display(box)
        return box


# ------------------------------------------------------------------ 5. score
def _match(task: str, a, b) -> bool:
    return same(task, a, b)


def _view_and_frames(a: str, b: str) -> tuple[bool, bool | None]:
    x, y = json.loads(a), json.loads(b)
    view_ok = x.get("view") == y.get("view")
    if y.get("view") not in VIEWS[:4] or not view_ok:
        return view_ok, None                     # frames only scored where both name the same usable view
    ok = all(x.get(k) is not None and y.get(k) is not None and abs(x[k] - y[k]) <= FRAME_TOLERANCE
             for k in ("ed", "es"))
    return view_ok, ok


def score(store: AnnotationStore, sets: pd.DataFrame, trainee: str, role: str, attempt: int = 1,
          marks: dict | None = None, lead: str | None = None) -> dict:
    """One trainee's qualification attempt against the reference answers and the pass marks."""
    m = {**DEFAULT_MARKS, **(marks or {})}
    if role not in ("data_scientist", "cardiologist"):
        raise ValueError("role is data_scientist or cardiologist")
    items = qualification_items(sets, attempt)
    key = reference_key(store, lead)
    got = answers(store, "qualification", attempt, trainee).set_index(["item_id", "task"]).value.to_dict()
    echo = items[items.task == ECHO_TASK]; ph = items[items.task != ECHO_TASK]
    out = {"trainee": trainee, "role": role, "attempt": attempt,
           "answered": sum((i, t) in got for i, t in zip(items.item_id, items.task)), "items": len(items)}
    ev = [_view_and_frames(got[(i, t)], key[(i, t)][0]) for i, t in zip(echo.item_id, echo.task) if (i, t) in got]
    out["echo_items"] = len(echo)
    out["echo_view_correct"] = sum(v for v, _ in ev)
    out["echo_frames_correct"] = sum(1 for _, f in ev if f) + sum(1 for v, f in ev if v and f is None)
    out["echo_both_correct"] = sum(1 for v, f in ev if v and f is not False)
    pv = [(got[(i, t)], key[(i, t)][0]) for i, t in zip(ph.item_id, ph.task) if (i, t) in got]
    out["records_items"] = len(ph)
    out["records_agree"] = sum(a == b for a, b in pv)
    out["records_unsure"] = sum(a == "unsure" and b != "unsure" for a, b in pv)   # Unsure where the key is definite
    full = lambda n, k: n * len_scale(k, m["qualification_size"])   # marks are written out of 20
    if role == "data_scientist":
        r = m["data_scientist"]
        checks = {"echo_view": out["echo_view_correct"] >= full(r["echo_view"], len(echo)),
                  "echo_frames": out["echo_frames_correct"] >= full(r["echo_frames"], len(echo)),
                  "records": out["records_agree"] >= full(r["phenotype"], len(ph)),
                  "records_unsure": out["records_unsure"] <= r["phenotype_max_unsure"]}
    else:
        r = m["cardiologist"]
        checks = {"echo": out["echo_both_correct"] >= full(r["echo"], len(echo)),
                  "records": out["records_agree"] >= full(r["phenotype"], len(ph))}
    out.update({f"pass_{k}": bool(v) for k, v in checks.items()})
    out["complete"] = out["answered"] == out["items"]
    out["passed"] = out["complete"] and all(checks.values())
    return out


def len_scale(n_items: int, size: int) -> float:
    """Pass marks are written out of `size` (20); scale them when a half has a different size."""
    return n_items / size if size else 1.0


def qualification_report(store: AnnotationStore, sets: pd.DataFrame, trainees: dict[str, str],
                         attempt: int = 1, marks: dict | None = None, lead: str | None = None) -> pd.DataFrame:
    """One row per trainee: counts, each pass check and the overall result. Contains reviewer
    names and scores only, no patient data."""
    return pd.DataFrame([score(store, sets, t, role, attempt, marks, lead) for t, role in trainees.items()])


def pair_kappa(store: AnnotationStore, sets: pd.DataFrame, a: str, b: str, attempt: int = 1) -> dict:
    """Cohen's kappa between two cardiologists over the qualification items both answered
    (records: yes / no / unsure; echo: the view)."""
    from sklearn.metrics import cohen_kappa_score
    items = qualification_items(sets, attempt)
    A = answers(store, "qualification", attempt, a).set_index(["item_id", "task"]).value
    B = answers(store, "qualification", attempt, b).set_index(["item_id", "task"]).value
    both = [k for k in zip(items.item_id, items.task) if k in A.index and k in B.index]
    lab = lambda v, t: json.loads(v)["view"] if t == ECHO_TASK else v
    x = [lab(A[k], k[1]) for k in both]; y = [lab(B[k], k[1]) for k in both]
    k = cohen_kappa_score(x, y) if len(set(x) | set(y)) > 1 else 1.0
    return {"a": a, "b": b, "items": len(both), "kappa": round(float(k), 3)}


def marks_from_config(cfg) -> dict:
    t = getattr(cfg, "training", None) or {}
    m = {**DEFAULT_MARKS}
    for k in ("data_scientist", "cardiologist"):
        m[k] = {**DEFAULT_MARKS[k], **(t.get("pass_marks", {}).get(k) or {})}
    m["cardiologist_pair_kappa"] = float(t.get("pass_marks", {}).get("cardiologist_pair_kappa",
                                                                       DEFAULT_MARKS["cardiologist_pair_kappa"]))
    m["qualification_size"] = int(t.get("qualification_size", DEFAULT_MARKS["qualification_size"]))
    return m


def refresher_report(store: AnnotationStore, sets: pd.DataFrame, month: int, trainees: list[str],
                     lead: str | None = None) -> pd.DataFrame:
    """Per reviewer: how many of the month's refresher items match the reference answer
    (echo: same view and frames within tolerance; records: same answer)."""
    key = reference_key(store, lead)
    items = refresher_items(sets, month)
    df = store.read()
    p = f"refresh{month}:"
    df = df[df.task.str.startswith(p, na=False)].copy()
    df["task"] = df.task.str.slice(len(p))
    df = df.sort_values("ts", kind="stable").groupby(["annotator", "item_id", "task"]).tail(1)
    rows = []
    for t in trainees:
        mine = df[df.annotator == t].set_index(["item_id", "task"]).value
        done = [k for k in zip(items.item_id, items.task) if k in mine.index]
        ok = sum(_match(k[1], mine[k], key[k][0]) for k in done)
        rows.append({"reviewer": t, "month": month, "items": len(items), "answered": len(done),
                     "match_reference": ok, "agreement": round(ok / len(done), 3) if done else None})
    return pd.DataFrame(rows)
