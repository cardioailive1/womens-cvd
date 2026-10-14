"""In-workspace review tool (JupyterLab, ipywidgets). No server, no network, no extra install
beyond ipywidgets, so it runs in the MCP workbench where heavier tools (CVAT, OHIF, 3D Slicer /
MONAI Label) may not be allowed. Every decision is written to the append-only AnnotationStore.

Echo clips (views and cardiac phase), in a notebook cell:

    from annotate.review_ui import EchoReviewer
    EchoReviewer(store="$WB/annot/echo.csv", cache="$WB/annot/cache", annotator="dr_smith").show()

Condition labels (chart review, blinded to the automatic label):

    from annotate.review_ui import PhenotypeReviewer
    PhenotypeReviewer(sample="$WB/annot/phenotype_sample.csv", context="$WB/annot/context.pkl",
                      store="$WB/annot/phenotype.csv", annotator="dr_smith").show()

The classes keep all logic in plain methods (current, submit, skip) so they are testable without
a browser; `show()` only wires those methods to widgets.
"""
from __future__ import annotations

import html
import io
from pathlib import Path

import numpy as np
import pandas as pd

from .echo import T_ED, T_ES, T_VIEW, VIEWS, cavity_signal, read_clip, review_queue
from .phenotype import QUESTION, VERDICTS, load_context, task_name
from .store import AnnotationStore

HUMAN_VIEWS = VIEWS + ["UNUSABLE"]


def _png(arr: np.ndarray, scale: int = 1) -> bytes:
    from PIL import Image
    im = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale))
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return buf.getvalue()


def _signal_png(sig: np.ndarray, ed: int | None, es: int | None, cur: int, w: int = 480, h: int = 90) -> bytes:
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (w, h), "white"); dr = ImageDraw.Draw(im)
    if len(sig) > 1 and np.ptp(sig) > 0:
        x = np.linspace(8, w - 8, len(sig)); y = h - 10 - (sig - sig.min()) / np.ptp(sig) * (h - 20)
        dr.line(list(zip(x, y)), fill=(60, 60, 60), width=2)
        for idx, col in ((ed, (37, 99, 235)), (es, (220, 38, 38)), (cur, (150, 150, 150))):
            if idx is not None and 0 <= idx < len(sig):
                dr.line([(x[idx], 4), (x[idx], h - 4)], fill=col, width=1 if col[0] == 150 else 2)
    dr.text((8, 2), "cavity signal  (blue = ED, red = ES, grey = shown frame)", fill=(90, 90, 90))
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return buf.getvalue()


class EchoReviewer:
    def __init__(self, store, cache, annotator: str, threshold: float = 0.9, qa_fraction: float = 0.05):
        self.store = store if isinstance(store, AnnotationStore) else AnnotationStore(store)
        self.cache = Path(cache)
        self.annotator = annotator
        self.paths = pd.read_csv(self.cache / "index.csv").set_index("item_id").path.to_dict()
        self.queue = review_queue(self.store, threshold, qa_fraction)
        self.pos = 0
        self._clip = None

    # ---- logic
    def remaining(self) -> int:
        return max(0, len(self.queue) - self.pos)

    def current(self) -> dict | None:
        if self.pos >= len(self.queue):
            return None
        q = self.queue.iloc[self.pos]
        lat = self.store.latest()
        lat = lat[lat.item_id == q.item_id].set_index("task")
        get = lambda t: lat.value.get(t) if t in lat.index else None
        toint = lambda v: int(float(v)) if v not in (None, "") and not pd.isna(v) else None
        if self._clip is None or self._clip[0] != q.item_id:
            frames, _ = read_clip(self.paths[q.item_id])
            self._clip = (q.item_id, frames, cavity_signal(frames))
        return {"item_id": q.item_id, "reason": q.reason, "min_conf": float(q.min_conf),
                "view": get(T_VIEW) or None, "ed": toint(get(T_ED)), "es": toint(get(T_ES)),
                "frames": self._clip[1], "signal": self._clip[2]}

    def submit(self, view: str, ed: int | None = None, es: int | None = None, note: str | None = None) -> None:
        if view not in HUMAN_VIEWS:
            raise ValueError(f"view must be one of {HUMAN_VIEWS}")
        cur = self.current()
        if cur is None:
            return
        self.store.add_human(cur["item_id"], T_VIEW, view, self.annotator, note)
        if view in VIEWS[:4]:
            n = len(cur["frames"])
            for t, v in ((T_ED, ed), (T_ES, es)):
                if v is not None:
                    if not 0 <= int(v) < n:
                        raise ValueError(f"frame {v} outside clip (0-{n - 1})")
                    self.store.add_human(cur["item_id"], t, int(v), self.annotator)
        self.pos += 1

    def skip(self) -> None:
        self.pos += 1

    # ---- widgets
    def show(self):
        import ipywidgets as W
        from IPython.display import display

        head = W.HTML(); img = W.Image(format="png"); sig = W.Image(format="png")
        frame = W.IntSlider(description="frame", min=0, max=1, continuous_update=True)
        view = W.ToggleButtons(options=HUMAN_VIEWS, description="view")
        ed = W.IntText(description="ED frame"); es = W.IntText(description="ES frame")
        set_ed = W.Button(description="ED = this frame"); set_es = W.Button(description="ES = this frame")
        note = W.Text(description="note", placeholder="optional")
        save = W.Button(description="Save and next", button_style="primary")
        skip = W.Button(description="Skip"); msg = W.HTML()
        st = {}

        def draw_frame(*_):
            if not st: return
            f = st["frames"]; i = min(frame.value, len(f) - 1)
            img.value = _png(f[i], 1 if f.shape[1] >= 300 else 2)
            sig.value = _signal_png(st["signal"], ed.value, es.value, i)

        def load(*_):
            c = self.current()
            if c is None:
                head.value = "<b>Queue empty.</b> Re-run <code>annotate.echo auto</code> or <code>train-views</code> to refresh it."
                return
            st.clear(); st.update(c)
            head.value = (f"<b>{self.remaining()} clips left</b> · item <code>{c['item_id']}</code> · "
                          f"{html.escape(c['reason'])} (auto confidence {c['min_conf']:.2f}) · "
                          f"proposed view <b>{c['view'] or 'none'}</b>")
            view.value = c["view"] if c["view"] in HUMAN_VIEWS else None
            ed.value = c["ed"] if c["ed"] is not None else 0
            es.value = c["es"] if c["es"] is not None else 0
            frame.max = len(c["frames"]) - 1; frame.value = ed.value; note.value = ""
            draw_frame()

        def on_save(_):
            if view.value is None:
                msg.value = "<span style='color:#b91c1c'>Choose a view first.</span>"; return
            try:
                self.submit(view.value, ed.value, es.value, note.value or None)
                msg.value = "saved"; load()
            except ValueError as e:
                msg.value = f"<span style='color:#b91c1c'>{html.escape(str(e))}</span>"

        frame.observe(draw_frame, "value"); ed.observe(draw_frame, "value"); es.observe(draw_frame, "value")
        set_ed.on_click(lambda _: setattr(ed, "value", frame.value))
        set_es.on_click(lambda _: setattr(es, "value", frame.value))
        save.on_click(on_save); skip.on_click(lambda _: (self.skip(), load()))
        load()
        ui = W.VBox([head, img, frame, sig, view, W.HBox([ed, set_ed]), W.HBox([es, set_es]), note,
                     W.HBox([save, skip]), msg])
        display(ui)
        return ui


class PhenotypeReviewer:
    def __init__(self, sample, context, store, annotator: str):
        self.sample = pd.read_csv(sample) if isinstance(sample, (str, Path)) else sample
        self.ctx = load_context(context) if isinstance(context, (str, Path)) else context
        self.store = store if isinstance(store, AnnotationStore) else AnnotationStore(store)
        self.annotator = annotator
        done = {(r.item_id, r.task) for r in self.store.read().query("source == 'human'").itertuples()}
        self.queue = self.sample[[(str(p), task_name(c)) not in done
                                  for p, c in zip(self.sample.person_id, self.sample.condition)]].reset_index(drop=True)
        self.pos = 0

    def remaining(self) -> int:
        return max(0, len(self.queue) - self.pos)

    def current(self) -> dict | None:
        if self.pos >= len(self.queue):
            return None
        r = self.queue.iloc[self.pos]
        pick = lambda k: self.ctx[k][self.ctx[k].person_id == r.person_id].sort_values("rel_day")
        # algo_label and stratum are deliberately NOT returned: the reviewer is blinded
        return {"person_id": int(r.person_id), "condition": r.condition,
                "question": QUESTION.get(r.condition, r.condition),
                **{k: pick(k) for k in ("diagnoses", "procedures", "labs", "notes")}}

    def submit(self, verdict: str, note: str | None = None) -> None:
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}")
        c = self.current()
        if c is None:
            return
        self.store.add_human(str(c["person_id"]), task_name(c["condition"]), verdict, self.annotator, note)
        self.pos += 1

    def skip(self) -> None:
        self.pos += 1

    @staticmethod
    def render(c: dict) -> str:
        def table(df, cols):
            if df.empty:
                return "<i>none</i>"
            h = "".join(f"<th style='text-align:left;padding:2px 8px'>{x}</th>" for x in cols)
            rows = "".join("<tr>" + "".join(f"<td style='padding:2px 8px'>{html.escape(str(v))}</td>"
                                             for v in r) + "</tr>" for r in df[cols].itertuples(index=False))
            return f"<table><tr>{h}</tr>{rows}</table>"
        notes = "".join(
            f"<details><summary>day {int(n.rel_day)} · {html.escape(str(n.title or 'note'))}</summary>"
            f"<pre style='white-space:pre-wrap;max-height:300px;overflow:auto'>{html.escape(str(n.text))}</pre></details>"
            for n in c["notes"].itertuples()) or "<i>no notes available</i>"
        return (f"<h3>{html.escape(c['question'])}</h3><p>Days are relative to the index date (day 0).</p>"
                f"<h4>Diagnoses</h4>{table(c['diagnoses'], ['rel_day', 'code', 'name'])}"
                f"<h4>Procedures</h4>{table(c['procedures'], ['rel_day', 'code', 'name'])}"
                f"<h4>Key labs</h4>{table(c['labs'], ['rel_day', 'name', 'value', 'unit'])}"
                f"<h4>Notes</h4>{notes}")

    def show(self):
        import ipywidgets as W
        from IPython.display import display
        head, body, msg = W.HTML(), W.HTML(), W.HTML()
        note = W.Text(description="note", placeholder="optional: why")
        btns = {v: W.Button(description=v.capitalize(), button_style={"yes": "success", "no": "danger"}.get(v, ""))
                for v in VERDICTS}
        skip = W.Button(description="Skip")

        def load(*_):
            c = self.current()
            if c is None:
                head.value, body.value = "<b>Review queue empty.</b>", ""; return
            head.value = f"<b>{self.remaining()} records left</b> · condition <b>{c['condition']}</b>"
            body.value = self.render(c); note.value = ""

        for v, b in btns.items():
            b.on_click(lambda _, v=v: (self.submit(v, note.value or None), load()))
        skip.on_click(lambda _: (self.skip(), load()))
        load()
        ui = W.VBox([head, body, note, W.HBox(list(btns.values()) + [skip]), msg])
        display(ui)
        return ui
