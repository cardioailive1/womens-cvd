"""JupyterLab screens for the two-tier label audit (annotate.audit).

Tier 1, data scientists — the automatic label is shown; answer Agree, Disagree (with the
correction) or Uncertain:

    from annotate.audit_ui import AuditScreen
    AuditScreen(audit, annotator="ds_jane", tier=1, phenotype_context=ctx, echo_cache=cache).show()

Tier 2, cardiologist — the same screen with tier=2. No automatic or data-scientist label is shown,
and disagreements, uncertain items and spot-checks are shuffled together, so the cardiologist
labels every item independently.

Logic lives in plain methods (current, submit, skip); show() only wires them to widgets.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

from .audit import ECHO_TASK, LabelAudit, echo_value
from .echo import VIEWS, cavity_signal, read_clip
from .phenotype import QUESTION
from .review_ui import HUMAN_VIEWS, PhenotypeReviewer, _png, _signal_png


class AuditScreen:
    def __init__(self, audit: LabelAudit, annotator: str, tier: int,
                 phenotype_context: dict | str | None = None, echo_cache: str | Path | None = None):
        if tier not in (1, 2):
            raise ValueError("tier is 1 (data scientists) or 2 (cardiologist)")
        allowed = audit.data_scientists if tier == 1 else audit.cardiologists
        if annotator not in allowed:
            raise PermissionError(f"{annotator} is not listed for tier {tier}")
        self.audit, self.annotator, self.tier = audit, annotator, tier
        if isinstance(phenotype_context, (str, Path)):
            phenotype_context = pd.read_pickle(phenotype_context)
        self.ctx = phenotype_context
        self.paths = {}
        if echo_cache is not None:
            self.paths = pd.read_csv(Path(echo_cache) / "index.csv").set_index("item_id").path.to_dict()
        self.refresh()

    def refresh(self):
        self.queue = (self.audit.tier1_queue(self.annotator) if self.tier == 1
                      else self.audit.tier2_queue(self.annotator)[["item_id", "task"]])
        self.pos = 0
        self._clip = None

    def remaining(self) -> int:
        return max(0, len(self.queue) - self.pos)

    def current(self) -> dict | None:
        """Everything the screen shows for the current item. At tier 2 there is no automatic
        label, no data-scientist label and no reason (blinded)."""
        if self.pos >= len(self.queue):
            return None
        r = self.queue.iloc[self.pos]
        item = {"item_id": r.item_id, "task": r.task, "kind": "echo" if r.task == ECHO_TASK else "phenotype"}
        if self.tier == 1:
            item["auto_value"] = self.audit._auto(r.item_id, r.task)
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

    def submit(self, *, verdict: str | None = None, value=None, view=None, ed=None, es=None,
               note: str | None = None) -> None:
        """Tier 1: verdict = agree | disagree | uncertain (disagree on echo needs view/ed/es).
        Tier 2: value = yes | no | unsure for conditions, or view (+ ed, es) for echo."""
        c = self.current()
        if c is None:
            return
        if c["kind"] == "echo" and (view is not None):
            value = echo_value(view, ed, es) if view in VIEWS[:4] else echo_value(view)
        if self.tier == 1:
            self.audit.submit_tier1(c["item_id"], c["task"], verdict, self.annotator, value=value, note=note)
        else:
            if value is None:
                raise ValueError("tier 2 needs the cardiologist's own label")
            self.audit.submit_tier2(c["item_id"], c["task"], value, self.annotator, note)
        self.pos += 1

    def skip(self) -> None:
        self.pos += 1

    # ---- widgets
    def show(self):
        import ipywidgets as W
        from IPython.display import display

        head, body, msg = W.HTML(), W.HTML(), W.HTML()
        img, sig = W.Image(format="png"), W.Image(format="png")
        frame = W.IntSlider(description="frame", min=0, max=1)
        view = W.ToggleButtons(options=HUMAN_VIEWS, description="view")
        ed, es = W.IntText(description="ED frame"), W.IntText(description="ES frame")
        set_ed, set_es = W.Button(description="ED = this frame"), W.Button(description="ES = this frame")
        note = W.Text(description="note", placeholder="optional; say why when disagreeing")
        b_agree = W.Button(description="Agree", button_style="success")
        b_dis = W.Button(description="Disagree (save my correction)", button_style="danger",
                         layout=W.Layout(width="auto"))
        b_unc = W.Button(description="Uncertain")
        b_yes = W.Button(description="Yes", button_style="success")
        b_no = W.Button(description="No", button_style="danger")
        b_uns = W.Button(description="Unsure")
        b_save = W.Button(description="Save and next", button_style="primary")
        b_skip = W.Button(description="Skip")
        box = W.VBox()
        st = {}

        def draw(*_):
            if st.get("kind") != "echo":
                return
            f = st["frames"]; i = min(frame.value, len(f) - 1)
            img.value = _png(f[i], 1 if f.shape[1] >= 300 else 2)
            sig.value = _signal_png(st["signal"], ed.value, es.value, i)

        def load(*_):
            c = self.current()
            st.clear()
            if c is None:
                head.value = "<b>Queue empty.</b>"; box.children = [head]; return
            st.update(c)
            tier_txt = ("Tier 1 audit: check the automatic label" if self.tier == 1
                        else "Tier 2: give your own label (no other labels are shown)")
            head.value = f"<b>{self.remaining()} left</b> · {tier_txt}"
            note.value = ""; msg.value = ""
            if c["kind"] == "phenotype":
                shown = PhenotypeReviewer.render(c)
                if self.tier == 1:
                    shown = (f"<p style='font-size:15px'><b>Automatic label: {html.escape(str(c['auto_value']).upper())}</b>"
                             f" — {html.escape(c['question'])}</p>") + shown
                body.value = shown
                btns = [b_agree, b_dis, b_unc] if self.tier == 1 else [b_yes, b_no, b_uns]
                box.children = [head, body, note, W.HBox(btns + [b_skip]), msg]
            else:
                v = json.loads(c["auto_value"]) if self.tier == 1 else {"view": None, "ed": None, "es": None}
                body.value = (f"<p><b>Automatic: view {v['view']}, ED frame {v['ed']}, ES frame {v['es']}</b>. "
                              "Agree if right; otherwise set the correct view and frames and press Disagree.</p>"
                              if self.tier == 1 else "<p>Choose the view and mark the ED and ES frames.</p>")
                view.value = v["view"] if v["view"] in HUMAN_VIEWS else None
                ed.value, es.value = v["ed"] or 0, v["es"] or 0
                frame.max = len(c["frames"]) - 1; frame.value = ed.value
                draw()
                btns = [b_agree, b_dis, b_unc] if self.tier == 1 else [b_save]
                box.children = [head, body, img, frame, sig, view, W.HBox([ed, set_ed]), W.HBox([es, set_es]),
                                note, W.HBox(btns + [b_skip]), msg]

        def act(**kw):
            try:
                self.submit(note=note.value or None, **kw); load()
            except (ValueError, PermissionError) as e:
                msg.value = f"<span style='color:#b91c1c'>{html.escape(str(e))}</span>"

        def echo_kw():
            return {"view": view.value, "ed": ed.value, "es": es.value}

        b_agree.on_click(lambda _: act(verdict="agree"))
        b_unc.on_click(lambda _: act(verdict="uncertain"))
        b_dis.on_click(lambda _: act(verdict="disagree", **(echo_kw() if st.get("kind") == "echo" else {})))
        b_yes.on_click(lambda _: act(value="yes")); b_no.on_click(lambda _: act(value="no"))
        b_uns.on_click(lambda _: act(value="unsure"))
        b_save.on_click(lambda _: act(**echo_kw()) if view.value else
                        setattr(msg, "value", "<span style='color:#b91c1c'>Choose a view first.</span>"))
        b_skip.on_click(lambda _: (self.skip(), load()))
        frame.observe(draw, "value"); ed.observe(draw, "value"); es.observe(draw, "value")
        set_ed.on_click(lambda _: setattr(ed, "value", frame.value))
        set_es.on_click(lambda _: setattr(es, "value", frame.value))
        load()
        display(box)
        return box
