"""Automatic tier-1 audit: an optional first pass before (or instead of) the data scientists.

The automatic labels were made by rules and models. Checking them by re-running the same rules
would always agree, so the automatic auditor looks for *independent* evidence instead:

  Echo clips  — the metadata rule and a view model (preferably one retrained after the review
                round) must agree; a model-only view needs the check model to agree confidently;
                the ED/ES frames must come from a confident phase detector, sit in a plausible
                order and spacing, and match a second, differently computed phase estimate.
  Conditions  — the labels come from diagnosis codes, so the auditor reads what the codes do not
                use: the clinical and report notes (with negation and history cues), the
                procedures (catheterisation or PCI shortly before a coronary dissection code
                suggests an iatrogenic dissection, not SCAD) and the LVEF stated in echo reports
                (PPCM needs a reduced EF).

Each item gets agree / disagree / uncertain with a confidence and a note saying why. It
disagrees only when the evidence clearly points the other way (and then gives the corrected
value), and says uncertain whenever the evidence is missing or mixed, so it never confirms a
label it found nothing for.

How the verdicts are used is set by `audit.auto_tier1` in config/study.yaml (annotate.audit):
  off      — not used (default).
  triage   — items the auditor agrees with count as tier-1 accepted; every item it disagrees
             with or is unsure about goes to a data scientist, whose answer then decides.
  replace  — the auditor is the whole of tier 1; its disagreements and uncertain items go
             straight to the cardiologist.
Either way the cardiologist's blinded spot-check samples the auto-accepted items separately
(`auto_spot_check_fraction`), so report 09d says how accurate auto-accepted labels are.

    from annotate.auto_audit import audit_phenotypes, audit_echo
    audit.submit_auto(audit_phenotypes(audit.items, ctx))
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from .audit import AUTO_AUDITOR, AUTO_SOURCE  # noqa: F401  (re-exported)
from .echo import T_ED, T_RULE, T_VIEW, VIEWS, cache_path, read_clip
from .store import AnnotationStore

COLUMNS = ["item_id", "task", "verdict", "value", "confidence", "note"]


def _row(item_id, task, verdict, conf, why, value=None) -> dict:
    return {"item_id": str(item_id), "task": task, "verdict": verdict, "value": value,
            "confidence": round(float(conf), 3), "note": "; ".join(why) if isinstance(why, list) else why}


# ================================================================== conditions (notes)
TERMS = {
    "SCAD": [r"spontaneous\s+coronary\s+(artery\s+)?dissection", r"\bSCAD\b",
             r"spontaneous\s+dissection"],
    "PPCM": [r"peri-?partum\s+cardiomyopathy", r"post-?partum\s+cardiomyopathy", r"\bPPCM\b",
             r"pregnancy[- ]associated\s+cardiomyopathy"],
    "PREECLAMPSIA_CVD": [r"pre-?eclampsia", r"\beclampsia", r"\bHELLP\b",
                         r"pregnancy[- ]induced\s+hypertension\s+with\s+proteinuria"],
    "TAKOTSUBO": [r"tako-?tsubo", r"stress(-induced)?\s+cardiomyopathy", r"apical\s+ballooning",
                  r"broken\s+heart\s+syndrome"],
    "CMD_INOCA": [r"microvascular\s+(dysfunction|disease|angina)", r"\bINOCA\b", r"\bANOCA\b",
                  r"\bCMD\b", r"cardiac\s+syndrome\s+x", r"non-?obstructive\s+coronary"],
    "FMD": [r"fibromuscular\s+dysplasia", r"\bFMD\b", r"string[- ]of[- ]beads"],
    # outcomes
    "mace": [r"myocardial\s+infarction", r"\bN?STEMI\b", r"\bMI\b", r"\bstroke\b", r"\bCVA\b",
             r"cerebral\s+infarct", r"intracerebral\s+ha?emorrhage", r"(acute|decompensated)\s+heart\s+failure",
             r"heart\s+failure\s+(exacerbation|admission)", r"\bADHF\b", r"\bexpired\b", r"\bdeceased\b"],
    "postpartum_cvd": [r"myocardial\s+infarction", r"\bN?STEMI\b", r"\bstroke\b", r"\bCVA\b",
                       r"heart\s+failure", r"cardiomyopathy", r"coronary\s+(artery\s+)?dissection", r"\bSCAD\b",
                       r"tako-?tsubo", r"stress\s+cardiomyopathy"],
}
# Days after index whose notes count as evidence (outcomes: after the 7-day blanking period).
WINDOW = {"mace": (8, 365), "postpartum_cvd": (8, 365)}
NEGATION = re.compile(r"\b(no|not|denies|denied|without|negative\s+for|ruled?\s+out|r/o|rule\s+out|"
                      r"unlikely|excluded|free\s+of|no\s+evidence\s+of|absence\s+of|not\s+consistent\s+with)\b"
                      r"[^.;:\n]{0,40}$", re.I)
HISTORY = re.compile(r"\b(history\s+of|hx\s+of|h/o|prior|previous|remote|in\s+20\d\d)\b[^.;:\n]{0,30}$", re.I)
FAMILY = re.compile(r"\b(mother|father|sister|brother|aunt|family\s+history|FHx)\b[^.;:\n]{0,30}$", re.I)
IATROGENIC = re.compile(r"iatrogenic|catheter[- ]induced|guide\s*catheter|procedur(e|al)[- ]related\s+dissection|"
                        r"dissection\s+(during|after|following|complicating)\s+(the\s+)?(pci|cath|angioplasty|stent|intervention)|"
                        r"traumatic\s+dissection", re.I)
LVEF = re.compile(r"\b(?:LVEF|EF|ejection\s+fraction)\b[^0-9%\n]{0,20}(\d{1,2})(?:\s*(?:-|to|–)\s*(\d{1,2}))?\s*%", re.I)
# CPT: diagnostic left-heart catheterisation / coronary angiography, and PCI
CATH_PCI = re.compile(r"^(9345[1-9]|9346[01]|9292[0-9]|9293[0-9]|9294[0-4]|9297[3-9]|C96(0[0-8]))$")
DISSECTION_ICD = ("I25.42",)


def mentions(text: str, condition: str) -> list[str]:
    """Every mention of the condition in the text, each classed as
    affirmed | negated | historical | family."""
    out = []
    for pat in TERMS[condition]:
        for m in re.finditer(pat, text or "", re.I):
            before = text[max(0, m.start() - 80):m.start()]
            before = re.split(r"[.;\n]", before)[-1]        # same sentence only
            if NEGATION.search(before):
                out.append("negated")
            elif FAMILY.search(before):
                out.append("family")
            elif HISTORY.search(before):
                out.append("historical")
            else:
                out.append("affirmed")
    return out


def lvef_values(text: str) -> list[float]:
    vals = []
    for m in LVEF.finditer(text or ""):
        a, b = float(m.group(1)), m.group(2)
        vals.append((a + float(b)) / 2 if b else a)
    return [v for v in vals if 5 <= v <= 85]


def _evidence(condition: str, notes: pd.DataFrame, procs: pd.DataFrame, dx: pd.DataFrame) -> dict:
    """Independent evidence for one woman and one condition (index window only: rel_day >= 0)."""
    lo, hi = WINDOW.get(condition, (0, None))
    if "rel_day" in notes:
        notes = notes[(notes.rel_day >= lo) & ((notes.rel_day <= hi) if hi is not None else True)]
    kinds: list[str] = []
    for t in notes.text.fillna("").astype(str):
        kinds += mentions(t, condition)
    ev = {"affirmed": kinds.count("affirmed"), "negated": kinds.count("negated"),
          "historical": kinds.count("historical") + kinds.count("family"),
          "n_notes": len(notes), "support": [], "contrary": []}
    if condition == "SCAD":
        text = " ".join(notes.text.fillna("").astype(str))
        if IATROGENIC.search(text):
            ev["contrary"].append("notes describe an iatrogenic or traumatic dissection")
        diss = dx[dx.code.astype(str).str.startswith(DISSECTION_ICD)] if len(dx) else dx
        if len(diss) and len(procs):
            d0 = diss.rel_day.min()
            pc = procs[procs.code.astype(str).str.match(CATH_PCI)]
            before = pc[(pc.rel_day < d0) & (pc.rel_day >= d0 - 2)]
            if len(before):
                ev["contrary"].append(f"cath/PCI {int(d0 - before.rel_day.max())} day(s) before the dissection code")
    if condition == "PPCM":
        efs = [v for t in notes.text.fillna("").astype(str) for v in lvef_values(t)]
        if efs:
            if min(efs) <= 45:
                ev["support"].append(f"LVEF {min(efs):.0f}% in reports")
            else:
                ev["contrary"].append(f"every reported LVEF > 45% (lowest {min(efs):.0f}%)")
    return ev


def phenotype_verdict(auto_value: str, ev: dict) -> tuple[str, float, list[str], str | None]:
    """-> verdict, confidence, reasons, corrected value (for disagree)."""
    a, neg, hist = ev["affirmed"], ev["negated"], ev["historical"]
    sup, con = ev["support"], ev["contrary"]
    why = [f"notes: {a} affirmed, {neg} negated, {hist} historical/family mention(s) in {ev['n_notes']} note(s)"]
    why += [f"supports: {s}" for s in sup] + [f"against: {c}" for c in con]
    if str(auto_value).lower() == "yes":
        if (a or sup) and not con:
            return "agree", min(0.97, 0.75 + 0.07 * (a + len(sup))), why, None
        if con and not a and not sup:
            return "disagree", 0.8 if len(con) > 1 or neg else 0.7, why, "no"
        if neg and not a and not sup:
            return "disagree", 0.7, why, "no"
        if not (a or sup or con or neg):
            why.append("no independent evidence found for the condition")
        return "uncertain", 0.5, why, None
    # automatic label is "no"
    if a and not con and not neg:
        return "disagree", min(0.9, 0.65 + 0.08 * a), why, "yes"
    if a or hist or sup:
        return "uncertain", 0.5, why, None
    if ev["n_notes"] == 0:
        return "uncertain", 0.5, why + ["no notes to check against"], None
    return "agree", 0.85 if neg or con else 0.8, why, None


def audit_phenotypes(items: pd.DataFrame, context: dict | str | Path) -> pd.DataFrame:
    """Automatic tier-1 verdicts for condition-label items (annotate.audit.phenotype_audit_items),
    from the context fetched by annotate.phenotype.fetch_context."""
    if isinstance(context, (str, Path)):
        context = pd.read_pickle(context)
    get = lambda k: context.get(k, pd.DataFrame(columns=["person_id", "rel_day", "code", "text"]))
    notes, procs, dx = get("notes"), get("procedures"), get("diagnoses")
    by = lambda df: {p: g for p, g in df.groupby("person_id")} if len(df) else {}
    nb, pb, db = by(notes), by(procs), by(dx)
    empty = lambda df: df.iloc[0:0]
    rows = []
    for r in items.itertuples():
        if not str(r.task).startswith("phenotype:"):
            continue
        cond = r.task.split(":", 1)[1]
        if cond not in TERMS:
            rows.append(_row(r.item_id, r.task, "uncertain", 0.0, f"no evidence rules for {cond}"))
            continue
        pid = int(r.item_id)
        ev = _evidence(cond, nb.get(pid, empty(notes)), pb.get(pid, empty(procs)), db.get(pid, empty(dx)))
        verdict, conf, why, value = phenotype_verdict(r.auto_value, ev)
        rows.append(_row(r.item_id, r.task, verdict, conf, why, value))
    return pd.DataFrame(rows, columns=COLUMNS)


# ================================================================== echo clips
def _alt_ed(frames: np.ndarray) -> tuple[int | None, int | None]:
    """Second ED estimate, computed differently from annotate.echo.detect_phases: the frame
    with the lowest mean brightness in a wider central box (largest blood pool), after
    removing slow drift, within the first cycle found by peak spacing. -> (ED frame, cycle length)."""
    T, H, W = frames.shape
    if T < 8:
        return None, None
    box = frames[:, int(H * .2):int(H * .9), int(W * .15):int(W * .85)]
    s = -box.reshape(T, -1).mean(1)
    s = s - signal.savgol_filter(s, min(T - (1 - T % 2), 31), 1) if T >= 9 else s - s.mean()
    peaks, _ = signal.find_peaks(s, distance=3)
    if not len(peaks):
        return int(np.argmax(s)), None
    if len(peaks) >= 2:
        period = int(round(np.median(np.diff(peaks))))
        first_cycle = peaks[peaks < peaks[0] + period]
        return int(first_cycle[np.argmax(s[first_cycle])]), period
    return int(peaks[0]), None


def audit_echo(items: pd.DataFrame, echo_store: AnnotationStore, cache_dir: str | Path,
               view_model: str | Path | None = None, min_view_conf: float = 0.9,
               min_phase_conf: float = 0.6, frame_tolerance: int = 2,
               frame_time_default_ms: float = 33.3) -> pd.DataFrame:
    """Automatic tier-1 verdicts for echo items (annotate.audit.echo_audit_items).

    view_model: a view model trained AFTER the labelling pass (e.g. on the reviewed seed set
    plus the audited items so far). Re-using the model that made the labels is not independent;
    without a check model, model-only views are left for a human (uncertain)."""
    lat = echo_store.latest()
    lat = lat[~lat.source.eq("human")]
    get = lambda t: lat[lat.task == t].set_index("item_id")
    rule, view, ed_s = get(T_RULE), get(T_VIEW), get(T_ED)
    paths = {}
    idx = Path(cache_dir) / "index.csv"
    if idx.exists():
        paths = pd.read_csv(idx).set_index("item_id").path.to_dict()
    probs = {}
    if view_model is not None:
        from .echo import load_view_model, predict_views
        net = load_view_model(view_model)
        ids = [i for i in items.item_id if cache_path(cache_dir, i).exists()]
        for s in range(0, len(ids), 256):
            chunk = ids[s:s + 256]
            p = predict_views(net, [np.load(cache_path(cache_dir, i))["thumbs"] for i in chunk])
            probs.update(dict(zip(chunk, p)))

    rows = []
    for r in items.itertuples():
        v = json.loads(r.auto_value)
        why, flags, conf = [], [], 1.0
        rv = rule.value.get(r.item_id) if len(rule) else None
        rv = rv if isinstance(rv, str) and rv else None
        rc = float(rule.confidence.get(r.item_id, 0) or 0) if len(rule) else 0.0
        src = view.source.get(r.item_id, "") if len(view) else ""
        # ---- view
        model_view = None
        if r.item_id in probs:
            p = probs[r.item_id]
            model_view, mc = VIEWS[int(p.argmax())], float(p.max())
        if src == "auto:rules!=model":
            flags.append("labelling pass: metadata rule and view model disagreed")
        if rv == v["view"] and rc >= 0.95:
            why.append(f"view {v['view']} from DICOM metadata (rule)")
            conf = min(conf, rc)
            if model_view is not None and model_view != v["view"] and v["view"] != "OTHER":
                flags.append(f"check model says {model_view} ({mc:.2f})")
            elif model_view == v["view"]:
                why.append(f"check model agrees ({mc:.2f})")
        elif model_view is not None:
            if model_view == v["view"] and mc >= min_view_conf:
                why.append(f"view {v['view']}: check model agrees ({mc:.2f})")
                conf = min(conf, mc)
            elif model_view != v["view"] and mc >= max(0.95, min_view_conf) and rv is None:
                # no metadata name, labelling model and check model confidently differ
                corr = {"view": model_view, "ed": v["ed"] if model_view in VIEWS[:4] else None,
                        "es": v["es"] if model_view in VIEWS[:4] else None}
                rows.append(_row(r.item_id, r.task, "disagree", mc,
                                 [f"view {v['view']} came from the labelling model; check model says "
                                  f"{model_view} ({mc:.2f})"], json.dumps(corr, sort_keys=True)))
                continue
            else:
                flags.append(f"check model {model_view} ({mc:.2f}) does not confirm {v['view']}")
        else:
            flags.append(f"view {v['view']} has no metadata name and no check model was given")
        # ---- phases
        if v["view"] in VIEWS[:4]:
            pc = float(ed_s.confidence.get(r.item_id, 0) or 0) if len(ed_s) else 0.0
            method = str(ed_s.source.get(r.item_id, "")) if len(ed_s) else ""
            if v["ed"] is None or v["es"] is None:
                flags.append("ED or ES frame missing")
            else:
                if pc < min_phase_conf:
                    flags.append(f"phase confidence {pc:.2f} < {min_phase_conf}")
                gap = v["es"] - v["ed"]
                frames = None
                if r.item_id in paths:
                    try:
                        frames, meta = read_clip(paths[r.item_id])
                    except Exception as e:                       # noqa: BLE001
                        flags.append(f"clip unreadable for re-check ({type(e).__name__})")
                ft = (meta.get("frame_time_ms") if frames is not None else None) or frame_time_default_ms
                systole_ms = gap * ft
                if gap <= 0:
                    flags.append("ES is not after ED")
                elif not 150 <= systole_ms <= 550:
                    flags.append(f"ED-to-ES {systole_ms:.0f} ms is outside 150-550 ms")
                if frames is not None:
                    alt, period = _alt_ed(frames)
                    d = abs(alt - v["ed"]) if alt is not None else None
                    if d is not None and period:                 # ED recurs every cycle
                        d = min(d % period, period - d % period)
                    if method.endswith("ecg+image") and pc >= 0.95:
                        why.append("ED from the ECG R-wave")
                    elif d is not None and d <= frame_tolerance:
                        why.append(f"second ED estimate agrees (frame {alt})")
                    elif alt is not None:
                        flags.append(f"second ED estimate is frame {alt}, labelled {v['ed']}")
                conf = min(conf, pc)
        if flags:
            rows.append(_row(r.item_id, r.task, "uncertain", 0.5, flags + why))
        else:
            rows.append(_row(r.item_id, r.task, "agree", conf, why))
    return pd.DataFrame(rows, columns=COLUMNS)
