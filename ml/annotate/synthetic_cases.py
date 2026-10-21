"""Synthetic training kit: made-up patient records and echo clips with known answers.

Reviewers practise on this kit BEFORE they get workspace access. Nothing in it comes from
patients: every record is generated from hand-written case templates, and every echo clip is a
drawn cartoon of a beating ventricle saved as DICOM. Because the answers are known, practice can
give feedback and qualification can be scored, all on a laptop.

    python -m annotate.synthetic_cases --out training_kit [--seed 0]

writes, under the output folder:
  context.pkl      diagnoses, procedures, labs and notes for every synthetic woman
  echo/*.dcm       synthetic echo clips; echo_auto.csv + cache/ from the automatic echo pass
  sets.csv         item_id, task, set (practice | reference | audit), part (A | B for reference)
  training.csv     the answer key ("ref:" tasks by "synthetic-key"), each with a teaching note
  audit_items.csv  a mock audit set: the automatic (code-based) label of each item, some wrong

The case templates cover the six conditions and three outcome labels, each with clear cases,
the traps review exists to catch (a catheter-induced dissection coded as SCAD, an LVEF of 58%
coded as PPCM, an event in the blanking days...), past-history mentions and records that cannot
settle the question. The clinical lead should read the key before trainees use it, and can
override any answer by relabelling it in TrainingScreen(mode="reference").

Limits: the echo clips are cartoons. They teach the review screen, the frame slider and how to
mark end-diastole and end-systole; they do not teach how real PLAX, PSAX, A4C and A2C images
look. View recognition is taught from real teaching images in the cardiologist-led session.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import ECHO_TASK, echo_value
from .phenotype import task_name
from .store import AnnotationStore

KEY_AUTHOR = "synthetic-key"
PERIOD = 16                       # frames per cardiac cycle in the synthetic clips
ARTERIES = ["LAD", "mid LAD", "left circumflex", "first obtuse marginal", "first diagonal", "RCA"]
YEARS = [2014, 2015, 2016, 2017, 2018, 2019]


# ================================================================== record templates
# Each template returns (gold answer, automatic code label, lesson, record) for one woman.
# record = {"dx": [(day, code, name, inpatient)], "px": [(day, code, name)],
#           "labs": [(day, name, value, unit)], "notes": [(day, title, text)]}

def _rec(dx=(), px=(), labs=(), notes=()):
    return {"dx": list(dx), "px": list(px), "labs": list(labs), "notes": list(notes)}


def _scad(r):
    a, d = r.choice(ARTERIES), int(r.integers(0, 3))
    return [
        ("yes", "yes", "Angiography describes a spontaneous dissection: answer Yes.",
         _rec([(d + 1, "I25.42", "Coronary artery dissection", True)], [(d, "93458", "Left heart catheterization with coronary angiography")],
              [(d, "troponin I", float(r.integers(300, 4000)), "ng/L")],
              [(d, "Cath report", f"Coronary angiogram: long smooth narrowing of the {a} consistent with spontaneous coronary artery dissection, type 2. No atherosclerotic disease. Managed conservatively.")])),
        ("no", "yes", "The dissection happened during PCI: catheter-induced, not spontaneous. Answer No; this is the main SCAD trap.",
         _rec([(d + 1, "I25.42", "Coronary artery dissection", True)], [(d, "92928", "Percutaneous coronary stent placement")],
              notes=[(d, "Procedure note", f"PCI to a 90% stenosis of the {a}. Guide catheter-induced dissection of the left main extending into the {a}, treated with an additional stent. Atherosclerotic disease elsewhere.")])),
        ("no", "yes", "SCAD is mentioned only as past history and this angiogram shows no new dissection. Not a new case: answer No.",
         _rec([(d, "I25.42", "Coronary artery dissection", True)],
              notes=[(d, "Cardiology consult", f"History of SCAD of the {a} in {r.choice(YEARS)}. Now chest pain; repeat angiogram shows healed vessel, no new dissection.")])),
        ("unsure", "yes", "Only the code exists, with no angiography report or note: the record cannot settle it. Answer Unsure.",
         _rec([(d + 2, "I25.42", "Coronary artery dissection", False), (d + 20, "I25.42", "Coronary artery dissection", False)])),
        ("yes", "no", "The note describes SCAD but it was coded as an NSTEMI: the code rule missed it. Answer Yes.",
         _rec([(d, "I21.4", "NSTEMI", True)], [(d, "93458", "Left heart catheterization with coronary angiography")],
              notes=[(d, "Cath report", f"Spontaneous dissection of the {a} with intramural hematoma. No plaque. Likely SCAD; screen for FMD.")])),
        ("no", "no", "Normal coronaries, no dissection anywhere: answer No.",
         _rec([(d, "R07.9", "Chest pain, unspecified", False)], [(d, "93458", "Left heart catheterization with coronary angiography")],
              notes=[(d, "Cath report", "Angiogram: smooth coronary arteries, no stenosis and no dissection.")])),
    ]


def _ppcm(r):
    ef_low, ef_ok, dl = int(r.integers(20, 40)), int(r.integers(55, 63)), -int(r.integers(10, 60))
    deliv = (dl, "O80", "Delivery", True)
    return [
        ("yes", "yes", f"New LVEF {ef_low}% after delivery with no other cause: PPCM. Answer Yes.",
         _rec([deliv, (5, "O90.3", "Peripartum cardiomyopathy", True)],
              notes=[(5, "Echo report", f"LVEF {ef_low}%, global hypokinesis, LV mildly dilated. Prenatal echo was normal. No valve disease, coronaries normal on CTA.")])),
        ("no", "yes", f"Coded as PPCM but the LVEF is {ef_ok}%. PPCM needs LV dysfunction: answer No.",
         _rec([deliv, (3, "O90.3", "Peripartum cardiomyopathy", True)],
              notes=[(3, "Echo report", f"LVEF {ef_ok}%. Normal LV size and function. Dyspnoea likely from anaemia.")])),
        ("no", "yes", "The cardiomyopathy was known before pregnancy, so it is not peripartum. Answer No.",
         _rec([deliv, (4, "O90.3", "Peripartum cardiomyopathy", True)],
              notes=[(4, "Cardiology note", f"Known dilated cardiomyopathy since {r.choice(YEARS)}, LVEF 35% before this pregnancy, unchanged now.")])),
        ("yes", "no", "Coded as a general cardiomyopathy (I42.0), but the note describes PPCM. The code rule missed it: answer Yes.",
         _rec([deliv, (6, "I42.0", "Dilated cardiomyopathy", True)],
              notes=[(6, "Echo report", f"New LVEF {ef_low}% six weeks postpartum. No ischaemia, no valve disease. Impression: peripartum cardiomyopathy.")])),
        ("unsure", "yes", "Coded twice, but no LVEF is reported anywhere. Answer Unsure.",
         _rec([deliv, (2, "O90.3", "Peripartum cardiomyopathy", False), (15, "O90.3", "Peripartum cardiomyopathy", False)],
              notes=[(2, "Clinic note", "Postpartum dyspnoea. Echo ordered, result pending.")])),
    ]


def _preeclampsia(r):
    w, sbp = int(r.integers(30, 39)), int(r.integers(160, 175))
    return [
        ("yes", "yes", "Hypertension after 20 weeks with proteinuria, diagnosed as preeclampsia: answer Yes.",
         _rec([(1, "O14.1", "Severe preeclampsia", True)], labs=[(1, "urine protein", 3.0, "g/24h")],
              notes=[(1, "Obstetric note", f"{w} weeks. BP {sbp}/105, proteinuria 3 g/24 h, headache. Diagnosis: preeclampsia with severe features.")])),
        ("yes", "yes", "Low platelets, raised liver enzymes and haemolysis: HELLP counts. Answer Yes.",
         _rec([(2, "O14.2", "HELLP syndrome", True)], labs=[(2, "platelets", float(r.integers(40, 90)), "10*3/uL")],
              notes=[(2, "Obstetric note", f"{w} weeks. Platelets low, AST 310, LDH raised, haemolysis on smear. HELLP syndrome.")])),
        ("no", "yes", "The code says preeclampsia but the note rules it out: gestational hypertension only. Answer No.",
         _rec([(1, "O14.9", "Preeclampsia, unspecified", True)],
              notes=[(1, "Obstetric note", f"{w} weeks. BP 146/92, no proteinuria, labs normal. Gestational hypertension; preeclampsia ruled out.")])),
        ("no", "no", "Chronic hypertension that predates pregnancy is not preeclampsia. Answer No.",
         _rec([(0, "O10.019", "Pre-existing essential hypertension in pregnancy", False)],
              notes=[(0, "Obstetric note", "Chronic hypertension on labetalol since before pregnancy. No proteinuria. Stable.")])),
        ("unsure", "no", "Raised blood pressure in pregnancy with no stated diagnosis and no proteinuria result: Unsure.",
         _rec([(1, "O16.3", "Unspecified maternal hypertension", False)],
              notes=[(1, "Triage note", f"{w} weeks. BP 152/96. Urine sample sent. Follow-up arranged.")])),
    ]


def _takotsubo(r):
    return [
        ("yes", "yes", "Apical ballooning, non-obstructive coronaries, emotional trigger, diagnosed as Takotsubo: answer Yes.",
         _rec([(1, "I51.81", "Takotsubo syndrome", True)], [(0, "93458", "Left heart catheterization with coronary angiography")],
              notes=[(1, "Cardiology note", "Chest pain after a family bereavement. Angiogram: no obstructive disease. Ventriculogram: apical ballooning. Takotsubo cardiomyopathy.")])),
        ("no", "yes", "Initially coded as Takotsubo, but a culprit LAD occlusion was found and stented: this was an MI. Answer No.",
         _rec([(0, "I51.81", "Takotsubo syndrome", True), (1, "I21.09", "STEMI anterior wall", True)], [(0, "92928", "Percutaneous coronary stent placement")],
              notes=[(1, "Discharge summary", "Apical akinesis. Angiogram showed proximal LAD occlusion, stented. Final diagnosis anterior STEMI, not Takotsubo.")])),
        ("no", "yes", "Takotsubo is past history; nothing new in this window. Answer No.",
         _rec([(2, "I51.81", "Takotsubo syndrome", True)],
              notes=[(2, "Clinic note", f"History of Takotsubo in {r.choice(YEARS)}, LV function recovered. Now palpitations only.")])),
        ("unsure", "yes", "Wall-motion abnormality but no coronary imaging, so a culprit lesion cannot be excluded. Answer Unsure.",
         _rec([(1, "I51.81", "Takotsubo syndrome", True)],
              notes=[(1, "Echo report", "Apical hypokinesis with preserved basal function. Angiography not performed.")])),
    ]


def _cmd(r):
    cfr = round(float(r.uniform(1.5, 2.0)), 1)
    return [
        ("yes", "yes", f"Angina, non-obstructive coronaries and abnormal coronary function testing (CFR {cfr}): CMD. Answer Yes.",
         _rec([(1, "I20.81", "Angina with coronary microvascular dysfunction", False), (30, "I20.81", "Angina with coronary microvascular dysfunction", False)],
              [(1, "93458", "Left heart catheterization with coronary angiography")],
              notes=[(1, "Cath report", f"No obstructive coronary disease. Coronary flow reserve {cfr}, IMR 32. Coronary microvascular dysfunction.")])),
        ("no", "yes", "A 70% stenosis treated with PCI explains the symptoms: obstructive disease, not INOCA. Answer No.",
         _rec([(1, "I25.85", "Chronic coronary microvascular dysfunction", False), (10, "I25.85", "Chronic coronary microvascular dysfunction", False)],
              [(1, "92928", "Percutaneous coronary stent placement")],
              notes=[(1, "Cath report", "70% mid LAD stenosis, FFR 0.72, treated with a drug-eluting stent.")])),
        ("unsure", "yes", "Non-obstructive coronaries but no microvascular testing and no diagnosis: Unsure.",
         _rec([(1, "I20.81", "Angina with coronary microvascular dysfunction", True)],
              notes=[(1, "Cath report", "Mild luminal irregularities, no obstructive disease. No further testing performed.")])),
        ("yes", "no", "The note documents microvascular dysfunction but it was coded as unspecified angina. Answer Yes.",
         _rec([(2, "I20.9", "Angina, unspecified", False)],
              notes=[(2, "Cardiology note", "Acetylcholine testing: microvascular spasm without epicardial spasm. Diagnosis: INOCA due to coronary microvascular dysfunction.")])),
    ]


def _fmd(r):
    bed = r.choice(["renal", "carotid", "vertebral", "iliac"])
    return [
        ("yes", "yes", f"String-of-beads appearance of the {bed} arteries on imaging: FMD. Answer Yes.",
         _rec([(3, "I77.3", "Arterial fibromuscular dysplasia", False), (40, "I77.3", "Arterial fibromuscular dysplasia", False)],
              notes=[(3, "CTA report", f"String-of-beads appearance of both {bed} arteries, consistent with multifocal fibromuscular dysplasia.")])),
        ("no", "yes", "The FMD belongs to her mother; her own imaging is normal. Answer No.",
         _rec([(2, "I77.3", "Arterial fibromuscular dysplasia", True)],
              notes=[(2, "Clinic note", f"Family history: mother has fibromuscular dysplasia. Patient's {bed} CTA is normal.")])),
        ("no", "no", "Atherosclerotic stenosis is not FMD. Answer No.",
         _rec([(5, "I70.1", "Atherosclerosis of renal artery", False)],
              notes=[(5, "Duplex report", "Ostial renal artery stenosis from calcified atherosclerotic plaque.")])),
        ("unsure", "yes", "Imaging calls it only possible FMD and asks for follow-up: Unsure.",
         _rec([(4, "I77.3", "Arterial fibromuscular dysplasia", True)],
              notes=[(4, "Ultrasound report", f"Irregular {bed} artery contour, possible FMD. CTA advised.")])),
    ]


def _mace(r):
    day = int(r.integers(30, 330))
    return [
        ("yes", "yes", f"An NSTEMI admission on day {day}, inside day 8 to 12 months: MACE. Answer Yes.",
         _rec([(day, "I21.4", "NSTEMI", True)], notes=[(day, "Discharge summary", "Admitted with NSTEMI, troponin rise, treated medically.")])),
        ("yes", "yes", f"An ischaemic stroke admission on day {day}: MACE. Answer Yes.",
         _rec([(day, "I63.9", "Cerebral infarction", True)], notes=[(day, "Neurology note", "Acute ischaemic stroke, left MCA territory, admitted.")])),
        ("no", "no", "The only event is on day 3, inside the index episode (days 1–7): answer No.",
         _rec([(3, "I21.4", "NSTEMI", True)], notes=[(3, "Progress note", "Troponin rise on day 3 of the index admission, type 2 MI.")])),
        ("no", "no", "Heart failure managed in clinic without admission or ED care does not count. Answer No.",
         _rec([(day, "I50.9", "Heart failure, unspecified", False)], notes=[(day, "Clinic note", "Mild heart failure symptoms, diuretic increased. No admission.")])),
        ("yes", "no", f"Death on day {day} counts as MACE even without a cardiac code. Answer Yes.",
         _rec(notes=[(day, "Death note", "Found unresponsive at home; pronounced dead. Presumed cardiac arrest.")])),
        ("unsure", "no", "A 'heart attack' at another hospital with no date or setting cannot be placed in the window. Answer Unsure.",
         _rec(notes=[(60, "Clinic note", "Patient reports she had a 'heart attack' at an outside hospital some time this year. No records available.")])),
    ]


def _postpartum(r):
    day = int(r.integers(30, 300))
    deliv = (-int(r.integers(10, 90)), "O80", "Delivery", True)
    return [
        ("yes", "yes", f"Cardiomyopathy admission on day {day}, within 12 months of delivery: answer Yes.",
         _rec([deliv, (day, "I42.9", "Cardiomyopathy", True)], notes=[(day, "Discharge summary", "Admitted with new cardiomyopathy, LVEF 30%.")])),
        ("no", "no", "The SCAD happened at delivery, before the index date: not counted. Answer No.",
         _rec([deliv, (deliv[0], "I25.42", "Coronary artery dissection", True)], notes=[(1, "Clinic note", "SCAD at delivery, recovered. No new events since.")])),
        ("no", "no", "The only event is on day 4, inside the blanking days. Answer No.",
         _rec([deliv, (4, "I50.9", "Heart failure", True)], notes=[(4, "Progress note", "Pulmonary oedema on day 4 of the index admission, resolved.")])),
        ("unsure", "no", "An event is mentioned but its timing relative to delivery is unclear: Unsure.",
         _rec([deliv], notes=[(90, "Clinic note", "Reports a stroke 'after the baby was born'; no dates or records.")])),
    ]


def _atypical(r):
    return [
        ("yes", "yes", "ACS presenting with fatigue, nausea and jaw pain, chest pain denied: atypical. Answer Yes.",
         _rec([(0, "I21.4", "NSTEMI", True), (0, "R53.83", "Fatigue", True), (0, "R11.0", "Nausea", True)],
              notes=[(0, "ED note", "Two days of fatigue, nausea and jaw ache. Denies chest pain. Troponin raised: NSTEMI.")])),
        ("no", "yes", "The codes have no chest-pain code, but the note documents chest pressure: not atypical. Answer No.",
         _rec([(0, "I21.4", "NSTEMI", True), (0, "R11.0", "Nausea", True)],
              notes=[(0, "ED note", "Central chest pressure radiating to the left arm, with nausea. NSTEMI.")])),
        ("unsure", "no", "The presenting symptoms are not recorded anywhere: Unsure.",
         _rec([(0, "I21.4", "NSTEMI", True)], notes=[(0, "Transfer note", "Transferred with NSTEMI for angiography. History not available.")])),
    ]


TEMPLATES = {"SCAD": _scad, "PPCM": _ppcm, "PREECLAMPSIA_CVD": _preeclampsia, "TAKOTSUBO": _takotsubo,
             "CMD_INOCA": _cmd, "FMD": _fmd, "mace": _mace, "postpartum_cvd": _postpartum, "atypical": _atypical}


# ================================================================== echo clips
def make_clip(view: str, n: int = 40, size: int = 112, phase0: int = 0, seed: int = 0) -> tuple[np.ndarray, int]:
    """Cartoon cine: dark cavities whose area peaks every PERIOD frames (end-diastole at phase0)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    fr = np.zeros((n, size, size), np.uint8)
    shapes = {"PLAX": [(0.55, 0.5, 0.12, 0.30)], "PSAX": [(0.55, 0.5, 0.20, 0.20)],
              "A4C": [(0.55, 0.38, 0.25, 0.10), (0.55, 0.62, 0.25, 0.10)],
              "A2C": [(0.55, 0.5, 0.30, 0.10)], "OTHER": []}[view]
    for t in range(n):
        img = 150 + 40 * rng.standard_normal((size, size))
        scale = 0.75 + 0.25 * np.cos(2 * np.pi * (t - phase0) / PERIOD)
        for cy, cx, ry, rx in shapes:
            m = ((yy - cy * size) / (ry * size * scale)) ** 2 + ((xx - cx * size) / (rx * size * scale)) ** 2 <= 1
            img[m] = 20 + 5 * rng.standard_normal(m.sum())
        if view == "OTHER":
            img = 255 * rng.random((size, size))
        fr[t] = np.clip(img, 1, 255)
    return fr, phase0


def write_dicom(path: Path, frames: np.ndarray, desc: str = "", frame_time: float = 33.3) -> None:
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.3.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset(); ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID; ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "US"; ds.SeriesDescription = desc; ds.PatientName = "SYNTHETIC^TRAINING"
    ds.NumberOfFrames = len(frames); ds.Rows, ds.Columns = frames.shape[1:]
    ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8; ds.BitsStored = 8; ds.HighBit = 7; ds.PixelRepresentation = 0
    ds.FrameTime = frame_time; ds.UltrasoundColorDataPresent = 0
    ds.PixelData = frames.tobytes()
    ds.save_as(path, enforce_file_format=True)


ECHO_NAMES = {"PLAX": ["PLAX", "Parasternal long", ""], "PSAX": ["PSAX MV", "SAX pap", ""],
              "A4C": ["Apical 4CH", "AP4", ""], "A2C": ["A2C", "Apical 2 chamber", ""], "OTHER": ["Subcostal", ""]}


# ================================================================== the kit
def _women(n_per_set: dict[str, int], rng) -> list[dict]:
    """Synthetic women, one label each, spread round-robin over the nine labels and their variants."""
    labels = list(TEMPLATES)
    offset = {"practice": 0, "reference": 1, "audit": 2}
    out, pid = [], 900000
    for set_name, n in n_per_set.items():
        for k in range(n):
            lab = labels[k % len(labels)]
            variants = TEMPLATES[lab](rng)
            gold, code, lesson, rec = variants[(k // len(labels) + offset.get(set_name, 0)) % len(variants)]
            out.append({"person_id": pid, "label": lab, "set": set_name, "gold": gold, "code": code,
                        "lesson": lesson, "rec": rec, "k": k})
            pid += 1
    return out


def _context(women: list[dict]) -> dict[str, pd.DataFrame]:
    dx, px, labs, notes = [], [], [], []
    for w in women:
        p, r = w["person_id"], w["rec"]
        dx += [(p, d, c, n, 9201 if inpt else 9202) for d, c, n, inpt in r["dx"]]
        px += [(p, d, "CPT4", c, n) for d, c, n in r["px"]]
        labs += [(p, d, n, v, u) for d, n, v, u in r["labs"]]
        notes += [(p, d, t, x) for d, t, x in r["notes"]]
    return {"diagnoses": pd.DataFrame(dx, columns=["person_id", "rel_day", "code", "name", "visit_concept_id"]),
            "procedures": pd.DataFrame(px, columns=["person_id", "rel_day", "vocab", "code", "name"]),
            "labs": pd.DataFrame(labs, columns=["person_id", "rel_day", "name", "value", "unit"]),
            "notes": pd.DataFrame(notes, columns=["person_id", "rel_day", "title", "text"])}


def build_kit(out, seed: int = 0, n_reference: int = 40, n_practice: int = 60, n_audit: int = 72,
              n_echo_reference: int = 40, n_echo_practice: int = 40, n_echo_audit: int = 24,
              audit_error_rate: float = 0.15) -> dict:
    """Write the whole kit under `out` and return its paths."""
    from . import echo as E
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # ---- records
    women = _women({"practice": n_practice, "reference": n_reference, "audit": n_audit}, rng)
    pd.to_pickle(_context(women), out / "context.pkl")
    rows, key, audit = [], [], []
    for w in women:
        iid, task = str(w["person_id"]), task_name(w["label"])
        # alternating halves: with nine labels cycling, every label lands in both halves
        part = ("A" if w["k"] % 2 == 0 else "B") if w["set"] == "reference" else ""
        rows.append({"item_id": iid, "task": task, "set": w["set"], "part": part})
        key.append({"item_id": iid, "task": "ref:" + task, "value": w["gold"], "confidence": 1.0,
                    "source": "human", "annotator": KEY_AUTHOR, "note": w["lesson"]})
        if w["set"] == "audit":
            audit.append({"item_id": iid, "task": task, "auto_value": w["code"]})

    # ---- echo clips
    (out / "echo").mkdir(exist_ok=True)
    echo_rows, truth = [], {}
    views = ["PLAX", "PSAX", "A4C", "A2C", "OTHER"]
    n_echo = n_echo_practice + n_echo_reference + n_echo_audit
    for k in range(n_echo):
        v = views[k % 5]
        ed = int(rng.integers(0, PERIOD // 2))
        fr, _ = make_clip(v, n=40, phase0=ed, seed=seed * 1000 + k)
        p = out / "echo" / f"clip{k:03d}.dcm"
        write_dicom(p, fr, desc=str(rng.choice(ECHO_NAMES[v])))
        iid = E.item_id(str(p))
        truth[iid] = (v, ed)
        set_name = "practice" if k < n_echo_practice else ("reference" if k < n_echo_practice + n_echo_reference else "audit")
        part = ""
        if set_name == "reference":
            part = "A" if (k - n_echo_practice) % 2 == 0 else "B"
        rows.append({"item_id": iid, "task": ECHO_TASK, "set": set_name, "part": part})
        val = echo_value(v, ed, ed + PERIOD // 2) if v != "OTHER" else echo_value("OTHER")
        lesson = (f"{v}: end-diastole is frame {ed}, where the cavity is largest; end-systole is frame "
                  f"{ed + PERIOD // 2}, half a cycle later, where it is smallest." if v != "OTHER"
                  else "No ventricle can be measured in this clip: OTHER, and no frames.")
        key.append({"item_id": iid, "task": "ref:" + ECHO_TASK, "value": val, "confidence": 1.0,
                    "source": "human", "annotator": KEY_AUTHOR, "note": lesson})
        echo_rows.append({"person_id": 990000 + k, "record_date": "2020-01-01", "path": str(p), "set": set_name})
    manifest = pd.DataFrame(echo_rows)
    manifest.to_csv(out / "echo_manifest.csv", index=False)
    echo_store = AnnotationStore(out / "echo_auto.csv")
    E.run_auto(manifest, echo_store, out / "cache")

    # mock audit: the automatic echo labels, with some deliberately wrong so there is something to find
    from .audit import echo_audit_items
    ai = echo_audit_items(echo_store, manifest[manifest.set == "audit"])
    for i, r in ai.iterrows():
        if rng.random() < audit_error_rate:
            v = json.loads(r.auto_value)
            if v["view"] in views[:4] and rng.random() < 0.5 and v["ed"] is not None and v["es"] is not None:
                v["ed"], v["es"] = min(v["ed"] + 5, 30), min(v["es"] + 5, 39)      # frames off by 5
            else:
                v = {"view": "A2C" if v["view"] != "A2C" else "A4C", "ed": v["ed"], "es": v["es"]}
            ai.at[i, "auto_value"] = echo_value(v["view"], v["ed"], v["es"])
    audit += ai.to_dict("records")

    sets = pd.DataFrame(rows)
    sets.to_csv(out / "sets.csv", index=False)
    pd.DataFrame(audit).to_csv(out / "audit_items.csv", index=False)
    store = AnnotationStore(out / "training.csv")
    if store.path.exists():
        store.path.unlink()
    store.add(key)
    (out / "README.txt").write_text(
        "SYNTHETIC TRAINING KIT - no patient data. Generated by annotate.synthetic_cases.\n"
        "Open ml/notebooks/reviewer_training.ipynb and point KIT at this folder.\n")
    return {"out": out, "sets": out / "sets.csv", "context": out / "context.pkl", "store": out / "training.csv",
            "cache": out / "cache", "audit_items": out / "audit_items.csv", "echo_store": out / "echo_auto.csv",
            "n_records": len(women), "n_clips": n_echo}


def sizes_from_config(cfg) -> dict:
    """build_kit keyword arguments from `training.kit_size` in config/study.yaml."""
    k = (getattr(cfg, "training", None) or {}).get("kit_size") or {}
    g = lambda s, kind, d: int((k.get(s) or {}).get(kind, d))
    return {"n_reference": g("reference", "records", 40), "n_echo_reference": g("reference", "echo", 40),
            "n_practice": g("practice", "records", 60), "n_echo_practice": g("practice", "echo", 40),
            "n_audit": g("mock_audit", "records", 72), "n_echo_audit": g("mock_audit", "echo", 24)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the synthetic reviewer-training kit (no patient data).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None, help="config/study.yaml, for training.kit_size")
    a = ap.parse_args(argv)
    kw = {}
    if a.config:
        from mcp_adapter.config import StudyConfig
        kw = sizes_from_config(StudyConfig.load(a.config))
    r = build_kit(a.out, seed=a.seed, **kw)
    print(f"wrote {r['n_records']} synthetic records and {r['n_clips']} synthetic clips to {r['out']}")


if __name__ == "__main__":
    main()
