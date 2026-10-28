"""Getting imaging data into the workbench on Mayo Clinic Platform.

Per MCP (14-15 Oct 2026):
  * DICOM files are not in a table. They are inventoried in FACT_RADIOLOGY_DICOM_INVENTORY and
    must be *requested*. `build_dicom_request` makes the request list: echo studies (and any
    other configured modalities) for women with a cardiac evaluation, within days of it.
  * ECG waveforms may be stored as encoded strings in a table. `extract_ecg_table` converts
    them once into .npz files plus a manifest, which the pipeline then reads like any file source.

Column names are not known until the MCP radiology / data training, so every column comes from
config/study.yaml (`dicom_inventory:` and `ecg_table:`). Both functions refuse to run with a
TODO left, naming the missing setting.
"""
from __future__ import annotations

import re

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import concepts as C
from . import queries as Q
from .backend import Backend, days_between, regex_match
from .config import StudyConfig
from .features.waveforms import leads_from_base64, parse_muse_xml
from .privacy import suppress, write_report

log = logging.getLogger("hcman.imaging")


def _require(section: str, src: dict, keys: tuple[str, ...]) -> None:
    missing = [k for k in keys if not src.get(k)]
    if missing:
        raise ValueError(f"set {', '.join(f'platform.{section}.{k}' for k in missing)} in config/study.yaml")


# ------------------------------------------------------------------ patient key mapping
def person_join(cfg: StudyConfig, alias: str, key_expr: str) -> str:
    """SQL join mapping a Mayo clinic number (PATIENT_CLINIC_NUMBER) to OMOP person_id through the
    configured person column. Adds `{alias}.person_id`."""
    col = (cfg.person_map or {}).get("clinic_number_column")
    if not col:
        raise ValueError("set platform.person_map.clinic_number_column in config/study.yaml")
    return (f"JOIN {cfg.cdm_schema}.person {alias} "
            f"ON CAST({alias}.{col} AS STRING) = CAST({key_expr} AS STRING)")


# ------------------------------------------------------------------ DICOM request list
SUBMISSION_COLUMNS = ["STUDY_INSTANCE_UID", "PATIENT_DK", "MODALITY", "STMO_NAME_IN_ADL"]


def build_dicom_request(be: Backend, cfg: StudyConfig, out_csv: str | Path,
                        exclude_person_ids=None, submission_csv: str | Path | None = None) -> pd.DataFrame:
    """Study-level request list from FACT_RADIOLOGY_DICOM_INVENTORY (one row per DICOM instance).

    Keeps instances that can be de-identified, of the configured modality and study description,
    belonging to women with a cardiac evaluation; groups them by study, modality and storage
    segment (STMO), the key MCP joins on; dates each study (FACT_RADIOLOGY via accession number by
    default) and keeps those within `window_days` of a cardiac evaluation, plus undated studies
    when `keep_undated` is set.

    Writes two files: the working list (`out_csv`, with person_id, dates and counts, for the
    manifest step) and the de-identification submission MCP asks for (`submission_csv`, default
    `<out_csv stem>_deid_submission.csv`): exactly STUDY_INSTANCE_UID, PATIENT_DK, MODALITY,
    STMO_NAME_IN_ADL, one row per study / modality / STMO. Plus a suppressed count report."""
    inv = cfg.dicom_inventory
    _require("dicom_inventory", inv, ("table", "patient_column", "study_uid_column", "patient_dk_column",
                                      "modality_column", "stmo_column"))
    evals = Q.cardiac_eval_events(be, cfg)[["person_id", "event_date"]].drop_duplicates()
    if exclude_person_ids is not None:
        evals = evals[~evals.person_id.isin(set(exclude_person_ids))]
    be.register(pd.DataFrame({"person_id": evals.person_id.unique()}), "hcman_eval_people")

    q = lambda v: "'" + str(v).replace("'", "''") + "'"
    cond = []
    if inv.get("deid_status_column") and inv.get("deid_ok"):
        cond.append(f"i.{inv['deid_status_column']} IN ({', '.join(q(v) for v in inv['deid_ok'])})")
    if inv.get("modality_column") and inv.get("modalities"):
        cond.append(f"UPPER(i.{inv['modality_column']}) IN ({', '.join(q(m.upper()) for m in inv['modalities'])})")
    if inv.get("description_column") and inv.get("description_pattern"):
        col = f"UPPER(i.{inv['description_column']})"
        cond.append(regex_match(cfg.dialect, col, inv["description_pattern"]))
        if inv.get("description_exclude"):       # e.g. fetal echo, TEE: not the TTE views the model reads
            cond.append(f"NOT {regex_match(cfg.dialect, f'COALESCE({col}, {chr(39)}{chr(39)})', inv['description_exclude'])}")
    keys = {"modality": inv["modality_column"], "stmo": inv["stmo_column"], "patient_dk": inv["patient_dk_column"]}
    opt = {k: inv.get(f"{k}_column") for k in ("accession", "description")}
    sel = "".join(f", i.{c} AS {k}" for k, c in keys.items())
    sel += "".join(f", MAX(i.{c}) AS {k}" for k, c in opt.items() if c)
    group = ", ".join(f"i.{c}" for c in keys.values())
    nser = f", COUNT(DISTINCT i.{inv['series_uid_column']}) AS n_series" if inv.get("series_uid_column") else ""
    inv_date = (f", MIN(CAST(i.{inv['date_column']} AS DATE)) AS study_date"
                if inv.get("date_source") == "inventory" and inv.get("date_column") else "")
    studies = be.sql(f"""
SELECT i.{inv['study_uid_column']} AS study_uid, pm.person_id, COUNT(*) AS n_instances{nser}{sel}{inv_date}
FROM {inv['table']} i {person_join(cfg, 'pm', 'i.' + inv['patient_column'])}
JOIN hcman_eval_people e ON e.person_id = pm.person_id
{'WHERE ' + ' AND '.join(cond) if cond else ''}
GROUP BY i.{inv['study_uid_column']}, pm.person_id, {group}""")

    if inv.get("date_source") == "fact_radiology" and "accession" in studies and len(studies):
        be.register(studies[["study_uid", "accession"]].dropna(), "hcman_req_acc")
        dates = be.sql(f"""
SELECT a.study_uid, MIN(CAST(r.{inv['date_column']} AS DATE)) AS study_date
FROM hcman_req_acc a JOIN {inv['date_table']} r
  ON CAST(r.{inv['date_accession_column']} AS STRING) = CAST(a.accession AS STRING)
GROUP BY a.study_uid""")
        studies = studies.merge(dates, on="study_uid", how="left")
    if "study_date" not in studies:
        studies["study_date"] = pd.NaT

    studies["study_date"] = pd.to_datetime(studies.study_date)
    lo, hi = inv.get("window_days", [-30, 2])
    m = studies.merge(evals.assign(event_date=pd.to_datetime(evals.event_date)), on="person_id")
    d = (m.study_date - m.event_date).dt.days
    in_window = set(m.loc[d.between(lo, hi), "study_uid"])
    undated = studies.study_date.isna()
    keep = studies.study_uid.isin(in_window) | (undated & bool(inv.get("keep_undated", True)))
    df = studies[keep].assign(dated=~undated[keep]).reset_index(drop=True)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    sub = (df.rename(columns={"study_uid": "STUDY_INSTANCE_UID", "patient_dk": "PATIENT_DK",
                              "modality": "MODALITY", "stmo": "STMO_NAME_IN_ADL"})[SUBMISSION_COLUMNS]
             .drop_duplicates().sort_values(SUBMISSION_COLUMNS))
    submission_csv = Path(submission_csv or Path(out_csv).with_name(Path(out_csv).stem + "_deid_submission.csv"))
    sub.to_csv(submission_csv, index=False)
    log.info("de-identification submission for the TSM: %d rows -> %s", len(sub), submission_csv)
    rep = pd.DataFrame([{"item": "studies", "n": df.study_uid.nunique()},
                        {"item": "studies with a date", "n": int(df.dated.sum())},
                        {"item": "women", "n": df.person_id.nunique()},
                        {"item": "instances", "n": int(df.n_instances.sum())}])
    write_report(suppress(rep, ["n"], cfg.min_cell_size), "00_dicom_request", cfg.report_dir)
    log.info("request list: %d studies (%d dated) for %d women -> %s",
             len(df), int(df.dated.sum()), df.person_id.nunique(), out_csv)
    return df


# ------------------------------------------------------------------ ECG waveforms from a table
def extract_ecg_table(be: Backend, cfg: StudyConfig, out_dir: str | Path, person_ids=None,
                      batch_people: int = 2000) -> pd.DataFrame:
    """Convert ECGs stored as encoded strings into .npz files (raw mV + sampling rate) and write
    `manifest.csv` (person_id, record_date, path). Point ecg_source.manifest at that file.

    format: muse_xml      — one row per ECG, the column holds MUSE XML text or base64 of it
    format: lead_base64   — one row per lead, base64 int16 samples (MUSE WaveFormData style)
    format: mayo_waveform — Mayo FACT_ECG_WAVEFORM (MCP, 21 Oct 2026): rows keyed by PATIENT_DK +
                            ACQUISITION_TS, one URL-encoded Base64 int16 record per lead, only
                            WAVEFORM_TYPE = Rhythm, leads I, II, V1-V6; women are found through
                            DIM_PATIENT (PATIENT_DK -> PATIENT_CLINIC_NUMBER) and person_map;
                            sampling rate from FACT_ECG_WAVEFORM_RESTING_MEASUREMENT.ECGSAMPLEBASE
    """
    src = cfg.ecg_table
    if src.get("format") == "mayo_waveform":
        return _extract_mayo_waveform(be, cfg, out_dir, person_ids, batch_people)
    _require("ecg_table", src, ("table", "person_column", "date_column", "id_column", "waveform_column"))
    fmt = src.get("format", "muse_xml")
    if fmt == "lead_base64":
        _require("ecg_table", src, ("lead_column",))
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if person_ids is None:
        person_ids = Q.cardiac_eval_events(be, cfg).person_id.unique()
    person_ids = list(pd.unique(pd.Series(person_ids)))
    rows, failed = [], 0
    for s in range(0, len(person_ids), batch_people):
        chunk = pd.DataFrame({"person_id": person_ids[s:s + batch_people]})
        be.register(chunk, "hcman_ecg_people")
        lead = f", t.{src['lead_column']} AS lead" if fmt == "lead_base64" else ""
        fs_col = (f", t.{src['sample_rate_column']} AS fs"            # a rate column in the same table
                  if src.get("sample_rate_column") and not src.get("sample_rate_table") else "")
        df = be.sql(f"""
SELECT t.{src['id_column']} AS ecg_id, t.{src['person_column']} AS person_id,
       CAST(t.{src['date_column']} AS DATE) AS record_date, t.{src['waveform_column']} AS payload{lead}{fs_col}
FROM {src['table']} t JOIN hcman_ecg_people p ON p.person_id = t.{src['person_column']}""")
        for ecg_id, g in df.groupby("ecg_id"):
            try:
                if fmt == "muse_xml":
                    x, fs = parse_muse_xml(g.payload.iloc[0])
                else:
                    x = leads_from_base64(g.lead, g.payload, float(src.get("units_per_bit", 4.88)))
                    fs = int(g.fs.iloc[0]) if "fs" in g else int(src.get("sample_rate", 500))
            except Exception as e:                              # skip, count, keep going
                failed += 1
                log.debug("ECG %s skipped: %s", ecg_id, e)
                continue
            path = out / f"{ecg_id}.npz"
            np.savez_compressed(path, x=x.astype(np.float32), fs=fs)
            rows.append({"person_id": g.person_id.iloc[0], "record_date": g.record_date.iloc[0], "path": str(path)})
    man = pd.DataFrame(rows, columns=["person_id", "record_date", "path"])
    man.to_csv(out / "manifest.csv", index=False)
    write_report(suppress(pd.DataFrame([{"item": "ECGs extracted", "n": len(man)},
                                        {"item": "ECGs unreadable", "n": failed}]), ["n"], cfg.min_cell_size),
                 "00_ecg_extraction", cfg.report_dir)
    log.info("extracted %d ECGs (%d unreadable) -> %s", len(man), failed, out / "manifest.csv")
    return man


def _extract_mayo_waveform(be: Backend, cfg: StudyConfig, out_dir, person_ids, batch_people) -> pd.DataFrame:
    from .features.waveforms import AmbiguousECG, leads_from_records
    src, dim = cfg.ecg_table, cfg.dim_patient
    _require("ecg_table", src, ("table", "person_column", "date_column", "waveform_column", "lead_column",
                                "type_column", "rhythm_value"))
    _require("dim_patient", dim, ("table", "dk_column", "clinic_number_column"))
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if person_ids is None:
        person_ids = Q.cardiac_eval_events(be, cfg).person_id.unique()
    person_ids = list(pd.unique(pd.Series(person_ids)))
    w, dk, ts = src["table"], src["person_column"], src["date_column"]
    fs_join, fs_col = "", ""
    if src.get("sample_rate_table") and src.get("sample_rate_column"):
        keys = src.get("sample_rate_keys") or [dk, ts]
        on = " AND ".join(f"m.{k} = w.{k}" for k in keys)
        fs_join = (f"LEFT JOIN (SELECT {', '.join(keys)}, MAX({src['sample_rate_column']}) AS fs "
                   f"FROM {src['sample_rate_table']} GROUP BY {', '.join(keys)}) m ON {on}")
        fs_col = ", m.fs AS fs"
    rows, failed, ambiguous, seen = [], 0, 0, set()
    for s0 in range(0, len(person_ids), batch_people):
        be.register(pd.DataFrame({"person_id": person_ids[s0:s0 + batch_people]}), "hcman_ecg_people")
        df = be.sql(f"""
SELECT w.{dk} AS dk, CAST(w.{ts} AS STRING) AS acq, p.person_id, CAST(w.{ts} AS DATE) AS record_date,
       w.{src['lead_column']} AS lead, w.{src['waveform_column']} AS payload{fs_col}
FROM {w} w
JOIN {dim['table']} d ON d.{dim['dk_column']} = w.{dk}
{person_join(cfg, 'p', f"d.{dim['clinic_number_column']}")}
JOIN hcman_ecg_people b ON b.person_id = p.person_id
{fs_join}
WHERE w.{src['type_column']} = '{src['rhythm_value']}'""")
        for (pid, acq), g in df.groupby(["person_id", "acq"]):      # one acquisition per woman and time
            if (pid, acq) in seen:
                continue                                             # same ECG under a second PATIENT_DK
            seen.add((pid, acq))
            g = g.drop_duplicates(["lead", "payload"])
            fs = g.fs.dropna().iloc[0] if "fs" in g and g.fs.notna().any() else None
            try:
                x, fs = leads_from_records(g.lead, g.payload, fs, float(src.get("units_per_bit", 4.88)),
                                           bool(src.get("url_encoded", True)))
            except AmbiguousECG as e:
                ambiguous += 1; log.debug("ECG %s %s ambiguous: %s", pid, acq, e); continue
            except Exception as e:                                   # skip, count, keep going
                failed += 1; log.debug("ECG %s %s skipped: %s", pid, acq, e); continue
            name = re.sub(r"[^0-9A-Za-z]+", "", f"{pid}_{acq}")
            path = out / f"{name}.npz"
            np.savez_compressed(path, x=x.astype(np.float32), fs=fs)
            rows.append({"person_id": pid, "record_date": g.record_date.iloc[0], "path": str(path), "fs": fs})
    man = pd.DataFrame(rows, columns=["person_id", "record_date", "path", "fs"])
    man.to_csv(out / "manifest.csv", index=False)
    write_report(suppress(pd.DataFrame([{"item": "ECGs extracted", "n": len(man)},
                                        {"item": "ECGs unreadable", "n": failed},
                                        {"item": "ECGs skipped: several records per lead, none the full 10 s", "n": ambiguous},
                                        {"item": "ECGs at 250 Hz (resampled to 500)", "n": int((man.fs == 250).sum())}]),
                          ["n"], cfg.min_cell_size), "00_ecg_extraction", cfg.report_dir)
    log.info("extracted %d ECGs (%d unreadable, %d ambiguous) -> %s", len(man), failed, ambiguous, out / "manifest.csv")
    return man


# ------------------------------------------------------------------ manifest from delivered files
def manifest_from_folder(root: str | Path, request_csv: str | Path, out_csv: str | Path,
                         modality: str | None = "US") -> pd.DataFrame:
    """Once requested DICOMs are delivered into the workspace, link each file back to its woman
    and date through the request list (StudyInstanceUID -> person_id, study_date). Reads headers
    only, not pixels. Output columns: person_id, record_date, path (+ series description), ready
    for echo_source.manifest (views are added by annotate.echo) or ecg_source.manifest."""
    import pydicom
    req = pd.read_csv(request_csv, dtype={"study_uid": str})
    req = req.drop_duplicates("study_uid").set_index("study_uid")
    rows, skipped = [], 0
    for f in Path(root).rglob("*"):
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:
            skipped += 1
            continue
        uid = str(getattr(ds, "StudyInstanceUID", ""))
        if uid not in req.index or (modality and str(getattr(ds, "Modality", "")).upper() != modality):
            skipped += 1
            continue
        date = req.at[uid, "study_date"] if "study_date" in req else None
        if date is None or pd.isna(date):                     # undated in the request list
            sd = str(getattr(ds, "StudyDate", "") or "")
            date = f"{sd[:4]}-{sd[4:6]}-{sd[6:8]}" if len(sd) == 8 else None
        rows.append({"person_id": req.at[uid, "person_id"], "record_date": date,
                     "path": str(f), "study_uid": uid,
                     "series_description": str(getattr(ds, "SeriesDescription", "") or "")})
    man = pd.DataFrame(rows, columns=["person_id", "record_date", "path", "study_uid", "series_description"])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    man.to_csv(out_csv, index=False)
    log.info("manifest: %d files linked, %d skipped -> %s", len(man), skipped, out_csv)
    return man


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m mcp_adapter.imaging_sources")
    ap.add_argument("cmd", choices=["request", "ecg", "manifest"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="request/manifest: CSV path; ecg: output folder")
    ap.add_argument("--files", help="manifest: folder the requested DICOMs were delivered to")
    ap.add_argument("--request", help="manifest: the request list CSV made by `request`")
    ap.add_argument("--modality", default="US", help="manifest: keep only this DICOM modality ('' = all)")
    ap.add_argument("--submission", help="request: path of the CSV for the TSM "
                                         "(default <out stem>_deid_submission.csv)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if a.cmd == "manifest":
        manifest_from_folder(a.files, a.request, a.out, a.modality or None)
        return
    from .backend import make_backend
    cfg = StudyConfig.load(a.config)
    be = make_backend(cfg)
    if a.cmd == "request":
        build_dicom_request(be, cfg, a.out, submission_csv=a.submission)
    else:
        extract_ecg_table(be, cfg, a.out)


if __name__ == "__main__":
    main()
