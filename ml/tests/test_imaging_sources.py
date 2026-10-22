"""Tests for getting imaging into the workbench: DICOM request list and ECG-table extraction."""
from __future__ import annotations

import base64
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from mcp_adapter.backend import DuckDBBackend
from mcp_adapter.config import StudyConfig
from mcp_adapter.features.waveforms import N_SAMPLES, load_ecg
from mcp_adapter.imaging_sources import build_dicom_request, extract_ecg_table
from tests.synthetic_omop import load_into_duckdb

ROOT = Path(__file__).resolve().parents[1]
LEADS8 = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]


@pytest.fixture(scope="module")
def db():
    con = duckdb.connect(); load_into_duckdb(con)
    return con


def cfg_for(tmp_path, **kw):
    return StudyConfig.load(ROOT / "config/study.yaml", dialect="duckdb", report_dir=str(tmp_path / "rep"), **kw)


def test_request_needs_columns(db, tmp_path):
    cfg = cfg_for(tmp_path)
    cfg.dicom_inventory["patient_column"] = None
    with pytest.raises(ValueError, match="patient_column"):
        build_dicom_request(DuckDBBackend(db), cfg, tmp_path / "req.csv")


def test_dicom_request_list(db, tmp_path):
    """Inventory rows are DICOM instances keyed by Mayo clinic number, with no date; dates come
    from FACT_RADIOLOGY via the accession number (column names from the MCP Radiology deck)."""
    ev = db.execute("""SELECT p.person_id, MIN(p.procedure_date) AS d, pe.gender_concept_id AS g
                       FROM omop_cdm.procedure_occurrence p JOIN omop_cdm.person pe USING (person_id)
                       GROUP BY p.person_id, pe.gender_concept_id""").df()
    inv, rad = [], []
    for i, r in enumerate(ev.itertuples()):
        mc, d = f"MC{r.person_id:07d}", pd.Timestamp(r.d)
        def study(tag, modality, desc, status="Ready", date=None, n=3):
            uid = f"{tag}.{i}"
            for k in range(n):
                inv.append((mc, 900000 + r.person_id, uid, f"{uid}.s{k % 2}", f"{uid}.i{k}", f"A{uid}",
                            modality, "CARDIO", desc, status))
            if date is not None:
                rad.append((f"A{uid}", date))
        study("NEAR", "US", "TTE ECHO COMPLETE", date=d)                       # keep
        study("FAR", "US", "TTE ECHO COMPLETE", date=d + pd.Timedelta(days=200))   # outside window
        study("NODATE", "US", "ECHOCARDIOGRAM")                                 # undated: keep
        study("CT", "CT", "CT CHEST", date=d)                                   # wrong modality
        study("OB", "US", "OB ULTRASOUND", date=d)                              # not cardiac
        study("NA", "US", "TTE ECHO", status="Not Available", date=d)            # cannot be de-identified
    cols = ["PATIENT_CLINIC_NUMBER", "PATIENT_DK", "STUDY_INSTANCE_UID", "SERIES_INSTANCE_UID", "SOP_INSTANCE_UID",
            "DICOM_ACCESSION_NUMBER", "MODALITY", "STMO_NAME_IN_ADL", "STUDY_DESCRIPTION", "IMAGE_DEID_STATUS"]
    db.register("inv_df", pd.DataFrame(inv, columns=cols))
    db.execute("CREATE OR REPLACE TABLE FACT_RADIOLOGY_DICOM_INVENTORY AS SELECT * FROM inv_df")
    db.register("rad_df", pd.DataFrame(rad, columns=["ACCESSION_NBR", "RADIOLOGY_DTM"]))
    db.execute("CREATE OR REPLACE TABLE FACT_RADIOLOGY AS SELECT * FROM rad_df")

    cfg = cfg_for(tmp_path)                                    # shipped config, nothing overridden
    req = build_dicom_request(DuckDBBackend(db), cfg, tmp_path / "req.csv")
    kinds = set(req.study_uid.str.split(".").str[0])
    assert kinds == {"NEAR", "NODATE"}
    assert (req.n_instances == 3).all() and (req.n_series == 2).all()
    assert req[req.study_uid.str.startswith("NEAR")].dated.all()
    women = set(ev[ev.g == 8532].person_id)
    assert set(req.person_id) <= women and len(req)
    cfg.dicom_inventory["keep_undated"] = False
    req2 = build_dicom_request(DuckDBBackend(db), cfg, tmp_path / "req2.csv")
    assert set(req2.study_uid.str.split(".").str[0]) == {"NEAR"}
    assert (tmp_path / "rep" / "00_dicom_request.csv").exists()

    # the CSV MCP asked for: exactly these columns, one row per study / modality / STMO
    sub = pd.read_csv(tmp_path / "req_deid_submission.csv")
    assert list(sub.columns) == ["STUDY_INSTANCE_UID", "PATIENT_DK", "MODALITY", "STMO_NAME_IN_ADL"]
    assert sorted(sub.STUDY_INSTANCE_UID) == sorted(req.study_uid)
    assert (sub.MODALITY == "US").all() and (sub.STMO_NAME_IN_ADL == "CARDIO").all()
    assert not sub.duplicated().any()
    assert set(sub.PATIENT_DK) == {900000 + p for p in req.person_id}


def _muse(seed):
    v = (np.sin(np.arange(5000) / (40 + seed)) * 300).astype("<i2")
    body = "".join(f"<LeadData><LeadID>{l}</LeadID><LeadAmplitudeUnitsPerBit>4.88</LeadAmplitudeUnitsPerBit>"
                   f"<WaveFormData>{base64.b64encode(v.tobytes()).decode()}</WaveFormData></LeadData>" for l in LEADS8)
    return (f"<RestingECG><Waveform><WaveformType>Rhythm</WaveformType><SampleBase>500</SampleBase>"
            f"{body}</Waveform></RestingECG>"), v


@pytest.mark.parametrize("fmt", ["muse_xml", "lead_base64"])
def test_extract_ecg_table(db, tmp_path, fmt):
    pids = db.execute("SELECT person_id FROM omop_cdm.person LIMIT 3").df().person_id.tolist()
    rows = []
    for k, pid in enumerate(pids):
        xml, v = _muse(k)
        if fmt == "muse_xml":
            payload = xml if k % 2 == 0 else base64.b64encode(xml.encode()).decode()   # both encodings
            rows.append((f"E{k}", pid, "2020-01-01", payload, None))
        else:
            rows += [(f"E{k}", pid, "2020-01-01", base64.b64encode(v.tobytes()).decode(), l) for l in LEADS8]
    rows.append(("BAD", pids[0], "2020-01-02", "not an ecg", "I"))                    # skipped, counted
    t = pd.DataFrame(rows, columns=["ecg_id", "pat", "acq_dt", "wave", "lead"])
    db.register("ecg_df", t); db.execute("CREATE OR REPLACE TABLE ecg_wave AS SELECT * FROM ecg_df")
    cfg = cfg_for(tmp_path)
    cfg.ecg_table.update(table="ecg_wave", format=fmt, person_column="pat", date_column="acq_dt",
                         id_column="ecg_id", waveform_column="wave", lead_column="lead")
    man = extract_ecg_table(DuckDBBackend(db), cfg, tmp_path / "ecg", person_ids=pids)
    assert len(man) == 3 and (tmp_path / "ecg" / "manifest.csv").exists()
    x = load_ecg(man.path.iloc[0], "npy_dir")
    assert x.shape == (12, N_SAMPLES) and np.isfinite(x).all()
    rep = pd.read_csv(tmp_path / "rep" / "00_ecg_extraction.csv")
    assert set(rep.item) == {"ECGs extracted", "ECGs unreadable"}


def test_manifest_from_delivered_files(tmp_path):
    from tests.test_annotate import make_clip, write_dicom
    import pydicom
    from mcp_adapter.imaging_sources import manifest_from_folder
    req = pd.DataFrame({"study_uid": ["1.2.3", "1.2.4"], "person_id": [11, 12],
                        "study_date": ["2020-01-01", None]})              # second one undated
    req.to_csv(tmp_path / "req.csv", index=False)
    files = tmp_path / "delivered"; files.mkdir()
    for k, uid in enumerate(["1.2.3", "1.2.4", "9.9.9"]):               # last one was not requested
        fr, _ = make_clip("A4C", n=20, size=64, seed=k)
        p = files / f"f{k}.dcm"
        write_dicom(p, fr, desc="A4C")
        ds = pydicom.dcmread(p); ds.StudyInstanceUID = uid; ds.StudyDate = "20200315"
        ds.save_as(p, enforce_file_format=True)
    (files / "notes.txt").write_text("not dicom")
    man = manifest_from_folder(files, tmp_path / "req.csv", tmp_path / "man.csv")
    assert sorted(man.person_id) == [11, 12]
    assert man.set_index("person_id").record_date.to_dict() == {11: "2020-01-01", 12: "2020-03-15"}
    assert set(man.columns) >= {"person_id", "record_date", "path"}


def test_extract_mayo_fact_ecg_waveform(db, tmp_path):
    """FACT_ECG_WAVEFORM as MCP described it (21 Oct 2026): URL-encoded Base64 int16 per lead,
    Rhythm vs Median rows, PATIENT_DK -> DIM_PATIENT -> clinic number, rate from a measurement table."""
    from urllib.parse import quote
    from mcp_adapter.features.waveforms import decode_waveform
    people = db.execute("SELECT person_id, person_source_value FROM omop_cdm.person LIMIT 4").df()
    enc = lambda v: quote(base64.b64encode(v.astype("<i2").tobytes()).decode(), safe="")
    sig = lambda n, k: (np.sin(np.arange(n) / (30 + k)) * 400).astype("<i2")
    rows, dims, meas = [], [], []
    for k, r in enumerate(people.itertuples()):
        dk = 7000 + k
        dims.append((dk, r.person_source_value, "N"))
        fs = 250 if k == 1 else 500
        acq = f"2020-03-0{k + 1} 10:15:00"
        for l in LEADS8:
            rows.append((dk, acq, l, "Rhythm", enc(sig(fs * 10, k))))
            rows.append((dk, acq, l, "Median", enc(sig(600, k))))                 # never used
        if k != 2:
            meas.append((dk, acq, fs))                                           # k = 2: rate inferred from length
    dims.append((7100, people.person_source_value[0], "Y"))                     # same woman, second system
    rows += [(7100, "2020-03-01 10:15:00", l, "Rhythm", enc(sig(5000, 0))) for l in LEADS8]
    rows += [(7003, "2020-05-05 09:00:00", l, "Rhythm", enc(sig(1200, 3))) for l in LEADS8]  # two partial
    rows += [(7003, "2020-05-05 09:00:00", l, "Rhythm", enc(sig(1300, 3))) for l in LEADS8]  # records: skip
    rows.append((7002, "2020-06-06 08:00:00", "I", "Rhythm", "%%%not-base64"))  # unreadable
    db.register("w_df", pd.DataFrame(rows, columns=["PATIENT_DK", "ACQUISITION_TS", "LEAD_ID", "WAVEFORM_TYPE", "WAVEFORM"]))
    db.register("d_df", pd.DataFrame(dims, columns=["PATIENT_DK", "PATIENT_CLINIC_NUMBER", "PATIENT_MERGED_FLAG"]))
    db.register("m_df", pd.DataFrame(meas, columns=["PATIENT_DK", "ACQUISITION_TS", "ECGSAMPLEBASE"]))
    for t, src in (("fact_ecg_waveform", "w_df"), ("dim_patient", "d_df"), ("fact_ecg_meas", "m_df")):
        db.execute(f"CREATE OR REPLACE TABLE {t} AS SELECT * FROM {src}")
    cfg = cfg_for(tmp_path)
    assert cfg.ecg_table["format"] == "mayo_waveform" and cfg.dim_patient["table"] == "DIM_PATIENT"
    cfg.ecg_table.update(table="fact_ecg_waveform", sample_rate_table="fact_ecg_meas")
    cfg.dim_patient.update(table="dim_patient")
    man = extract_ecg_table(DuckDBBackend(db), cfg, tmp_path / "ecg", person_ids=people.person_id.tolist())
    assert len(man) == 4                                   # 4 women; the second-system copy is not double-counted
    assert sorted(man.fs.tolist()) == [250, 500, 500, 500]
    for p in man.path:
        x = load_ecg(p, "npy_dir")
        assert x.shape == (12, N_SAMPLES) and np.isfinite(x).all()
    raw = np.load(man.path.iloc[0]); assert raw["x"].shape[0] == 12
    rep = pd.read_csv(tmp_path / "rep" / "00_ecg_extraction.csv").set_index("item")
    assert "ECGs skipped: several records per lead, none the full 10 s" in rep.index
    v = np.array([-3, 0, 32767, -32768], dtype="<i2")
    assert (decode_waveform(enc(v)) == v).all()                                # URL + Base64 round trip
