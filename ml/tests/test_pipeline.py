"""End-to-end tests on a synthetic OMOP database (no real data)."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from mcp_adapter import concepts as C
from mcp_adapter.backend import DuckDBBackend
from mcp_adapter.config import StudyConfig
from mcp_adapter.features.repro import FEATURE_NAMES
from mcp_adapter.features.waveforms import N_SAMPLES, preprocess_ecg, read_muse_xml
from mcp_adapter.pipeline import build_study, run_feasibility
from mcp_adapter.privacy import suppress
from tests.synthetic_omop import load_into_duckdb

ROOT = Path(__file__).resolve().parents[1]


def _imaging(tmp: Path, cohort: pd.DataFrame, n: int = 40):
    """Synthetic ECG (.npy) and echo (.npy cine) files + manifests for the first n women."""
    rng = np.random.default_rng(0)
    ecg_rows, echo_rows = [], []
    (tmp / "ecg").mkdir(); (tmp / "echo").mkdir()
    t = np.arange(N_SAMPLES) / 500
    for pid, idx in cohort[["person_id", "index_date"]].head(n).itertuples(index=False):
        x = np.stack([np.sin(2 * np.pi * 1.2 * t + k) + .05 * rng.standard_normal(N_SAMPLES) for k in range(12)])
        np.save(tmp / "ecg" / f"{pid}.npy", x.astype(np.float32))
        ecg_rows.append({"person_id": pid, "path": f"{pid}.npy", "record_date": idx})
        for v in ("A4C", "PLAX"):
            np.save(tmp / "echo" / f"{pid}_{v}.npy", rng.random((20, 96, 128)).astype(np.float32))
            echo_rows.append({"person_id": pid, "path": f"{pid}_{v}.npy", "record_date": idx, "view": v})
    pd.DataFrame(ecg_rows).to_csv(tmp / "ecg_manifest.csv", index=False)
    pd.DataFrame(echo_rows).to_csv(tmp / "echo_manifest.csv", index=False)


@pytest.fixture(scope="session")
def study(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("study")
    con = duckdb.connect(); load_into_duckdb(con)
    be = DuckDBBackend(con)
    cfg = StudyConfig.load(ROOT / "config/study.yaml", dialect="duckdb", report_dir=str(tmp / "reports"),
                           imaging_required=False)   # first pass only: manifests need the cohort
    feas = run_feasibility(be, cfg)
    t = build_study(be, cfg, tmp / "tables")
    _imaging(tmp, t["cohort"])
    cfg.ecg_source.kind, cfg.ecg_source.path, cfg.ecg_source.manifest = "npy_dir", str(tmp / "ecg"), str(tmp / "ecg_manifest.csv")
    cfg.echo_source.kind, cfg.echo_source.path, cfg.echo_source.manifest = "dicom_dir", str(tmp / "echo"), str(tmp / "echo_manifest.csv")
    cfg.imaging_required = True
    t = build_study(be, cfg, tmp / "tables")              # rebuild with imaging manifests
    from mcp_adapter.pipeline import build_pretraining_pool
    pool = build_pretraining_pool(be, cfg, t["cohort"], tmp / "pool")
    return {"tmp": tmp, "cfg": cfg, "tables": t, "feas": feas, "con": con, "pool": pool}


def test_imaging_is_required(tmp_path):
    from mcp_adapter.pipeline import check_imaging_config
    cfg = StudyConfig.load(ROOT / "config/study.yaml")
    with pytest.raises(ValueError, match="imaging_required"):
        check_imaging_config(cfg)          # shipped config has TODO manifests -> must fail loudly


def test_pretraining_pool_excludes_eval_women(study):
    co, pool = study["tables"]["cohort"], study["pool"]["pool"]
    held = co[co.split.isin(["val", "test"])].person_id
    assert not pool.person_id.isin(held).any()
    assert pool.person_id.isin(co[co.split == "train"].person_id).any()
    assert len(study["pool"]["ecg_records"]) > 0 and len(study["pool"]["echo_records"]) > 0


# ------------------------------------------------------------------ cohort
def test_cohort_is_women_one_row_each(study):
    co = study["tables"]["cohort"]
    men = study["con"].execute("SELECT person_id FROM omop_cdm.person WHERE gender_concept_id = 8507").df()
    assert not co.person_id.isin(men.person_id).any()
    assert co.person_id.is_unique
    assert set(co.stratum) == {"A_pregnant_high_risk", "B_postpartum", "C_scad_fmd",
                               "D_postmeno_takotsubo_cmd", "E_pcos_endo_autoimmune", "F_prior_preeclampsia"}


def test_congenital_exclusion_uses_full_history(study):
    rep = pd.read_csv(Path(study["cfg"].report_dir) / "02_attrition.csv")
    n = pd.to_numeric(rep.n_women, errors="coerce")
    assert n.iloc[1] < n.iloc[0], "Q2x codes years before the study window must still exclude"


def test_lab_code_coverage_report(study):
    rep = pd.read_csv(Path(study["cfg"].report_dir) / "01b_lab_code_coverage.csv")
    assert {"item", "loinc", "n_rows", "n_women"} <= set(rep.columns)
    assert set(rep.item) >= {"bnp", "troponin_i", "creatinine"}
    zero = rep[rep.loinc == "6598-7"]                       # troponin T: absent in synthetic data
    assert str(zero.n_women.iloc[0]) == "0"


def test_attrition_is_monotone(study):
    rep = pd.read_csv(Path(study["cfg"].report_dir) / "02_attrition.csv")
    n = pd.to_numeric(rep.n_women[:6], errors="coerce").dropna().values
    assert (np.diff(n) <= 0).all()


# ------------------------------------------------------------------ leakage
def test_no_target_codes_in_ehr_inputs(study):
    from hcman.data import StudyData
    data = StudyData(study["tmp"] / "tables", study["cfg"])
    inv = {v: k for k, v in data.pre.ehr.vocab.items()}
    for arr in data.ehr.values():
        for tid in arr[:, 0].astype(int):
            tok = inv[tid]
            if tok.startswith("DX:"):
                assert not any(tok[3:].startswith(p) for p in C.LEAKY_CODE_PREFIXES), tok
        assert (arr[1:, 3] <= 0).all(), "EHR inputs must come from day <= 0"


def test_history_features_and_incident_labels_are_disjoint(study):
    df = study["tables"]["repro"].merge(study["tables"]["labels"], on="person_id")
    assert not ((df.fmd_history == 1) & (df.y_FMD == 1)).any()


def test_masks(study):
    lab = study["tables"]["labels"]
    rep = study["tables"]["repro"].set_index("person_id")
    pp = lab[lab.m_postpartum_cvd > 0].person_id
    assert (rep.loc[pp, "pregnancy_status=postpartum"] == 1).mean() > .9
    assert ((lab.y_atypical == 1) <= (lab.m_atypical == 1)).all()
    assert lab.y_fmd_trigger.equals(lab.y_SCAD)


def test_reports_are_suppressed(study):
    for f in Path(study["cfg"].report_dir).glob("*.csv"):
        df = pd.read_csv(f)
        for c in [c for c in df.columns if c.startswith("n_")]:
            v = pd.to_numeric(df[c], errors="coerce")
            assert not ((v > 0) & (v < study["cfg"].min_cell_size)).any(), f"{f.name}:{c}"


def test_suppress_complementary():
    df = pd.DataFrame({"g": list("abc"), "n_women": [5, 40, 100]})
    out = suppress(df, ["n_women"], 11)
    assert out.n_women.tolist() == ["<11", "<11", 100]


# ------------------------------------------------------------------ waveforms
def test_muse_xml_reader(tmp_path):
    n = 5000
    leads = {l: (np.sin(np.arange(n) / 50) * 200).astype("<i2") for l in ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]}
    body = "".join(f"<LeadData><LeadID>{l}</LeadID><LeadAmplitudeUnitsPerBit>4.88</LeadAmplitudeUnitsPerBit>"
                   f"<WaveFormData>{base64.b64encode(v.tobytes()).decode()}</WaveFormData></LeadData>"
                   for l, v in leads.items())
    p = tmp_path / "ecg.xml"
    p.write_text(f"<RestingECG><Waveform><WaveformType>Rhythm</WaveformType><SampleBase>500</SampleBase>{body}</Waveform></RestingECG>")
    x, fs = read_muse_xml(p)
    assert x.shape == (12, n) and fs == 500
    assert np.allclose(x[2], x[1] - x[0])                 # lead III derived
    y = preprocess_ecg(x, fs)
    assert y.shape == (12, N_SAMPLES) and np.isfinite(y).all()


# ------------------------------------------------------------------ model
def test_train_export_serve(study, tmp_path):
    torch = pytest.importorskip("torch")
    from hcman import train
    from hcman.export import export
    from hcman.model import HCMAN, HCMANConfig
    from hcman.serve import Predictor
    import shutil

    cfg_path = tmp_path / "study.yaml"
    import yaml
    raw = yaml.safe_load((ROOT / "config/study.yaml").read_text())
    raw["platform"]["dialect"] = "duckdb"
    raw["platform"]["ecg_source"] = {"kind": "npy_dir", "path": None, "manifest": None}
    raw["privacy"]["report_dir"] = str(tmp_path / "reports")
    raw["model"]["freeze_imaging_backbones_epochs"] = 1
    raw["pretraining"]["batch_size"] = 16
    cfg_path.write_text(yaml.safe_dump(raw))
    # first-time training: pre-train every encoder from scratch, then fine-tune from it
    from hcman import pretrain
    pre_dir = tmp_path / "pretrain"
    summ = pretrain.main(["--config", str(cfg_path), "--pool-dir", str(study["tmp"] / "pool"),
                          "--out", str(pre_dir), "--size", "small", "--epochs", "1", "--workers", "0"])
    assert summ["stages_done"] == ["ecg", "echo", "ehr", "labs"]
    run = tmp_path / "run"
    best = train.main(["--config", str(cfg_path), "--study-dir", str(study["tmp"] / "tables"),
                       "--out", str(run), "--size", "small", "--epochs", "3", "--batch-size", "16",
                       "--warmup", "10", "--workers", "0", "--n-boot", "50",
                       "--report-dir", str(tmp_path / "reports"), "--site-adv-weight", "0.1",
                       "--grad-accum", "2",
                       "--init-weights", str(pre_dir / "encoders.pt"),
                       "--preprocessors", str(pre_dir / "preprocessors.pkl")])
    assert best >= 0
    metrics = pd.read_csv(tmp_path / "reports" / "10_test_metrics.csv")
    assert set(C.CONDITION_ORDER) <= set(metrics.task)
    summary = json.loads((tmp_path / "reports" / "12_training_summary.json").read_text())
    assert summary["epochs_run"] >= 1

    ck = torch.load(run / "hcman_best.pt", map_location="cpu")
    mc = HCMANConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["model_config"].items()})
    m = HCMAN(mc); m.load_state_dict(ck["state_dict"])
    bundle = tmp_path / "bundle"
    export(m, bundle)
    for f in ("preprocessors.pkl", "thresholds.json"):
        shutil.copy(run / f, bundle / f)
    pred = Predictor(bundle).predict({
        "age": 33, "repro": {"pregnancy_status": "postpartum", "preeclampsia_history": True},
        "observations": [{"loinc": "30934-4", "value": 900, "unit": "pg/mL", "daysBeforeIndex": 2},
                         {"loinc": "8480-6", "value": 150, "unit": "mm[Hg]", "daysBeforeIndex": 0}]})
    assert set(pred["conditions"]) == set(C.CONDITION_ORDER)
    assert 0 <= pred["mace12m"] <= 1 and len(pred["riskTier"]) == 4
    assert pred["postpartumCvd"] is not None

    # (a) padding is exact: unpadded PyTorch == padded PyTorch; (b) ONNX == PyTorch
    import onnxruntime as ort
    from hcman.data import StudyData, collate
    from hcman.export import EHR_LEN, INPUTS, LABS_LEN, InferenceWrapper, pad_to
    from hcman.data import Preprocessors
    ds = StudyData(study["tmp"] / "tables", study["cfg"],
                   pre=Preprocessors.load(run / "preprocessors.pkl")).split("test", echo_frames=mc.echo_frames)
    b = collate([ds[i] for i in range(4)])
    w = InferenceWrapper(m)
    x = [b[k] for k in INPUTS]
    with torch.no_grad():
        ref = w(*x)
        e, em = pad_to(b["ehr"].numpy(), b["ehr_mask"].numpy(), EHR_LEN)
        l, lm = pad_to(b["labs"].numpy(), b["labs_mask"].numpy(), LABS_LEN)
        xp = dict(zip(INPUTS, x)); xp.update(ehr=torch.tensor(e), ehr_mask=torch.tensor(em),
                                             labs=torch.tensor(l), labs_mask=torch.tensor(lm))
        padded = w(*[xp[k] for k in INPUTS])
    for r, pdd in zip(ref, padded):
        assert torch.allclose(r, pdd, atol=1e-5)
    sess = ort.InferenceSession(str(bundle / "hcman.onnx"))
    got = sess.run(None, {k: v.numpy() for k, v in xp.items()})
    for r, g in zip(ref, got):
        assert np.allclose(r.numpy(), g, atol=1e-4)


def test_full_size_matches_form(study):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from hcman.model import HCMAN, HCMANConfig
    m = HCMAN(HCMANConfig(ehr_vocab=20000))
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"full HCMAN parameters: {n:.1f}M")
    assert n > 52, "form's ~52M understates a 12-layer d=768 ECG transformer alone (~85M)"
