# ml/ — HCMAN women's CVD cohort study on Mayo Clinic Platform

Code for training **HCMAN** (Hierarchical Cross-Modal Attention Network, Cardio AI) on de-identified
Mayo Clinic Platform (MCP) data, and for serving it to the CardioAI app in this repository
(`src/server/services/hcmanAdapter.ts`).

**Training on MCP? Follow [RUNBOOK.md](RUNBOOK.md)** — the step-by-step guide with commands,
go / no-go gates, troubleshooting and a run checklist. This README is the short overview.

```
config/study.yaml        every study parameter (windows, thresholds, sources, splits)
mcp_adapter/             OMOP CDM v5.4 -> HCMAN inputs (runs inside the MCP workbench)
  concepts.py            ICD-10-CM / LOINC / RxNorm / CPT4 concept sets  <- clinical review
  queries.py             portable Spark SQL / DuckDB extraction (two passes)
  cohort.py              six strata, index date, eligibility, attrition
  labels.py              six heads + masks (incident conditions, MACE with blanking, ...)
  features/repro.py      stream 4 (15 women-specific features, with provenance)
  features/labs_ehr.py   streams 3 and 5 (tokenizer + serial labs, train-fit only)
  features/waveforms.py  streams 1 and 2 (MUSE XML / npy ECG, DICOM echo)
  splits.py, privacy.py  patient-level splits; small-cell suppression for every report
  pipeline.py            run_feasibility(), build_study(), build_pretraining_pool()
  imaging_sources.py     DICOM request list from the MCP inventory, manifests for delivered files, ECG-table extraction
hcman/                   PyTorch model, pre-training, training, evaluation, export, serving
annotate/                echo view + ED/ES auto-annotation, Jupyter review tool, chart review of labels
tests/                   synthetic OMOP database + end-to-end tests
```

## Run order inside the MCP workbench (first-time training)

There is no prior model or training data: every encoder is pre-trained from scratch inside MCP,
then the full model is fine-tuned on the cohort. ECG and echo imaging are in scope under the
Cardio AI MCP contract and are required (`imaging_required: true`).

1. **Submit this repository for MCP code review** (MCP whitelists repositories before use).
2. **Point the config at the data**: in `config/study.yaml` set `cdm_schema` / `vocab_schema`
   (Schema Visualizer) and the ECG and echo `manifest` / `path` (the TODO(MCP) lines) to the
   tables or folders MCP gives you. `build_study` refuses to run until imaging is wired.
3. **Feasibility** (aggregate only):
   ```python
   from mcp_adapter.backend import SparkBackend
   from mcp_adapter.config import StudyConfig
   from mcp_adapter.pipeline import run_feasibility, build_study, build_pretraining_pool
   cfg = StudyConfig.load("config/study.yaml")
   be = SparkBackend(spark)            # the session the MCP notebook provides
   run_feasibility(be, cfg)            # -> reports/01_feasibility_counts.csv (suppressed)
   ```
   Check the feasibility gates in protocol §6 before going further.
4. **Build the study tables** (row-level, stay in the workbench):
   `t = build_study(be, cfg, "/workbench/hcman_study")` — reports 02–06.
5. **Build the pre-training pool** (women with cardiac evaluations, minus the cohort's
   validation/test women): `build_pretraining_pool(be, cfg, t["cohort"], "/workbench/hcman_pool")` — report 07.
6. **Pre-train from scratch**:
   `python -m hcman.pretrain --config config/study.yaml --pool-dir /workbench/hcman_pool --out /workbench/pretrain`
   (ECG + echo SimCLR, EHR masked tokens, labs masked steps) — report 08.
7. **Fine-tune**:
   ```
   python -m hcman.train --config config/study.yaml --study-dir /workbench/hcman_study \
       --out /workbench/run1 --init-weights /workbench/pretrain/encoders.pt \
       --preprocessors /workbench/pretrain/preprocessors.pkl
   ```
   Add `--no-imaging` for the EHR-only ablation. Reports 10–12 (test metrics with CIs,
   fairness, training summary).
8. **Request export** of `reports/` (aggregate, suppressed) and, via the Model Details Form, the
   model bundle: `python -m hcman.export --run /workbench/run1 --out /workbench/hcman_export`.

## Local testing (no MCP access needed)

```bash
cd ml
pip install -r requirements.txt
python -m pytest tests -q            # builds a synthetic OMOP DB in DuckDB, runs the whole chain
```

## Serving to the CardioAI app

```bash
cd ml && python -m hcman.serve --bundle hcman_export --port 8080
# in the app: ASSESSMENT_ENGINE=hcman  HCMAN_INFERENCE_URL=http://<host>:8080
```
The app falls back to its rules engine whenever the service is unset or unreachable.

Synthetic data only in `tests/`. Nothing here is a validated medical device.
