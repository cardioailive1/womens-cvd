"""Study configuration loaded from config/study.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SourceConfig:
    kind: str = "none"
    path: str | None = None
    manifest: str | None = None


@dataclass
class StudyConfig:
    dialect: str = "spark"
    cdm_schema: str = "omop_cdm"
    vocab_schema: str = "omop_cdm"
    results_schema: str | None = None
    imaging_required: bool = True
    dicom_inventory: dict = field(default_factory=dict)
    person_map: dict = field(default_factory=dict)
    ecg_table: dict = field(default_factory=dict)
    ecg_source: SourceConfig = field(default_factory=SourceConfig)
    echo_source: SourceConfig = field(default_factory=SourceConfig)

    study_start: str = "2010-01-01"
    study_end: str = "2025-12-31"
    min_age: int = 18
    max_age_pregnancy_strata: int = 55
    lookback_days: int = 365
    ecg_echo_window_days: tuple[int, int] = (-30, 2)
    labs_window_days: int = 90
    label_window_days: int = 90
    outcome_window_days: int = 365
    outcome_blanking_days: int = 7
    min_prior_observation_days: int = 365
    postpartum_max_days: int = 182
    condition_label_rule: str = "two_codes_or_inpatient"

    stratum_targets: dict[str, int] = field(default_factory=dict)
    take_all_available: bool = True
    max_per_stratum: int | None = None

    split_scheme: str = "site_and_time"
    test_fraction: float = 0.15
    val_fraction: float = 0.15
    temporal_holdout_after_day: int | None = None
    seed: int = 20261013

    min_cell_size: int = 11
    report_dir: str = "reports"

    condition_head: str = "multilabel"
    model_size: str = "full"
    freeze_imaging_backbones_epochs: int = 5

    report_sources: dict[str, dict] = field(default_factory=dict)
    audit: dict = field(default_factory=dict)

    pretrain: dict[str, int] = field(default_factory=lambda: {
        "max_women": 200000, "ecg_epochs": 30, "echo_epochs": 30, "ehr_epochs": 20,
        "labs_epochs": 20, "batch_size": 256})

    @classmethod
    def load(cls, path: str | Path, **overrides: Any) -> "StudyConfig":
        raw = yaml.safe_load(Path(path).read_text())
        p, s, smp, sp, pv, m, pt = (raw.get(k, {}) or {} for k in
                                    ("platform", "study", "sampling", "splits", "privacy", "model",
                                     "pretraining"))
        cfg = cls(
            dialect=p.get("dialect", "spark"),
            cdm_schema=p.get("cdm_schema", "omop_cdm"),
            vocab_schema=p.get("vocab_schema", p.get("cdm_schema", "omop_cdm")),
            results_schema=p.get("results_schema"),
            imaging_required=p.get("imaging_required", True),
            dicom_inventory=p.get("dicom_inventory") or {},
            person_map=p.get("person_map") or {},
            ecg_table=p.get("ecg_table") or {},
            ecg_source=SourceConfig(**(p.get("ecg_source") or {})),
            echo_source=SourceConfig(**(p.get("echo_source") or {})),
            study_start=s.get("study_start", "2010-01-01"),
            study_end=s.get("study_end", "2025-12-31"),
            min_age=s.get("min_age", 18),
            max_age_pregnancy_strata=s.get("max_age_pregnancy_strata", 55),
            lookback_days=s.get("lookback_days", 365),
            ecg_echo_window_days=tuple(s.get("ecg_echo_window_days", (-30, 2))),
            labs_window_days=s.get("labs_window_days", 90),
            label_window_days=s.get("label_window_days", 90),
            outcome_window_days=s.get("outcome_window_days", 365),
            outcome_blanking_days=s.get("outcome_blanking_days", 7),
            min_prior_observation_days=s.get("min_prior_observation_days", 365),
            postpartum_max_days=s.get("postpartum_max_days", 182),
            condition_label_rule=s.get("condition_label_rule", "two_codes_or_inpatient"),
            stratum_targets=smp.get("stratum_targets", {}),
            take_all_available=smp.get("take_all_available", True),
            max_per_stratum=smp.get("max_per_stratum"),
            split_scheme=sp.get("scheme", "site_and_time"),
            test_fraction=sp.get("test_fraction", 0.15),
            val_fraction=sp.get("val_fraction", 0.15),
            temporal_holdout_after_day=sp.get("temporal_holdout_after_day"),
            seed=sp.get("seed", 20261013),
            min_cell_size=pv.get("min_cell_size", 11),
            report_dir=pv.get("report_dir", "reports"),
            condition_head=m.get("condition_head", "multilabel"),
            model_size=m.get("size", "full"),
            freeze_imaging_backbones_epochs=m.get("freeze_imaging_backbones_epochs", 5),
        )
        cfg.pretrain.update(pt)
        cfg.audit = raw.get("audit") or {}
        cfg.report_sources = {k: v for k, v in (raw.get("reports") or {}).items()
                              if v and v.get("enabled", True)}
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg
