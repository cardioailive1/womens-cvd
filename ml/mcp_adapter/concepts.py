"""Concept sets for the HCMAN women's CVD study.

Conditions are defined by ICD-10-CM *source* codes (OMOP: condition_source_concept_id ->
CONCEPT where vocabulary_id = 'ICD10CM'), because the six target conditions have specific
ICD-10-CM codes but some collapse into broader SNOMED standard concepts. Labs and vitals are
defined by LOINC codes on measurement_concept_id. Drugs are RxNorm ingredients expanded to
descendants through CONCEPT_ANCESTOR.

Every code list here is a *starting point for clinical review*. Check each against the MCP
Schema Visualizer and have a cardiologist sign off before the cohort is frozen.
A trailing "%" means prefix match (all descendants in the ICD-10-CM hierarchy).
"""
from __future__ import annotations

# --------------------------------------------------------------------------------------
# The six target conditions (condition head, output 1). Order = model output order.
# --------------------------------------------------------------------------------------
CONDITIONS: dict[str, list[str]] = {
    "SCAD": ["I25.42"],                                   # coronary artery dissection
    "PPCM": ["O90.3"],                                    # peripartum cardiomyopathy
    "PREECLAMPSIA_CVD": ["O11%", "O14%", "O15%"],         # preeclampsia, HELLP (O14.2), eclampsia
    "TAKOTSUBO": ["I51.81"],                              # takotsubo syndrome
    "CMD_INOCA": ["I20.81", "I25.85"],                    # angina w/ coronary microvascular
                                                          # dysfunction; chronic CMD (FY2024 codes)
    "FMD": ["I77.3"],                                     # arterial fibromuscular dysplasia
}
CONDITION_ORDER: list[str] = list(CONDITIONS)

# --------------------------------------------------------------------------------------
# Women-specific / reproductive features (stream 4) and stratum definitions.
# --------------------------------------------------------------------------------------
REPRO: dict[str, list[str]] = {
    "pregnant_state": ["Z33.1", "O09%", "Z34%", "Z3A%"],  # pregnant, high-risk supervision, weeks
    "high_risk_pregnancy": ["O09%"],
    "gest_weeks": ["Z3A%"],                               # Z3A.<ww> encodes gestational weeks
    "delivery": ["Z37%", "O80", "O82"],                   # outcome of delivery, normal / cesarean
    "postpartum_care": ["Z39%", "O90%"],
    "preeclampsia": ["O11%", "O14%", "O15%"],
    "hellp": ["O14.2%"],
    "gestational_htn": ["O13%"],
    "gestational_diabetes": ["O24.4%"],
    "menopause_natural": ["N95.1", "Z78.0"],
    "menopause_premature": ["E28.31%"],
    "menopause_surgical": ["N95.3", "E89.4%", "Z90.72%"],
    "perimenopause": ["N95.9", "N95.8"],
    "pcos": ["E28.2"],
    "endometriosis": ["N80%"],
    "sle": ["M32%"],
    "ra": ["M05%", "M06%"],
    "migraine_with_aura": ["G43.1%"],
    "fmd": ["I77.3"],
    "breast_cancer": ["C50%", "Z85.3"],
    "emotional_stress": ["F43.0", "F43.2%", "Z63%", "Z65%"],
}

# Cardiotoxic breast-cancer therapy: RxNorm ingredient names, expanded via CONCEPT_ANCESTOR.
CARDIOTOXIC_INGREDIENTS = ["doxorubicin", "epirubicin", "trastuzumab", "pertuzumab"]

# --------------------------------------------------------------------------------------
# Outcomes (MACE, postpartum CVD) and atypical-presentation symptoms.
# --------------------------------------------------------------------------------------
OUTCOMES: dict[str, list[str]] = {
    "mi": ["I21%", "I22%"],
    "stroke": ["I60%", "I61%", "I63%"],
    "heart_failure": ["I50%"],                            # counted only on inpatient/ED visits
    "cardiomyopathy": ["I42%", "I43%", "O90.3"],
    "acs": ["I20.0", "I21%", "I22%", "I24%"],
}
ATYPICAL_SYMPTOMS = ["R53.83", "R53.1", "R11%", "R68.84", "M54.6", "M54.9", "R10.13", "R06.0%"]
CHEST_PAIN = ["R07%", "I20.9"]

# Exclusions
EXCLUSIONS = ["Q20%", "Q21%", "Q22%", "Q23%", "Q24%", "Q25%", "Q26%", "Q27%", "Q28%", "Z94.1"]

# --------------------------------------------------------------------------------------
# Stream 5 serial labs + vitals (LOINC). Values converted to the canonical unit.
# --------------------------------------------------------------------------------------
LABS: dict[str, dict] = {
    "bnp":        {"loinc": ["30934-4"], "unit": "pg/mL"},
    "nt_probnp":  {"loinc": ["33762-6", "83107-3"], "unit": "pg/mL"},
    "troponin_i": {"loinc": ["10839-9", "42757-5", "89579-7"], "unit": "ng/L"},
    "troponin_t": {"loinc": ["6598-7", "67151-1"], "unit": "ng/L"},
    "creatinine": {"loinc": ["2160-0"], "unit": "mg/dL"},
    "potassium":  {"loinc": ["2823-3", "6298-4"], "unit": "mmol/L"},
    "sodium":     {"loinc": ["2951-2", "2947-0"], "unit": "mmol/L"},
    "hemoglobin": {"loinc": ["718-7"], "unit": "g/dL"},
    "platelets":  {"loinc": ["777-3", "26515-7"], "unit": "10*3/uL"},
}
LAB_ORDER: list[str] = list(LABS)

VITALS: dict[str, list[str]] = {
    "sbp": ["8480-6"], "dbp": ["8462-4"], "heart_rate": ["8867-4"],
    "bmi": ["39156-5"], "weight": ["29463-7"],
}

# Unit conversion to canonical units (source unit string, lower-cased -> multiplier).
UNIT_CONVERSIONS: dict[str, dict[str, float]] = {
    "troponin_i": {"ng/ml": 1000.0, "ug/l": 1000.0, "ng/l": 1.0, "pg/ml": 1.0},
    "troponin_t": {"ng/ml": 1000.0, "ug/l": 1000.0, "ng/l": 1.0, "pg/ml": 1.0},
    "creatinine": {"umol/l": 1 / 88.42, "mg/dl": 1.0},
    "hemoglobin": {"g/l": 0.1, "g/dl": 1.0, "mmol/l": 1.611},
    "bnp": {"ng/l": 1.0, "pg/ml": 1.0},
    "nt_probnp": {"ng/l": 1.0, "pg/ml": 1.0},
}

# --------------------------------------------------------------------------------------
# Events that count as a "cardiac evaluation" (anchor for the index date).
# --------------------------------------------------------------------------------------
CARDIAC_EVAL_LOINC = (LABS["bnp"]["loinc"] + LABS["nt_probnp"]["loinc"]
                      + LABS["troponin_i"]["loinc"] + LABS["troponin_t"]["loinc"])
# CPT4 procedure codes: ECG (93000-93010), TTE (93303-93308, 93350-93352)
CARDIAC_EVAL_CPT = ["93000", "93005", "93010", "93303", "93304", "93306", "93307",
                    "93308", "93350", "93351", "93352"]

# OMOP standard concept ids used directly
FEMALE_GENDER_CONCEPT_ID = 8532
INPATIENT_VISIT_IDS = [9201, 262]        # inpatient, ER+inpatient
ED_VISIT_IDS = [9203, 262]
BLACK_RACE_CONCEPT_ID = 8516


def all_icd_groups() -> dict[str, list[str]]:
    """Every ICD-10-CM group the cohort query needs, namespaced to avoid collisions."""
    groups: dict[str, list[str]] = {}
    groups.update({f"cond_{k}": v for k, v in CONDITIONS.items()})
    groups.update({f"repro_{k}": v for k, v in REPRO.items()})
    groups.update({f"out_{k}": v for k, v in OUTCOMES.items()})
    groups["sym_atypical"] = ATYPICAL_SYMPTOMS
    groups["sym_chest_pain"] = CHEST_PAIN
    groups["excl"] = EXCLUSIONS
    return groups


# ICD codes that leak the label and must never appear as EHR input tokens (stream 3).
LEAKY_CODE_PREFIXES: list[str] = [c.rstrip("%") for v in CONDITIONS.values() for c in v]
