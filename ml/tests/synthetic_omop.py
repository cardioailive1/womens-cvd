"""Synthetic OMOP CDM v5.4 database (DuckDB) for testing the adapter end to end.

Not real data. Profiles are planted so every stratum, label and stream is exercised:
SCAD (+/- FMD), postmenopausal Takotsubo / CMD, postpartum (+/- PPCM), high-risk pregnancy
(+/- preeclampsia), prior preeclampsia, PCOS / autoimmune, background women, and some men
(who must never enter the cohort).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ICD = ["I25.42", "O90.3", "O14.10", "O14.20", "O11.3", "I51.81", "I20.81", "I25.85", "I77.3",
       "Z33.1", "O09.513", "Z3A.28", "Z3A.34", "Z37.0", "O80", "Z39.2", "O24.410", "O13.3",
       "N95.1", "Z78.0", "N95.3", "E28.2", "N80.0", "M32.9", "M06.9", "G43.109", "C50.911",
       "F43.0", "I21.4", "I63.9", "I50.9", "I42.0", "I20.0", "R53.83", "R11.0", "R68.84",
       "M54.6", "R10.13", "R07.9", "Q21.1", "I10", "E11.9", "E78.5", "F17.210", "J45.909"]
LOINC = {"30934-4": "bnp", "33762-6": "nt_probnp", "89579-7": "troponin_i", "2160-0": "creatinine",
         "2823-3": "potassium", "2951-2": "sodium", "718-7": "hemoglobin", "777-3": "platelets",
         "8480-6": "sbp", "8462-4": "dbp", "8867-4": "heart_rate", "39156-5": "bmi"}
CPT = ["93000", "93306"]
ATC = ["C07AB", "C09AA", "B01AC", "L01DB", "L01FD"]
UNITS = ["pg/mL", "ng/L", "mg/dL", "mmol/L", "g/dL", "10*3/uL", "mm[Hg]", "/min", "kg/m2"]


def build(seed: int = 7, n_women: int = 900, n_men: int = 60):
    rng = np.random.default_rng(seed)
    cid = iter(range(1000, 10**7))
    concept, code2id = [], {}

    def add(code, vocab, cls="", name=None):
        i = next(cid); code2id[(vocab, code)] = i
        concept.append((i, name or code, vocab, cls, code)); return i

    for c in ICD: add(c, "ICD10CM")
    for c in LOINC: add(c, "LOINC")
    for c in CPT: add(c, "CPT4")
    for c in ATC: add(c, "ATC", "ATC 4th")
    for u in UNITS: add(u, "UCUM")
    ingr = {n: add(n, "RxNorm", "Ingredient", n) for n in ["metoprolol", "lisinopril", "aspirin",
                                                         "doxorubicin", "trastuzumab"]}
    atc_of = {"metoprolol": "C07AB", "lisinopril": "C09AA", "aspirin": "B01AC",
              "doxorubicin": "L01DB", "trastuzumab": "L01FD"}
    drugs, anc = {}, []
    for n, iid in ingr.items():
        d = add(f"{n} tablet", "RxNorm", "Clinical Drug"); drugs[n] = d
        anc += [(iid, d), (code2id[("ATC", atc_of[n])], d), (d, d)]

    people, obs, visits, conds, observations, meas, procs, drug_rows, deaths = ([] for _ in range(9))
    vid = iter(range(1, 10**8))
    base = pd.Timestamp("2016-01-01")
    sites = [(1, "Rochester"), (2, "Arizona"), (3, "Florida")]
    profiles = (["scad"] * 90 + ["tako"] * 80 + ["pp"] * 140 + ["preg"] * 120 + ["prior_pe"] * 110
                + ["pcos"] * 120)
    profiles += ["bg"] * (n_women - len(profiles))

    def visit(pid, d, kind=9202):
        v = next(vid); visits.append((v, pid, kind, d)); return v

    def dx(pid, code, d, kind=9202):
        v = visit(pid, d, kind)
        row = (pid, code2id[("ICD10CM", code)], d, v)
        # Z codes land in OBSERVATION in real OMOP ETLs — exercise the union
        (observations if code.startswith("Z") else conds).append(row)

    def lab(pid, loinc, d, val, unit):
        meas.append((pid, code2id[("LOINC", loinc)], d, float(val), code2id[("UCUM", unit)], unit))

    def workup(pid, d, trop=10.0, bnp=60.0):
        procs.append((pid, code2id[("CPT4", "93000")], d))
        procs.append((pid, code2id[("CPT4", "93306")], d + pd.Timedelta(days=1)))
        for k in range(3):
            dd = d - pd.Timedelta(days=int(rng.integers(0, 80)))
            lab(pid, "89579-7", dd, trop * rng.lognormal(0, .3), "ng/L")
            lab(pid, "30934-4", dd, bnp * rng.lognormal(0, .3), "pg/mL")
            lab(pid, "2160-0", dd, rng.normal(0.8, .15), "mg/dL")
            lab(pid, "2823-3", dd, rng.normal(4.1, .3), "mmol/L")
            lab(pid, "2951-2", dd, rng.normal(139, 2), "mmol/L")
            lab(pid, "718-7", dd, rng.normal(12.5, 1), "g/dL")
            lab(pid, "777-3", dd, rng.normal(250, 50), "10*3/uL")
        for k in range(4):
            dd = d - pd.Timedelta(days=int(rng.integers(0, 300)))
            lab(pid, "8480-6", dd, rng.normal(124, 14), "mm[Hg]")
            lab(pid, "8462-4", dd, rng.normal(78, 9), "mm[Hg]")
            lab(pid, "8867-4", dd, rng.normal(76, 10), "/min")
            lab(pid, "39156-5", dd, rng.normal(28, 5), "kg/m2")

    for pid, prof in enumerate(profiles + ["man"] * n_men, start=1):
        gender = 8507 if prof == "man" else 8532
        age0 = {"scad": 45, "tako": 64, "pp": 31, "preg": 36, "prior_pe": 40, "pcos": 33,
                "bg": 50, "man": 55}[prof] + int(rng.integers(-5, 6))
        t0 = base + pd.Timedelta(days=int(rng.integers(500, 2600)))   # anchor event
        yob = t0.year - age0
        race = int(rng.choice([8527, 8516, 8515, 0], p=[.7, .14, .08, .08]))
        site = sites[int(rng.integers(0, 3))][0]
        people.append((pid, gender, yob, race, 0, site, f"MC{pid:07d}"))  # de-identified clinic number
        obs.append((pid, base - pd.Timedelta(days=900), pd.Timestamp("2025-06-30")))
        for c in rng.choice(["I10", "E11.9", "E78.5", "F17.210", "J45.909"], int(rng.integers(0, 3)),
                            replace=False):
            dx(pid, str(c), t0 - pd.Timedelta(days=int(rng.integers(30, 600))))
        for n in rng.choice(["metoprolol", "lisinopril", "aspirin"], int(rng.integers(0, 3)), replace=False):
            drug_rows.append((pid, drugs[str(n)], t0 - pd.Timedelta(days=int(rng.integers(5, 300)))))

        if prof == "scad":
            workup(pid, t0 - pd.Timedelta(days=1), trop=900)
            dx(pid, "I25.42", t0, 9201); dx(pid, "I21.4", t0, 9201)
            dx(pid, "I25.42", t0 + pd.Timedelta(days=20))
            if rng.random() < .5:
                dx(pid, "R53.83", t0, 9201); dx(pid, "R68.84", t0, 9201)
            else:
                dx(pid, "R07.9", t0, 9201)
            if rng.random() < .4:
                fd = t0 + pd.Timedelta(days=int(rng.integers(10, 60)))
                dx(pid, "I77.3", fd); dx(pid, "I77.3", fd + pd.Timedelta(days=14))
            if rng.random() < .2:
                dx(pid, "I21.4", t0 + pd.Timedelta(days=int(rng.integers(60, 300))), 9201)
        elif prof == "tako":
            dx(pid, "N95.1", t0 - pd.Timedelta(days=3000))
            workup(pid, t0, trop=400, bnp=900)
            code = "I51.81" if rng.random() < .55 else "I20.81"
            dx(pid, code, t0, 9201); dx(pid, code, t0 + pd.Timedelta(days=30))
            if rng.random() < .6: dx(pid, "F43.0", t0 - pd.Timedelta(days=3))
            if rng.random() < .15: dx(pid, "I50.9", t0 + pd.Timedelta(days=100), 9201)
        elif prof == "pp":
            dx(pid, "Z33.1", t0 - pd.Timedelta(days=200)); dx(pid, "Z37.0", t0 - pd.Timedelta(days=40), 9201)
            if rng.random() < .3: dx(pid, "O14.10", t0 - pd.Timedelta(days=60), 9201)
            ppcm = rng.random() < .2
            workup(pid, t0, bnp=1500 if ppcm else 80)
            if ppcm:
                dx(pid, "O90.3", t0 + pd.Timedelta(days=5), 9201); dx(pid, "O90.3", t0 + pd.Timedelta(days=40))
                if rng.random() < .4: dx(pid, "I50.9", t0 + pd.Timedelta(days=150), 9201)
        elif prof == "preg":
            dx(pid, "O09.513", t0 - pd.Timedelta(days=60)); dx(pid, "Z3A.28", t0 - pd.Timedelta(days=10))
            workup(pid, t0)
            if rng.random() < .35:
                dx(pid, "O14.10", t0 + pd.Timedelta(days=7), 9201); dx(pid, "O14.10", t0 + pd.Timedelta(days=9), 9201)
                if rng.random() < .3: dx(pid, "O14.20", t0 + pd.Timedelta(days=9), 9201)
        elif prof == "prior_pe":
            dx(pid, "O14.10", t0 - pd.Timedelta(days=1500), 9201)
            dx(pid, "Z37.0", t0 - pd.Timedelta(days=1490), 9201)
            workup(pid, t0)
            if rng.random() < .15: dx(pid, "I63.9", t0 + pd.Timedelta(days=200), 9201)
        elif prof == "pcos":
            dx(pid, str(rng.choice(["E28.2", "N80.0", "M32.9", "M06.9"])), t0 - pd.Timedelta(days=400))
            if rng.random() < .3: dx(pid, "G43.109", t0 - pd.Timedelta(days=200))
            workup(pid, t0)
            if rng.random() < .1: dx(pid, "I20.81", t0 + pd.Timedelta(days=10), 9201)
        elif prof == "bg":
            if rng.random() < .3: workup(pid, t0)
            if rng.random() < .05: dx(pid, "C50.911", t0 - pd.Timedelta(days=700)); drug_rows.append(
                (pid, drugs["doxorubicin"], t0 - pd.Timedelta(days=600)))
        elif prof == "man":
            dx(pid, "I25.42", t0, 9201); workup(pid, t0)
        if prof == "scad" and rng.random() < .03:
            dx(pid, "Q21.1", t0 - pd.Timedelta(days=5000))   # congenital -> excluded
        if prof in ("tako", "scad") and rng.random() < .05:
            deaths.append((pid, t0 + pd.Timedelta(days=int(rng.integers(20, 300))), 0))

    D = lambda rows, cols: pd.DataFrame(rows, columns=cols)
    return {
        "concept": D(concept, ["concept_id", "concept_name", "vocabulary_id", "concept_class_id", "concept_code"]),
        "concept_ancestor": D(anc, ["ancestor_concept_id", "descendant_concept_id"]),
        "person": D(people, ["person_id", "gender_concept_id", "year_of_birth", "race_concept_id",
                             "ethnicity_concept_id", "care_site_id", "person_source_value"]),
        "care_site": D(sites, ["care_site_id", "care_site_name"]),
        "observation_period": D(obs, ["person_id", "observation_period_start_date", "observation_period_end_date"]),
        "visit_occurrence": D(visits, ["visit_occurrence_id", "person_id", "visit_concept_id", "visit_start_date"]),
        "condition_occurrence": D(conds, ["person_id", "condition_source_concept_id", "condition_start_date",
                                          "visit_occurrence_id"]),
        "observation": D(observations, ["person_id", "observation_source_concept_id", "observation_date",
                                        "visit_occurrence_id"]),
        "measurement": D([(m[0], m[1], m[2], m[3], m[4], m[5]) for m in meas],
                         ["person_id", "measurement_concept_id", "measurement_date", "value_as_number",
                          "unit_concept_id", "unit_source_value"]),
        "procedure_occurrence": D(procs, ["person_id", "procedure_concept_id", "procedure_date"]),
        "drug_exposure": D(drug_rows, ["person_id", "drug_concept_id", "drug_exposure_start_date"]),
        "death": D(deaths, ["person_id", "death_date", "cause_concept_id"]),
    }


def load_into_duckdb(con, schema: str = "omop_cdm", **kw) -> None:
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    for name, df in build(**kw).items():
        con.register("tmp_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {schema}.{name} AS SELECT * FROM tmp_df")
        con.unregister("tmp_df")
