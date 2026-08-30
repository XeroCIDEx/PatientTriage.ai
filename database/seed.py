"""
seed.py
=======
Populates the SQLite patient database with SYNTHETIC longitudinal
patient histories, deterministically (fixed seed), so the Patient ID
search demo is reproducible.

Includes, per spec Section 61 (robustness must be evaluated across):
  - patients with rich, relevant history (e.g. prior MI + current chest pain)
  - patients with history but UNRELATED to current complaint
  - patients with old history vs. recent history
  - patients with NO history at all (NO_RECORD — never seeded, so a
    lookup for a never-seeded ID correctly returns NO_RECORD)
  - one reserved ID that always returns UNAVAILABLE
    (database.repository.SIMULATED_OUTAGE_PATIENT_ID)

This is clearly-labeled synthetic demo data, not real patient records.

Run: python3 -m database.seed
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime, timedelta

from database.repository import PatientRepository
from database.schema import DB_PATH, init_db

SEED = 42

DIAGNOSIS_POOL_RELEVANT_CARDIAC = ["myocardial infarction", "hypertension", "atrial fibrillation", "coronary artery disease"]
DIAGNOSIS_POOL_UNRELATED = ["kidney stone", "seasonal allergies", "ankle sprain (resolved)", "appendectomy"]
DIAGNOSIS_POOL_CHRONIC = ["type 2 diabetes", "COPD", "chronic kidney disease", "asthma"]


def _iso(days_ago: int) -> str:
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")


def seed_patient_richhistory_relevant(repo: PatientRepository, rng):
    pid = "PT-1001"
    repo.add_patient(pid, age=67, sex="M")
    repo.add_diagnosis("DX-1001-1", pid, "myocardial infarction", event_date=_iso(730), status="chronic")
    repo.add_diagnosis("DX-1001-2", pid, "hypertension", event_date=_iso(1200), status="active")
    repo.add_diagnosis("DX-1001-3", pid, "type 2 diabetes", event_date=_iso(900), status="active")
    repo.add_encounter("ENC-1001-1", pid, event_date=_iso(400), chief_complaint="chest discomfort", esi_recorded=2, status="admitted")
    repo.add_encounter("ENC-1001-2", pid, event_date=_iso(40), chief_complaint="abnormal ECG on routine follow-up", esi_recorded=3)
    repo.add_vitals("VT-1001-1", pid, event_date=_iso(40), heart_rate=88, sbp=142, dbp=88, spo2=96, resp_rate=18, temperature=36.8)
    return pid


def seed_patient_history_unrelated(repo: PatientRepository, rng):
    pid = "PT-1002"
    repo.add_patient(pid, age=45, sex="F")
    repo.add_diagnosis("DX-1002-1", pid, "kidney stone", event_date=_iso(1800), status="resolved")
    repo.add_diagnosis("DX-1002-2", pid, "appendectomy", event_date=_iso(2500), status="resolved")
    repo.add_encounter("ENC-1002-1", pid, event_date=_iso(1800), chief_complaint="flank pain", esi_recorded=3)
    return pid


def seed_patient_recent_chronic(repo: PatientRepository, rng):
    pid = "PT-1003"
    repo.add_patient(pid, age=8, sex="F")
    repo.add_diagnosis("DX-1003-1", pid, "asthma", event_date=_iso(200), status="active")
    repo.add_encounter("ENC-1003-1", pid, event_date=_iso(60), chief_complaint="wheezing", esi_recorded=3)
    repo.add_vitals("VT-1003-1", pid, event_date=_iso(60), heart_rate=110, sbp=100, dbp=64, spo2=95, resp_rate=26, temperature=37.0)
    return pid


def seed_patient_old_history(repo: PatientRepository, rng):
    pid = "PT-1004"
    repo.add_patient(pid, age=72, sex="M")
    repo.add_diagnosis("DX-1004-1", pid, "coronary artery disease", event_date=_iso(3650), status="chronic")
    repo.add_encounter("ENC-1004-1", pid, event_date=_iso(3600), chief_complaint="chest pain", esi_recorded=2)
    return pid


def seed_patient_frequent_flyer(repo: PatientRepository, rng):
    pid = "PT-1005"
    repo.add_patient(pid, age=54, sex="M")
    repo.add_diagnosis("DX-1005-1", pid, "COPD", event_date=_iso(1500), status="chronic")
    for i, days in enumerate([300, 200, 90, 30, 5]):
        repo.add_encounter(f"ENC-1005-{i}", pid, event_date=_iso(days),
                            chief_complaint="shortness of breath", esi_recorded=3 if days > 10 else 2)
    repo.add_vitals("VT-1005-1", pid, event_date=_iso(5), heart_rate=98, sbp=128, dbp=80, spo2=91, resp_rate=24, temperature=37.2)
    return pid


def main():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    init_db(DB_PATH)
    repo = PatientRepository(DB_PATH)
    rng = np.random.default_rng(SEED)

    seeded = [
        seed_patient_richhistory_relevant(repo, rng),
        seed_patient_history_unrelated(repo, rng),
        seed_patient_recent_chronic(repo, rng),
        seed_patient_old_history(repo, rng),
        seed_patient_frequent_flyer(repo, rng),
    ]

    print(f"Seeded {len(seeded)} synthetic patients with longitudinal history: {seeded}")
    print("Try looking up: PT-1001 (relevant cardiac history), PT-1002 (unrelated history), "
          "PT-1003 (pediatric, recent chronic), PT-1004 (old history), PT-1005 (frequent ED visits)")
    print("Try a made-up ID (e.g. PT-9999) to see NO_RECORD.")
    print("Try 'SIMULATE-DB-OUTAGE' to see the UNAVAILABLE state.")


if __name__ == "__main__":
    main()
