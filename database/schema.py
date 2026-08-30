"""
schema.py
=========
SQLite schema for the patient history database (spec Section 10/11).

Deliberately NOT one giant text blob per patient — every historical fact
is its own timestamped row, so the system can reason about *when* things
happened (event_date) separately from *when they were documented*
(recorded_at), and so new facts can be appended without ever needing to
parse/rewrite existing history.

This is a lightweight SQLite schema for the prototype. `database/repository.py`
is the only module allowed to touch this database directly (spec Section 53:
"the ML models must NOT directly query the database") — everything else,
including the ML pipeline, goes through the repository's Python methods.
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patienttriage.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id      TEXT PRIMARY KEY,
    age             INTEGER,
    sex             TEXT,
    created_at      TEXT
);

CREATE TABLE IF NOT EXISTS encounters (
    encounter_id    TEXT PRIMARY KEY,
    patient_id      TEXT NOT NULL,
    encounter_type  TEXT,           -- 'ED', 'admission', 'outpatient', etc.
    event_date      TEXT NOT NULL,  -- when the encounter happened
    arrival_time    TEXT,
    chief_complaint TEXT,
    esi_recorded    INTEGER,        -- ESI actually assigned that encounter, if known
    status          TEXT,           -- 'completed', 'admitted', 'discharged', etc.
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE IF NOT EXISTS diagnoses (
    record_id       TEXT PRIMARY KEY,
    patient_id      TEXT NOT NULL,
    diagnosis       TEXT NOT NULL,
    event_date      TEXT,           -- onset/diagnosis date
    recorded_at     TEXT,           -- when it was documented
    status          TEXT,           -- 'active', 'resolved', 'chronic'
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE IF NOT EXISTS vitals (
    record_id       TEXT PRIMARY KEY,
    patient_id      TEXT NOT NULL,
    event_date      TEXT NOT NULL,
    heart_rate      REAL,
    sbp             REAL,
    dbp             REAL,
    spo2            REAL,
    resp_rate       REAL,
    temperature     REAL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE IF NOT EXISTS medications (
    record_id       TEXT PRIMARY KEY,
    patient_id      TEXT NOT NULL,
    medication      TEXT NOT NULL,
    dose            TEXT,
    frequency       TEXT,
    start_date      TEXT,
    end_date        TEXT,
    status          TEXT,           -- 'active', 'discontinued'
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE IF NOT EXISTS allergies (
    record_id       TEXT PRIMARY KEY,
    patient_id      TEXT NOT NULL,
    allergen        TEXT NOT NULL,
    reaction        TEXT,
    date_recorded   TEXT,
    status          TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE IF NOT EXISTS clinical_notes (
    record_id       TEXT PRIMARY KEY,
    patient_id      TEXT NOT NULL,
    note            TEXT NOT NULL,
    event_date      TEXT,
    recorded_at     TEXT,
    author_role     TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE INDEX IF NOT EXISTS idx_encounters_patient ON encounters(patient_id, event_date);
CREATE INDEX IF NOT EXISTS idx_diagnoses_patient ON diagnoses(patient_id, event_date);
CREATE INDEX IF NOT EXISTS idx_vitals_patient ON vitals(patient_id, event_date);
CREATE INDEX IF NOT EXISTS idx_medications_patient ON medications(patient_id);
CREATE INDEX IF NOT EXISTS idx_allergies_patient ON allergies(patient_id);
CREATE INDEX IF NOT EXISTS idx_notes_patient ON clinical_notes(patient_id, event_date);
"""


def init_db(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Schema initialized at {DB_PATH}")
