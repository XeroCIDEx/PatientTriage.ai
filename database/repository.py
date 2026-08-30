"""
repository.py
==============
The ONLY module in this codebase allowed to run SQL against the patient
database (spec Section 53: "the ML models must NOT directly query the
database... Frontend -> Authenticated Backend -> Authorization ->
Database -> Required clinical data -> Preprocessing -> ML models").
Everything else — the UI, the history-contextualization layer, the ML
pipeline — goes through the methods on `PatientRepository`, never touches
sqlite3 directly.

This prototype doesn't implement a separate network-facing backend
process (that's out of scope for a Streamlit prototype), but the
repository boundary itself is real: no other module opens the database
file, and this class is the seam where a real deployment would insert
authentication/authorization checks before any query runs.
"""

from __future__ import annotations
import sqlite3
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .schema import DB_PATH, init_db

# Reserved patient ID used by the demo to deliberately exercise the
# UNAVAILABLE history state (spec Section 8C) — simulates a database or
# service outage for a patient who DOES exist, distinct from a patient
# genuinely having no records (NO_RECORD).
SIMULATED_OUTAGE_PATIENT_ID = "SIMULATE-DB-OUTAGE"


class HistoryStatus:
    FOUND = "FOUND"
    NO_RECORD = "NO_RECORD"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass
class HistoryBundle:
    patient_id: str
    history_status: str
    encounters: list = field(default_factory=list)
    diagnoses: list = field(default_factory=list)
    vitals: list = field(default_factory=list)          # most-recent-first
    medications: list = field(default_factory=list)
    allergies: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    history_record_count: int = 0
    history_last_updated: Optional[str] = None


class PatientRepository:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        if not os.path.exists(db_path):
            init_db(db_path)

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def patient_exists(self, patient_id: str) -> bool:
        with self._conn() as conn:
            row = conn.execute("SELECT 1 FROM patients WHERE patient_id = ?", (patient_id,)).fetchone()
            return row is not None

    def get_history(self, patient_id: str) -> HistoryBundle:
        """The main entry point: given a patient_id, retrieve everything
        known about them and classify the history_status explicitly."""
        # simulate a technical failure (DB/service unavailable) for a
        # reserved demo ID, distinct from genuinely having no records
        if patient_id == SIMULATED_OUTAGE_PATIENT_ID:
            return HistoryBundle(patient_id=patient_id, history_status=HistoryStatus.UNAVAILABLE)

        if not self.patient_exists(patient_id):
            return HistoryBundle(patient_id=patient_id, history_status=HistoryStatus.NO_RECORD)

        with self._conn() as conn:
            encounters = [dict(r) for r in conn.execute(
                "SELECT * FROM encounters WHERE patient_id = ? ORDER BY event_date DESC", (patient_id,))]
            diagnoses = [dict(r) for r in conn.execute(
                "SELECT * FROM diagnoses WHERE patient_id = ? ORDER BY event_date DESC", (patient_id,))]
            vitals = [dict(r) for r in conn.execute(
                "SELECT * FROM vitals WHERE patient_id = ? ORDER BY event_date DESC", (patient_id,))]
            medications = [dict(r) for r in conn.execute(
                "SELECT * FROM medications WHERE patient_id = ? ORDER BY start_date DESC", (patient_id,))]
            allergies = [dict(r) for r in conn.execute(
                "SELECT * FROM allergies WHERE patient_id = ?", (patient_id,))]
            notes = [dict(r) for r in conn.execute(
                "SELECT * FROM clinical_notes WHERE patient_id = ? ORDER BY event_date DESC", (patient_id,))]

        total_records = len(encounters) + len(diagnoses) + len(vitals) + len(medications) + len(allergies) + len(notes)
        if total_records == 0:
            # patient exists as an identity but genuinely has no clinical
            # records on file — this is NO_RECORD, not UNAVAILABLE
            return HistoryBundle(patient_id=patient_id, history_status=HistoryStatus.NO_RECORD)

        last_updated = None
        dates = [e.get("event_date") for e in encounters + diagnoses + vitals + notes if e.get("event_date")]
        if dates:
            last_updated = max(dates)

        return HistoryBundle(
            patient_id=patient_id, history_status=HistoryStatus.FOUND,
            encounters=encounters, diagnoses=diagnoses, vitals=vitals,
            medications=medications, allergies=allergies, notes=notes,
            history_record_count=total_records, history_last_updated=last_updated,
        )

    def add_patient(self, patient_id: str, age: int, sex: str = ""):
        with self._conn() as conn:
            conn.execute("INSERT OR IGNORE INTO patients (patient_id, age, sex, created_at) VALUES (?, ?, ?, ?)",
                         (patient_id, age, sex, datetime.now(timezone.utc).isoformat()))
            conn.commit()

    def add_encounter(self, encounter_id: str, patient_id: str, event_date: str,
                       chief_complaint: str, esi_recorded: Optional[int] = None,
                       encounter_type: str = "ED", status: str = "completed", arrival_time: str = None):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO encounters "
                "(encounter_id, patient_id, encounter_type, event_date, arrival_time, chief_complaint, esi_recorded, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (encounter_id, patient_id, encounter_type, event_date, arrival_time, chief_complaint, esi_recorded, status))
            conn.commit()

    def add_vitals(self, record_id: str, patient_id: str, event_date: str, **vitals):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO vitals "
                "(record_id, patient_id, event_date, heart_rate, sbp, dbp, spo2, resp_rate, temperature) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (record_id, patient_id, event_date, vitals.get("heart_rate"), vitals.get("sbp"),
                 vitals.get("dbp"), vitals.get("spo2"), vitals.get("resp_rate"), vitals.get("temperature")))
            conn.commit()

    def add_diagnosis(self, record_id: str, patient_id: str, diagnosis: str,
                       event_date: str, status: str = "active", recorded_at: str = None):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO diagnoses (record_id, patient_id, diagnosis, event_date, recorded_at, status) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (record_id, patient_id, diagnosis, event_date, recorded_at or event_date, status))
            conn.commit()
