"""
history_context.py
===================
Turns a raw HistoryBundle (from database/repository.py) into what the
rest of the pipeline actually needs:

  1. A clinical-language text block (spec Section 17 format) — fed into
     the SAME text corpus both Model 1 and Model 2 already consume, so
     the "same complete evidence, different representation" principle
     (spec Section 2/18) extends naturally to history too, with zero
     pipeline changes needed elsewhere.
  2. Structured temporal/summary features (spec Section 15/16) —
     days_since_last_encounter, active_diagnosis_count, encounter_count,
     previous vitals for delta computation — returned separately so a
     future retrain can add them as explicit numeric model inputs
     (see the "Deferred" note at the bottom of this file).

Relevance note (spec Section 13/14): this layer does NOT decide that
"chest pain + cardiac history = high relevance" via a hardcoded rule —
it retrieves and organizes ALL available history with its timestamps
and lets it flow into the model as text; any learned relevance
weighting is left to the models themselves once retrained on data that
includes history. This module's job is retrieval and organization, not
clinical judgment.
"""

from __future__ import annotations
from datetime import datetime, date
from typing import Optional

from database.repository import HistoryBundle, HistoryStatus


def _days_since(date_str: Optional[str]) -> Optional[int]:
    if not date_str:
        return None
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (date.today() - d).days
    except ValueError:
        return None


def summarize_temporal_features(bundle: HistoryBundle) -> dict:
    """Structured, numeric-friendly summary — NOT yet wired into the
    trained models' feature vector (that requires regenerating the
    synthetic training data with matching history and retraining; see
    the module docstring's Deferred note). Available now for display
    and for a future retrain."""
    active_diagnoses = [d for d in bundle.diagnoses if d.get("status") in ("active", "chronic")]
    most_recent_encounter = bundle.encounters[0] if bundle.encounters else None
    most_recent_vitals = bundle.vitals[0] if bundle.vitals else None

    return {
        "history_status": bundle.history_status,
        "history_record_count": bundle.history_record_count,
        "active_diagnosis_count": len(active_diagnoses),
        "total_diagnosis_count": len(bundle.diagnoses),
        "encounter_count": len(bundle.encounters),
        "days_since_last_encounter": _days_since(most_recent_encounter["event_date"]) if most_recent_encounter else None,
        "days_since_last_vitals": _days_since(most_recent_vitals["event_date"]) if most_recent_vitals else None,
        "previous_vitals": most_recent_vitals,
        "medication_count": len([m for m in bundle.medications if m.get("status") == "active"]),
        "allergy_count": len(bundle.allergies),
    }


def compute_vital_deltas(previous_vitals: Optional[dict], current_vitals: dict) -> dict:
    """Section 16: previous-vs-current vital differences. Returns None
    for any field where we don't have both a previous and current value
    — we never fabricate a delta from a missing value."""
    if not previous_vitals:
        return {}
    deltas = {}
    field_map = {"heart_rate": "heart_rate", "sbp": "sbp", "spo2": "spo2",
                 "resp_rate": "resp_rate", "temperature": "temperature"}
    for key, cur_key in field_map.items():
        prev_v = previous_vitals.get(key)
        cur_v = current_vitals.get(cur_key)
        if prev_v is not None and cur_v is not None:
            deltas[f"delta_{key}"] = round(cur_v - prev_v, 1)
    return deltas


def build_clinical_text_block(bundle: HistoryBundle, max_diagnoses: int = 6, max_encounters: int = 3) -> str:
    """Section 17 format — the SAME text that gets appended to the
    existing `history` field consumed by both models' text corpus. Older
    events are compressed (Section 12: "do not send unlimited raw
    history") — capped at the most recent N of each record type."""
    if bundle.history_status == HistoryStatus.NO_RECORD:
        return "No prior medical history on file (first-time or no recorded history)."
    if bundle.history_status == HistoryStatus.UNAVAILABLE:
        return "Historical record unavailable (retrieval failed) — proceeding on current encounter information only."

    lines = []
    active = [d for d in bundle.diagnoses if d.get("status") in ("active", "chronic")]
    if active:
        lines.append("[ACTIVE HISTORY]")
        for d in active[:max_diagnoses]:
            lines.append(f"{d['diagnosis']} — since {d.get('event_date', 'unknown date')}")

    resolved = [d for d in bundle.diagnoses if d.get("status") == "resolved"]
    if resolved:
        lines.append("\n[OTHER / RESOLVED HISTORY]")
        for d in resolved[:max_diagnoses]:
            lines.append(f"{d['diagnosis']} (resolved) — {d.get('event_date', 'unknown date')}")

    if bundle.encounters:
        lines.append("\n[PREVIOUS ENCOUNTERS]")
        for e in bundle.encounters[:max_encounters]:
            esi_str = f", ESI {e['esi_recorded']}" if e.get("esi_recorded") else ""
            lines.append(f"{e.get('event_date', 'unknown date')}: {e.get('chief_complaint', 'encounter')}{esi_str}")

    active_meds = [m for m in bundle.medications if m.get("status") == "active"]
    if active_meds:
        lines.append("\n[MEDICATIONS]")
        for m in active_meds:
            lines.append(f"{m['medication']}" + (f" {m['dose']}" if m.get("dose") else ""))

    if bundle.allergies:
        lines.append("\n[ALLERGIES]")
        for a in bundle.allergies:
            lines.append(f"{a['allergen']}" + (f" — {a['reaction']}" if a.get("reaction") else ""))

    return "\n".join(lines) if lines else "History on file but no structured clinical facts recorded."


"""
DEFERRED (documented, not silently dropped):

The temporal/summary features above (days_since_last_encounter,
active_diagnosis_count, vital deltas, etc.) are computed and available,
but NOT YET wired into Model 1's numeric feature vector or used for a
history ablation study (spec Section 31: Model A "current only" vs.
Model B "current+history" vs. Model C "current+history+temporal").
Doing that properly requires:
  1. Regenerating the synthetic training dataset so each training
     patient has a matching multi-encounter history in the database
     (today's data_generation.py produces single-encounter patients).
  2. Adding the temporal fields to preprocessing.build_numeric_matrix().
  3. Retraining and running the actual ablation comparison (not just
     asserting history would help).
This is the next concrete piece of work, not implemented in this pass.
Today, history reaches the models ONLY via the clinical text block
above being appended to the existing text corpus — which is genuine
signal (both models already read history text), just not yet
represented as explicit structured deltas.
"""
