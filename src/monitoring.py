"""
monitoring.py
=============
Waiting-room monitoring / continuous reassessment (Section 14).

Simulates a patient's vitals drifting while they wait, and detects
deterioration TWO ways, independently:
  1. Vitals-based: simple, explicit delta thresholds (documented below).
     This is a prototype heuristic, not a validated early-warning score,
     though it is conceptually similar in spirit to early-warning scores
     used in real EDs (e.g. NEWS2) — no claim of equivalence is made.
  2. Wait-time-based (NEW): a patient can also be flagged purely because
     they have waited longer than is considered safe for their assigned
     ESI level, even if nobody has re-taken their vitals and nothing
     numeric has "changed." This matches the brief's explicit requirement
     that the system "monitor patients already in the waiting queue and
     trigger re-assessment if wait time exceeds safe thresholds for
     their severity level" — independent of the vitals-worsening check.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

# Deterioration thresholds: if ANY of these deltas occur between the
# initial and current vitals, we flag deterioration.
DELTA_THRESHOLDS = {
    "spo2_drop": 4,       # SpO2 falls by >= 4 points
    "hr_rise": 20,         # HR rises by >= 20 bpm
    "rr_rise": 8,          # RR rises by >= 8/min
    "sbp_drop": 15,        # SBP falls by >= 15 mmHg
    "pain_rise": 3,        # pain score rises by >= 3 points
}

# Maximum SAFE wait time (minutes) before a patient at a given ESI level
# must be automatically flagged for reassessment, regardless of whether
# their vitals have changed. These are illustrative prototype values,
# not a reproduction of any specific hospital's or standard's official
# targets, chosen to be directionally consistent with how ESI levels are
# commonly described (ESI 1 = immediate, ESI 5 = can safely wait longest).
MAX_SAFE_WAIT_MINUTES = {
    1: 0,     # immediate — should never be "waiting"
    2: 10,
    3: 30,
    4: 60,
    5: 120,
}


@dataclass
class WaitingPatient:
    patient_id: str
    initial_vitals: dict
    current_vitals: dict
    initial_esi: int
    current_esi: int
    wait_minutes: int
    status: str = "WAITING"   # WAITING | REASSESSMENT_REQUIRED | ESCALATED
    deterioration_reasons: list = None


def detect_deterioration(initial: dict, current: dict) -> tuple:
    reasons = []
    if current.get("spo2") is not None and initial.get("spo2") is not None:
        if initial["spo2"] - current["spo2"] >= DELTA_THRESHOLDS["spo2_drop"]:
            reasons.append(f"SpO2 dropped from {initial['spo2']}% to {current['spo2']}%.")
    if current.get("heart_rate") is not None and initial.get("heart_rate") is not None:
        if current["heart_rate"] - initial["heart_rate"] >= DELTA_THRESHOLDS["hr_rise"]:
            reasons.append(f"Heart rate rose from {initial['heart_rate']} to {current['heart_rate']} bpm.")
    if current.get("resp_rate") is not None and initial.get("resp_rate") is not None:
        if current["resp_rate"] - initial["resp_rate"] >= DELTA_THRESHOLDS["rr_rise"]:
            reasons.append(f"Respiratory rate rose from {initial['resp_rate']} to {current['resp_rate']}/min.")
    if current.get("sbp") is not None and initial.get("sbp") is not None:
        if initial["sbp"] - current["sbp"] >= DELTA_THRESHOLDS["sbp_drop"]:
            reasons.append(f"Systolic BP dropped from {initial['sbp']} to {current['sbp']} mmHg.")
    if current.get("pain_score") is not None and initial.get("pain_score") is not None:
        if current["pain_score"] - initial["pain_score"] >= DELTA_THRESHOLDS["pain_rise"]:
            reasons.append(f"Pain score rose from {initial['pain_score']} to {current['pain_score']}.")
    return (len(reasons) > 0), reasons


def check_wait_time_breach(esi: int, wait_minutes: int) -> tuple:
    """Returns (breached: bool, reason: str|None). Flags a patient purely
    based on elapsed wait time vs. the safe ceiling for their current ESI
    level — independent of any vitals check."""
    limit = MAX_SAFE_WAIT_MINUTES.get(esi, 120)
    if wait_minutes >= limit:
        return True, (f"Patient has waited {wait_minutes} min, exceeding the "
                       f"{limit}-min safe wait ceiling for ESI {esi}.")
    return False, None


def assess_waiting_patient(esi: int, wait_minutes: int, initial_vitals: dict, current_vitals: dict) -> dict:
    """Combines BOTH triggers (vitals deterioration + wait-time breach)
    into one status report for a waiting patient. Either trigger alone is
    sufficient to require reassessment — they are not mutually exclusive
    and both are surfaced separately so a nurse can see exactly why."""
    vitals_worse, vitals_reasons = detect_deterioration(initial_vitals, current_vitals)
    wait_breached, wait_reason = check_wait_time_breach(esi, wait_minutes)

    reasons = list(vitals_reasons)
    if wait_reason:
        reasons.append(wait_reason)

    if vitals_worse and wait_breached:
        status = "ESCALATED"
    elif vitals_worse or wait_breached:
        status = "REASSESSMENT_REQUIRED"
    else:
        status = "WAITING"

    return {
        "status": status,
        "vitals_deteriorated": vitals_worse,
        "wait_time_breached": wait_breached,
        "reasons": reasons,
    }


def simulate_wait_drift(vitals: dict, rng: np.random.Generator, deteriorate: bool) -> dict:
    """Simulate vitals changing over a waiting period. If `deteriorate`,
    bias the drift toward worsening values; otherwise small random noise."""
    v = dict(vitals)
    if deteriorate:
        v["spo2"] = max(60, (v.get("spo2") or 97) - rng.uniform(4, 12))
        v["heart_rate"] = (v.get("heart_rate") or 85) + rng.uniform(20, 45)
        v["resp_rate"] = (v.get("resp_rate") or 16) + rng.uniform(6, 16)
        v["sbp"] = max(50, (v.get("sbp") or 115) - rng.uniform(10, 35))
        v["pain_score"] = min(10, (v.get("pain_score") or 3) + rng.integers(2, 5))
    else:
        v["spo2"] = np.clip((v.get("spo2") or 97) + rng.uniform(-1, 1), 85, 100)
        v["heart_rate"] = (v.get("heart_rate") or 85) + rng.uniform(-4, 4)
        v["resp_rate"] = (v.get("resp_rate") or 16) + rng.uniform(-1, 1)
        v["sbp"] = (v.get("sbp") or 115) + rng.uniform(-5, 5)
        v["pain_score"] = v.get("pain_score", 3)
    for k in ["spo2", "heart_rate", "resp_rate", "sbp", "pain_score"]:
        if v.get(k) is not None:
            v[k] = round(float(v[k]), 1)
    return v
