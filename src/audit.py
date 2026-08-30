"""
audit.py
========
Append-only audit trail (Section 13/20). Every triage decision AND every
clinician override is recorded as a JSON line in logs/audit_log.jsonl so
nothing is silently lost. This is intentionally simple (JSONL, not a
database) so it is easy to inspect for a hackathon demo.
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timezone

from .version import MODEL_VERSION, PREPROCESSING_VERSION

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "audit_log.jsonl")


def _now():
    return datetime.now(timezone.utc).isoformat()


def log_event(event: dict):
    event = dict(event)
    event.setdefault("timestamp", _now())
    event.setdefault("model_version", MODEL_VERSION)
    event.setdefault("preprocessing_version", PREPROCESSING_VERSION)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")
    return event


def log_triage_decision(decision, patient: dict, assessment_id: str = None,
                         previous_assessment_id: str = None, reassessment: bool = False):
    return log_event({
        "event_type": "TRIAGE_DECISION",
        "patient_id": decision.patient_id,
        "assessment_id": assessment_id,
        "previous_assessment_id": previous_assessment_id,
        "reassessment": reassessment,
        "model1_prediction": decision.model1_result["esi_prediction"],
        "model1_confidence": decision.model1_result["confidence"],
        "model1_top2_margin": decision.model1_result.get("top2_margin"),
        "model2_prediction": decision.model2_result["esi_prediction"],
        "model2_confidence": decision.model2_result["confidence"],
        "model2_top2_margin": decision.model2_result.get("top2_margin"),
        "agreement": decision.agreement,
        "agreement_tier": decision.agreement_tier,
        "verification_status": decision.verification_status,
        "final_esi_recommendation": decision.final_esi,
        "priority_group": decision.priority_group,
        "uncertainty_level": decision.uncertainty.level,
        "uncertainty_reasons": decision.uncertainty.reasons,
        "safety_flags": [f.rule_id for f in decision.safety_flags],
        "missing_fields": decision.missing_fields,
        "nurse_review_required": decision.nurse_review_required,
    })


def log_clinician_override(patient_id: str, ai_esi: int, ai_confidence: float,
                            model1_result: dict, model2_result: dict,
                            clinician_esi: int, reason: str, input_completeness: float,
                            safety_flags: list, assessment_id: str = None):
    return log_event({
        "event_type": "CLINICIAN_OVERRIDE" if clinician_esi != ai_esi else "CLINICIAN_ACCEPT",
        "patient_id": patient_id,
        "assessment_id": assessment_id,
        "ai_recommended_esi": ai_esi,
        "ai_confidence": ai_confidence,
        "model1_result": model1_result,
        "model2_result": model2_result,
        "clinician_decision_esi": clinician_esi,
        "override_reason": reason,
        "input_completeness": input_completeness,
        "safety_flags": safety_flags,
    })


def log_deterioration(patient_id: str, before: dict, after: dict, escalated: bool):
    return log_event({
        "event_type": "DETERIORATION_DETECTED" if escalated else "REASSESSMENT_NO_CHANGE",
        "patient_id": patient_id,
        "before": before,
        "after": after,
        "escalated": escalated,
    })


def log_pathway_change(patient_id: str, previous_pathway: str, new_pathway: str, reason: str, assessment_id: str = None):
    return log_event({
        "event_type": "PATHWAY_CHANGED",
        "patient_id": patient_id,
        "assessment_id": assessment_id,
        "previous_state": previous_pathway,
        "new_state": new_pathway,
        "reason": reason,
    })


def log_main_ed_bed_released(patient_id: str, new_pathway: str, final_esi: int):
    return log_event({
        "event_type": "MAIN_ED_BED_RELEASED",
        "patient_id": patient_id,
        "previous_state": "MAIN_ED",
        "new_state": new_pathway,
        "final_esi": final_esi,
    })


def log_main_ed_bed_assigned(patient_id: str, previous_pathway: str):
    return log_event({
        "event_type": "MAIN_ED_BED_ASSIGNED",
        "patient_id": patient_id,
        "previous_state": previous_pathway,
        "new_state": "MAIN_ED",
    })


def log_wait_time_breach(patient_id: str, esi: int, elapsed_minutes: float):
    return log_event({
        "event_type": "WAIT_TIME_BREACH",
        "patient_id": patient_id,
        "esi": esi,
        "elapsed_minutes": round(elapsed_minutes, 1),
    })


def read_audit_log(limit: int = 500) -> list:
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH) as f:
        lines = f.readlines()[-limit:]
    return [json.loads(l) for l in lines]


def clear_audit_log():
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
