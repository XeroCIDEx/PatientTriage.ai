"""
verification.py
================
The Verification Engine — the core orchestration described in Section 11.

For a single patient it:
  1. Runs the deterministic Safety Layer.
  2. Runs Model 1 and Model 2.
  3. Compares their ESI predictions -> AGREEMENT / DISAGREEMENT.
  4. Computes an uncertainty assessment (margin, entropy, missingness).
  5. Applies safety-rule floor (never allow ML to be less urgent than a
     triggered safety rule).
  6. Produces a final structured decision object consumed by the UI and
     the audit log.

This module NEVER outputs a disease diagnosis — only an ESI priority
level with plain-language contributing factors, per Section 12/20.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .safety_rules import apply_safety_layer
from .uncertainty import assess_uncertainty

PRIORITY_GROUP = {1: "CRITICAL", 2: "CRITICAL", 3: "URGENT", 4: "NON-URGENT", 5: "NON-URGENT"}

DISCLAIMER = ("This prototype is an AI decision-support system for research/demo purposes and is "
              "not a medical device or a substitute for clinical judgment.")


@dataclass
class TriageDecision:
    patient_id: str
    model1_result: dict
    model2_result: dict
    agreement: bool
    agreement_tier: str                # AGREE | DISAGREE | MAJOR_DISAGREE (spec Section 20)
    verification_status: str          # AGREEMENT | DISAGREEMENT | INSUFFICIENT_DATA
    uncertainty: object
    safety_flags: list
    final_esi: int
    priority_group: str
    nurse_review_required: bool
    contributing_factors: list
    missing_fields: list
    disclaimer: str = DISCLAIMER


def classify_agreement_tier(esi1: int, esi2: int) -> str:
    """Section 20: distinguish plain disagreement from MAJOR disagreement
    (a >=2-level ESI gap), since a 2-vs-3 split is a very different signal
    from a 2-vs-5 split even though both are technically 'disagreement'."""
    gap = abs(esi1 - esi2)
    if gap == 0:
        return "AGREE"
    if gap == 1:
        return "DISAGREE"
    return "MAJOR_DISAGREE"


def missing_field_report(patient: dict) -> list:
    labels = {
        "sbp": "Systolic blood pressure", "dbp": "Diastolic blood pressure",
        "spo2": "Oxygen saturation (SpO2)", "resp_rate": "Respiratory rate",
        "temperature": "Temperature", "heart_rate": "Heart rate", "pain_score": "Pain score",
    }
    missing = [labels[k] for k in labels if patient.get(k) is None]
    if patient.get("history") is None or patient.get("has_history_info", 1) == 0:
        missing.append("Medical history")
    return missing


def contributing_factors_from_model1(model1_wrapper, x_numeric_row) -> list:
    top = model1_wrapper.feature_importance(top_k=6)
    friendly = {
        "spo2": "Oxygen saturation (SpO2)", "sbp": "Systolic blood pressure",
        "heart_rate": "Heart rate", "resp_rate": "Respiratory rate", "temperature": "Temperature",
        "pain_score": "Reported pain level", "shock_index": "Shock index (HR/SBP ratio)",
        "pulse_pressure": "Pulse pressure", "age": "Age", "missing_field_count": "Amount of missing intake data",
    }
    return [friendly.get(name, name) for name, _imp in top]


def run_verification(patient: dict, model1_wrapper, model2_wrapper,
                      x_numeric_row, x_text1_row, x_text2_row) -> TriageDecision:

    m1 = model1_wrapper.predict_single(x_numeric_row, x_text1_row)
    m2 = model2_wrapper.predict_single(x_numeric_row, x_text2_row)

    agreement = (m1["esi_prediction"] == m2["esi_prediction"])
    agreement_tier = classify_agreement_tier(m1["esi_prediction"], m2["esi_prediction"])
    missing_fields = missing_field_report(patient)
    missing_score = len(missing_fields) / 7.0

    probs_arr = np.array([m1["probabilities"][k] for k in sorted(m1["probabilities"])])
    unc = assess_uncertainty(probs_arr, missing_score, model_disagreement=not agreement)

    # base candidate ESI = Model 1's prediction unless models disagree,
    # in which case we do NOT auto-pick — nurse review is required and we
    # conservatively surface the MORE urgent (lower-numbered) of the two
    # as the provisional flag while awaiting the nurse's decision.
    if agreement:
        candidate_esi = m1["esi_prediction"]
        status = "AGREEMENT"
        nurse_review = False
    else:
        candidate_esi = min(m1["esi_prediction"], m2["esi_prediction"])
        status = "DISAGREEMENT"
        nurse_review = True

    if len(missing_fields) >= 3 or (agreement and missing_score > 0.3):
        status = "INSUFFICIENT_DATA" if status == "AGREEMENT" else status
        nurse_review = True

    final_esi, safety_flags = apply_safety_layer(patient, candidate_esi)
    if safety_flags:
        nurse_review = True

    if unc.level == "HIGH":
        nurse_review = True

    if agreement_tier == "MAJOR_DISAGREE":
        nurse_review = True  # a >=2-level ESI gap always forces review, regardless of other signals

    contributing = contributing_factors_from_model1(model1_wrapper, x_numeric_row)

    return TriageDecision(
        patient_id=patient.get("patient_id", "UNKNOWN"),
        model1_result=m1,
        model2_result=m2,
        agreement=agreement,
        agreement_tier=agreement_tier,
        verification_status=status,
        uncertainty=unc,
        safety_flags=safety_flags,
        final_esi=final_esi,
        priority_group=PRIORITY_GROUP[final_esi],
        nurse_review_required=nurse_review,
        contributing_factors=contributing,
        missing_fields=missing_fields,
    )
