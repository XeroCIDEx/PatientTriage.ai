"""
safety_rules.py
================
Deterministic, auditable "red-flag" safety layer — now AGE-AWARE.

These are PROTOTYPE HEURISTIC THRESHOLDS, written for demonstration
purposes (see age_bands.py for the full rationale and honesty notice
about their limits). They are NOT a substitute for, or a reproduction
of, any specific clinical guideline, and are NOT clinically validated.
Every rule that fires is logged with a human-readable reason so a nurse
can see exactly why the system escalated a patient.

Design principle (Section 8): if a rule fires, we only ever push the
patient's urgency UP (or flag for immediate review) — a triggered safety
rule can never be used to silently downgrade a patient below what the ML
model predicted.

CHANGE FROM v0.1: previously this module used ONE fixed threshold set
for every patient regardless of age (a known, documented gap — flagged
explicitly because applying adult-calibrated cutoffs uniformly can
silently under-triage pediatric or geriatric patients, whose normal and
dangerous vital-sign ranges differ meaningfully from an adult's). Rules
now look up age-band-specific thresholds from age_bands.py before
evaluating.
"""

from __future__ import annotations
from dataclasses import dataclass

from .age_bands import age_band, SAFETY_THRESHOLDS


@dataclass
class SafetyFlag:
    rule_id: str
    description: str
    forces_min_esi: int  # the rule guarantees ESI is AT MOST this number (i.e. at least this urgent)


def evaluate_safety_rules(patient: dict) -> list:
    """patient: dict with numeric vitals possibly None/NaN if missing.
    Returns list of SafetyFlag objects (empty if none triggered)."""
    flags = []

    def val(key, default=None):
        v = patient.get(key, default)
        return default if v is None else v

    spo2 = val("spo2")
    sbp = val("sbp")
    rr = val("resp_rate")
    hr = val("heart_rate")
    temp = val("temperature")
    age = val("age", 40)
    band = age_band(age)
    t = SAFETY_THRESHOLDS[band]
    symptoms_text = " ".join([
        str(patient.get("chief_complaint", "") or ""),
        str(patient.get("symptoms", "") or ""),
    ]).lower()

    band_tag = f"[{band}]"

    if spo2 is not None and spo2 < t["spo2_critical"]:
        flags.append(SafetyFlag("SAFETY_SPO2_CRITICAL",
                                 f"{band_tag} Severely low oxygen saturation (SpO2={spo2}%, "
                                 f"critical threshold for this age group is <{t['spo2_critical']}%).", 1))
    elif spo2 is not None and spo2 < t["spo2_warning"]:
        flags.append(SafetyFlag("SAFETY_SPO2_LOW",
                                 f"{band_tag} Low oxygen saturation (SpO2={spo2}%, "
                                 f"warning threshold for this age group is <{t['spo2_warning']}%).", 2))

    if sbp is not None and sbp < t["sbp_critical"]:
        flags.append(SafetyFlag("SAFETY_SBP_CRITICAL",
                                 f"{band_tag} Severe hypotension (SBP={sbp} mmHg, "
                                 f"critical threshold for this age group is <{t['sbp_critical']} mmHg).", 1))
    elif sbp is not None and sbp < t["sbp_warning"]:
        flags.append(SafetyFlag("SAFETY_SBP_LOW",
                                 f"{band_tag} Hypotension (SBP={sbp} mmHg, "
                                 f"warning threshold for this age group is <{t['sbp_warning']} mmHg).", 2))

    if rr is not None and (rr < t["rr_critical_low"] or rr > t["rr_critical_high"]):
        flags.append(SafetyFlag("SAFETY_RR_EXTREME",
                                 f"{band_tag} Extreme respiratory rate (RR={rr}/min, "
                                 f"critical range for this age group is <{t['rr_critical_low']} or >{t['rr_critical_high']}/min).", 1))
    elif rr is not None and (rr < t["rr_warning_low"] or rr > t["rr_warning_high"]):
        flags.append(SafetyFlag("SAFETY_RR_ABNORMAL",
                                 f"{band_tag} Abnormal respiratory rate (RR={rr}/min, "
                                 f"warning range for this age group is <{t['rr_warning_low']} or >{t['rr_warning_high']}/min).", 2))

    if hr is not None and (hr > t["hr_critical_high"] or hr < t["hr_critical_low"]):
        flags.append(SafetyFlag("SAFETY_HR_EXTREME",
                                 f"{band_tag} Extreme heart rate (HR={hr} bpm, "
                                 f"critical range for this age group is <{t['hr_critical_low']} or >{t['hr_critical_high']} bpm).", 1))
    elif hr is not None and (hr > t["hr_warning_high"] or hr < t["hr_warning_low"]):
        flags.append(SafetyFlag("SAFETY_HR_ABNORMAL",
                                 f"{band_tag} Abnormal heart rate (HR={hr} bpm, "
                                 f"warning range for this age group is <{t['hr_warning_low']} or >{t['hr_warning_high']} bpm).", 2))

    if temp is not None and temp >= t["temp_high"]:
        flags.append(SafetyFlag("SAFETY_TEMP_HIGH",
                                 f"{band_tag} High fever for this age group (Temp={temp}\u00b0C, "
                                 f"threshold is \u2265{t['temp_high']}\u00b0C).", 2))

    danger_terms = [
        "unresponsive", "not breathing", "gunshot", "severe bleeding",
        "uncontrolled bleeding", "facial droop", "slurred speech", "witnessed seizure",
        "severe head trauma", "severe allergic reaction",
    ]
    for term in danger_terms:
        if term in symptoms_text:
            flags.append(SafetyFlag("SAFETY_DANGER_KEYWORD", f"Reported presentation includes a recognized danger sign: '{term}'.", 1))
            break

    return flags


def apply_safety_layer(patient: dict, ml_predicted_esi: int) -> tuple:
    """Returns (final_min_esi_bound, flags). The ML prediction is never
    allowed to be *less* urgent (numerically higher) than the strictest
    triggered safety rule."""
    flags = evaluate_safety_rules(patient)
    if not flags:
        return ml_predicted_esi, flags
    strictest = min(f.forces_min_esi for f in flags)
    final_esi = min(ml_predicted_esi, strictest)
    return final_esi, flags
