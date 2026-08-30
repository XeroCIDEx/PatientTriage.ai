"""
age_bands.py
============
Single shared source of truth for age-band classification and age-band
specific "abnormal vital" thresholds, used by data_generation.py,
preprocessing.py, and safety_rules.py so all three agree on what counts
as pediatric / adult / geriatric and what counts as a dangerous vital
sign in each group.

IMPORTANT: these thresholds are PROTOTYPE HEURISTICS for a hackathon
demo, loosely informed by commonly cited pediatric/geriatric vital-sign
differences (e.g. children normally run faster heart rates and
respiratory rates than adults; older adults may not mount a fever or
tachycardia response even when seriously ill). They are NOT sourced from
a specific clinical guideline, NOT clinically validated, and NOT a
substitute for age-specific triage tools used in real EDs (e.g.
pediatric-specific ESI guidance). The purpose here is to demonstrate
*that* an age-aware architecture matters and *how* it would be wired in
— not to ship clinically defensible cutoffs.
"""

from __future__ import annotations


def age_band(age) -> str:
    """Returns 'pediatric' (<13), 'geriatric' (>=65), or 'adult'."""
    if age is None:
        return "adult"
    if age < 13:
        return "pediatric"
    if age >= 65:
        return "geriatric"
    return "adult"


# Age-band-specific "danger" cutoffs used by the deterministic safety
# layer. Each entry: (critical_threshold, warning_threshold) — crossing
# critical forces ESI 1, crossing warning forces ESI 2 (see safety_rules.py).
# For "low is bad" vitals (SpO2, SBP) the critical value is the lower
# bound; for "high is bad" vitals it's the upper bound. RR and HR have
# both a low and a high danger side.
SAFETY_THRESHOLDS = {
    "pediatric": {
        "spo2_critical": 92, "spo2_warning": 95,
        "sbp_critical": 70, "sbp_warning": 80,
        "rr_critical_low": 12, "rr_critical_high": 50, "rr_warning_low": 16, "rr_warning_high": 40,
        "hr_critical_low": 60, "hr_critical_high": 190, "hr_warning_low": 70, "hr_warning_high": 170,
        "temp_high": 39.5,
    },
    "adult": {
        "spo2_critical": 90, "spo2_warning": 94,
        "sbp_critical": 80, "sbp_warning": 95,
        "rr_critical_low": 8, "rr_critical_high": 32, "rr_warning_low": 10, "rr_warning_high": 26,
        "hr_critical_low": 40, "hr_critical_high": 150, "hr_warning_low": 50, "hr_warning_high": 130,
        "temp_high": 40.0,
    },
    "geriatric": {
        # Older adults often show a BLUNTED response to serious illness
        # (less able to mount tachycardia/fever), so warning thresholds
        # are set tighter (more sensitive) than the adult band even
        # though "normal" resting vitals can look similar — a prototype
        # simplification meant to avoid under-triaging frail patients
        # whose vitals look deceptively stable.
        "spo2_critical": 88, "spo2_warning": 92,
        "sbp_critical": 90, "sbp_warning": 100,
        "rr_critical_low": 10, "rr_critical_high": 28, "rr_warning_low": 12, "rr_warning_high": 24,
        "hr_critical_low": 45, "hr_critical_high": 130, "hr_warning_low": 50, "hr_warning_high": 110,
        "temp_high": 38.3,   # geriatric patients may be seriously ill at lower fever than adults
    },
}
