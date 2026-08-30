"""
demo_patients.py
================
Builds the required >=20-patient demonstration set (spec Section 16),
covering every mandated category, and runs each patient through the full
pipeline (safety layer, both models, verification, uncertainty), printing
a compact result table. Deterministic (fixed hand-built cases, no RNG),
so the demo is fully reproducible.

Run: python3 demo_patients.py   (after train.py)
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import get_pipeline
from src import audit

DEMO_PATIENTS = [
    # 1. Normal low-risk adult
    dict(patient_id="DEMO-01", category="Normal low-risk", age=29, heart_rate=76, resp_rate=15, sbp=118, dbp=76,
         spo2=99, temperature=36.9, pain_score=1, chief_complaint="requesting medication refill",
         symptoms="none reported", history="no significant past medical history"),

    # 2. Critical patient
    dict(patient_id="DEMO-02", category="Critical", age=61, heart_rate=132, resp_rate=30, sbp=76, dbp=48,
         spo2=84, temperature=37.8, pain_score=9, chief_complaint="crushing chest pain radiating to left arm",
         symptoms="shortness of breath; diaphoresis", history="prior myocardial infarction; hypertension"),

    # 3. Ambiguous presentation
    dict(patient_id="DEMO-03", category="Ambiguous", age=44, heart_rate=104, resp_rate=22, sbp=108, dbp=70,
         spo2=94, temperature=37.9, pain_score=6, chief_complaint="moderate abdominal pain",
         symptoms="persistent vomiting and diarrhea", history="type 2 diabetes"),

    # 4. Pediatric patient
    dict(patient_id="DEMO-04", category="Pediatric", age=4, heart_rate=150, resp_rate=34, sbp=92, dbp=58,
         spo2=95, temperature=39.1, pain_score=5, chief_complaint="high fever with confusion",
         symptoms="lethargy", history="no significant past medical history"),

    # 5. Geriatric patient
    dict(patient_id="DEMO-05", category="Geriatric", age=82, heart_rate=58, resp_rate=20, sbp=138, dbp=82,
         spo2=93, temperature=36.4, pain_score=4, chief_complaint="new onset severe headache, worst of life",
         symptoms="confusion", history="atrial fibrillation on blood thinners; hypertension"),

    # 6. First-time / zero-history patient
    dict(patient_id="DEMO-06", category="Zero-history", age=25, heart_rate=88, resp_rate=17, sbp=114, dbp=74,
         spo2=98, temperature=37.1, pain_score=3, chief_complaint="mild sore throat",
         symptoms="none reported", history="no prior medical history (first-time patient)"),

    # 7. Missing-vitals patient
    dict(patient_id="DEMO-07", category="Missing vitals", age=52, heart_rate=95, resp_rate=None, sbp=None, dbp=None,
         spo2=96, temperature=37.3, pain_score=5, chief_complaint="worsening cough and fever for 3 days",
         symptoms="fatigue", history="COPD"),

    # 8. Missing-history patient
    dict(patient_id="DEMO-08", category="Missing history", age=38, heart_rate=90, resp_rate=18, sbp=122, dbp=80,
         spo2=97, temperature=37.0, pain_score=4, chief_complaint="possible broken bone after fall",
         symptoms="swelling", history=None, has_history_info=0),

    # 9. Model disagreement case (constructed to be borderline / mixed signals)
    dict(patient_id="DEMO-09", category="Model disagreement (expected)", age=57, heart_rate=101, resp_rate=21,
         sbp=112, dbp=72, spo2=95, temperature=37.6, pain_score=6,
         chief_complaint="signs of infection with rapid heart rate",
         symptoms="mild ear pain; rash, no other symptoms", history="chronic kidney disease"),

    # 10. High-uncertainty case (probabilities expected to be close)
    dict(patient_id="DEMO-10", category="High uncertainty", age=48, heart_rate=98, resp_rate=19, sbp=116, dbp=76,
         spo2=95, temperature=37.4, pain_score=5, chief_complaint="migraine not responding to home treatment",
         symptoms="moderate laceration needing sutures", history="no significant past medical history"),

    # 11. Abnormal vitals, vague symptoms
    dict(patient_id="DEMO-11", category="Abnormal vitals / vague symptoms", age=67, heart_rate=128, resp_rate=26,
         sbp=94, dbp=60, spo2=91, temperature=37.0, pain_score=2, chief_complaint="mild low back pain, chronic",
         symptoms="none reported", history="hypertension; chronic kidney disease"),

    # 12. Concerning symptoms, near-normal vitals
    dict(patient_id="DEMO-12", category="Concerning symptoms / normal vitals", age=39, heart_rate=82, resp_rate=16,
         sbp=118, dbp=78, spo2=98, temperature=37.0, pain_score=7,
         chief_complaint="stroke-like symptoms, facial droop and slurred speech",
         symptoms="none reported", history="no significant past medical history"),

    # 13. Deterioration while waiting (initial mild) -> handled in monitoring demo below
    dict(patient_id="DEMO-13", category="Waiting-room baseline (will deteriorate)", age=71, heart_rate=90,
         resp_rate=18, sbp=128, dbp=80, spo2=96, temperature=37.2, pain_score=3,
         chief_complaint="mild low back pain, chronic", symptoms="none reported", history="hypertension"),

    # 14. Clinician-override demonstration case
    dict(patient_id="DEMO-14", category="Clinician override case", age=55, heart_rate=100, resp_rate=20, sbp=110,
         dbp=72, spo2=95, temperature=37.5, pain_score=6, chief_complaint="moderate abdominal pain",
         symptoms="patient appears pale and anxious per nurse", history="prior stroke"),

    # 15. Severe hypoxia -> safety layer trigger
    dict(patient_id="DEMO-15", category="Safety layer trigger (SpO2)", age=70, heart_rate=110, resp_rate=28,
         sbp=105, dbp=68, spo2=82, temperature=37.6, pain_score=4, chief_complaint="shortness of breath at rest",
         symptoms="cough", history="COPD"),

    # 16. Danger-sign keyword -> safety layer trigger
    dict(patient_id="DEMO-16", category="Safety layer trigger (danger keyword)", age=33, heart_rate=105,
         resp_rate=22, sbp=110, dbp=70, spo2=97, temperature=36.9, pain_score=5,
         chief_complaint="severe uncontrolled bleeding from laceration",
         symptoms="dizziness", history="no significant past medical history"),

    # 17. Non-urgent routine
    dict(patient_id="DEMO-17", category="Non-urgent routine", age=34, heart_rate=72, resp_rate=14, sbp=116, dbp=74,
         spo2=99, temperature=36.7, pain_score=0, chief_complaint="routine wound check",
         symptoms="none reported", history="no significant past medical history"),

    # 18. Pediatric with missing SpO2
    dict(patient_id="DEMO-18", category="Pediatric + missing SpO2", age=7, heart_rate=118, resp_rate=24, sbp=98,
         dbp=62, spo2=None, temperature=38.6, pain_score=4, chief_complaint="cold symptoms for 2 days",
         symptoms="fever", history="asthma"),

    # 19. Geriatric with multiple comorbidities, urgent complaint
    dict(patient_id="DEMO-19", category="Geriatric multi-comorbidity", age=88, heart_rate=96, resp_rate=20,
         sbp=100, dbp=64, spo2=93, temperature=37.7, pain_score=5,
         chief_complaint="urinary tract infection symptoms", symptoms="confusion",
         history="type 2 diabetes; chronic kidney disease; prior stroke"),

    # 20. Pregnant patient, urgent
    dict(patient_id="DEMO-20", category="Pregnant, urgent", age=27, heart_rate=112, resp_rate=22, sbp=100, dbp=64,
         spo2=96, temperature=37.9, pain_score=7, chief_complaint="severe abdominal pain, sudden onset",
         symptoms="none reported", history="pregnancy, second trimester"),

    # 21. Immunocompromised with fever
    dict(patient_id="DEMO-21", category="Immunocompromised + fever", age=58, heart_rate=108, resp_rate=22,
         sbp=104, dbp=66, spo2=95, temperature=38.9, pain_score=4,
         chief_complaint="high fever with confusion", symptoms="chills",
         history="immunocompromised, on chemotherapy"),

    # 22. Asthma attack not improving
    dict(patient_id="DEMO-22", category="Respiratory distress", age=19, heart_rate=118, resp_rate=28, sbp=122,
         dbp=78, spo2=92, temperature=37.0, pain_score=3,
         chief_complaint="asthma attack, not improving with inhaler", symptoms="wheezing", history="asthma"),
]


def run_demo():
    pipeline = get_pipeline()
    print(f"Running {len(DEMO_PATIENTS)} demonstration patients through the full pipeline...\n")
    print(f"{'ID':10} {'Category':38} {'M1':4} {'M2':4} {'Status':16} {'FinalESI':9} {'Group':11} {'Unc.':9} {'Nurse?':7}")
    print("-" * 120)
    results = []
    for p in DEMO_PATIENTS:
        p_clean = {k: v for k, v in p.items() if k != "category"}
        decision = pipeline.assess_patient(p_clean)
        audit.log_triage_decision(decision, p_clean)
        results.append((p, decision))
        print(f"{p['patient_id']:10} {p['category'][:38]:38} "
              f"{decision.model1_result['esi_prediction']:<4} {decision.model2_result['esi_prediction']:<4} "
              f"{decision.verification_status:16} ESI-{decision.final_esi:<5} {decision.priority_group:11} "
              f"{decision.uncertainty.level:9} {'YES' if decision.nurse_review_required else 'no'}")

    # ---- clinician override demonstration (case 14) ----
    override_patient = next(p for p in DEMO_PATIENTS if p["patient_id"] == "DEMO-14")
    override_decision = next(d for p, d in results if p["patient_id"] == "DEMO-14")
    audit.log_clinician_override(
        patient_id="DEMO-14",
        ai_esi=override_decision.final_esi,
        ai_confidence=override_decision.model1_result["confidence"],
        model1_result=override_decision.model1_result,
        model2_result=override_decision.model2_result,
        clinician_esi=2,
        reason="Patient appears clinically worse (pale, anxious, guarding) than captured by current intake information.",
        input_completeness=1.0 - len(override_decision.missing_fields) / 7.0,
        safety_flags=[f.rule_id for f in override_decision.safety_flags],
    )
    print(f"\nLogged a clinician override example for DEMO-14: AI={override_decision.final_esi} -> Clinician=ESI-2")

    # ---- waiting-room deterioration demonstration (case 13) ----
    from src.monitoring import simulate_wait_drift, detect_deterioration
    import numpy as np
    baseline = next(p for p in DEMO_PATIENTS if p["patient_id"] == "DEMO-13")
    rng = np.random.default_rng(13)
    drifted_vitals = simulate_wait_drift(baseline, rng, deteriorate=True)
    deteriorated, reasons = detect_deterioration(baseline, drifted_vitals)
    after_patient = {**baseline, **drifted_vitals}
    after_decision = pipeline.assess_patient({k: v for k, v in after_patient.items() if k != "category"})
    audit.log_deterioration("DEMO-13", baseline, drifted_vitals, deteriorated)
    print(f"\nWaiting-room deterioration demo for DEMO-13:")
    print(f"  Initial ESI: {[d for p,d in results if p['patient_id']=='DEMO-13'][0].final_esi}  "
          f"-> After drift ESI: {after_decision.final_esi}")
    for r in reasons:
        print(f"   - {r}")

    return results


if __name__ == "__main__":
    run_demo()
