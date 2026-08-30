"""
test_new_features.py
=====================
Tests for the three changes requested:
  1. Waiting Room -> Reassessment workflow (patient_state.py)
  2. Explainability (explainability.py)
  3. Clinician override controls the Waiting Room ESI (patient_state.py)

Self-contained (no pytest dependency required) — run with:
    python3 tests/test_new_features.py
Each test function asserts and prints PASS/FAIL; the script exits
non-zero if anything fails, so it can be used in CI too.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import get_pipeline
from src import patient_state as ps
from src.explainability import (
    shap_explain_model1, clinical_evidence_model2,
    build_agreement_explanation, build_disagreement_explanation,
    extract_history_evidence,
)
from src.preprocessing import full_preprocess, build_numeric_matrix, build_text_corpus
import pandas as pd

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def make_patient(**overrides):
    base = dict(patient_id="TEST-P1", age=55, heart_rate=90, resp_rate=18, sbp=120, dbp=78,
                spo2=97, temperature=37.0, pain_score=3, chief_complaint="mild sore throat",
                symptoms="none reported", history="no significant past medical history")
    base.update(overrides)
    return base


# ---------------------------------------------------------------------
# TEST 1 — WAITING TIME / REASSESSMENT WORKFLOW
# ---------------------------------------------------------------------

def test_reassessment_workflow():
    pipeline = get_pipeline()
    p1 = make_patient(heart_rate=90, sbp=125, spo2=98)
    decision1 = pipeline.assess_patient(p1)
    rec = ps.create_patient_record("TEST-P1", p1, decision1)

    check("initial assessment created", len(rec["assessments"]) == 1)
    first = rec["assessments"][0]
    check("initial assessment is not marked reassessment", first["reassessment"] is False)
    check("initial previous_assessment_id is None", first["previous_assessment_id"] is None)

    # simulate a wait-time breach (now computed live from real elapsed
    # time, not a stored flag — see test_realtime_and_pathways.py for
    # the live-computation tests)
    check("initial elapsed_seconds starts near zero", ps.elapsed_seconds(rec) < 5)

    # nurse clicks [REASSESS PATIENT]: prefill comes from first["input_snapshot"]
    prefill = dict(first["input_snapshot"])
    check("prefill contains previous HR", prefill["heart_rate"] == 90)
    check("prefill contains previous SBP", prefill["sbp"] == 125)

    # nurse edits values (deterioration) and resubmits -> reassessment
    p2 = dict(prefill)
    p2.update(heart_rate=125, sbp=90, spo2=91)
    decision2 = pipeline.assess_patient(p2)
    arrival_before = rec["arrival_timestamp"]
    rec = ps.add_reassessment(rec, p2, decision2)

    check("reassessment appended, not replacing history", len(rec["assessments"]) == 2)
    check("original assessment still present unmodified", rec["assessments"][0]["input_snapshot"]["heart_rate"] == 90)
    second = rec["assessments"][1]
    check("second assessment is marked reassessment", second["reassessment"] is True)
    check("second assessment links to first via previous_assessment_id",
          second["previous_assessment_id"] == first["assessment_id"])
    check("same patient_id preserved across reassessment", rec["patient_id"] == "TEST-P1")
    check("same encounter_id preserved across reassessment (not a new patient)",
          second["encounter_id"] == first["encounter_id"] == rec["encounter_id"])
    check("arrival_timestamp NOT reset by reassessment (real elapsed time preserved)",
          rec["arrival_timestamp"] == arrival_before)
    check("new assessment reflects updated (worse) vitals", second["input_snapshot"]["heart_rate"] == 125)


# ---------------------------------------------------------------------
# TEST 2 — HISTORY PRESERVED ACROSS REASSESSMENT
# ---------------------------------------------------------------------

def test_history_preserved_across_reassessment():
    pipeline = get_pipeline()
    history_text = "[ACTIVE HISTORY]\nmyocardial infarction — since 2024-08-27\nhypertension — since 2023-05-15"
    p1 = make_patient(history=history_text, has_history_info=1)
    decision1 = pipeline.assess_patient(p1)
    rec = ps.create_patient_record("TEST-P2", p1, decision1)

    # reassessment prefill should carry forward the SAME retrieved history
    # text without requiring the nurse to re-enter it
    prefill = dict(rec["assessments"][0]["input_snapshot"])
    check("history text carried into prefill unchanged", prefill["history"] == history_text)

    p2 = dict(prefill)
    p2.update(heart_rate=110)  # nurse only edits vitals
    decision2 = pipeline.assess_patient(p2)
    rec = ps.add_reassessment(rec, p2, decision2)

    check("history preserved in reassessment input_snapshot",
          rec["assessments"][1]["input_snapshot"]["history"] == history_text)

    evidence = extract_history_evidence(history_text)
    check("history evidence extraction finds active diagnoses", "myocardial infarction — since 2024-08-27" in evidence)


# ---------------------------------------------------------------------
# TEST 3 — AGREEMENT EXPLANATION
# ---------------------------------------------------------------------

def test_agreement_explanation():
    pipeline = get_pipeline()
    patient = make_patient(age=57, heart_rate=132, resp_rate=28, sbp=90, dbp=58, spo2=90,
                            temperature=38.0, pain_score=8, chief_complaint="severe chest pain",
                            symptoms="shortness of breath")
    decision = pipeline.assess_patient(patient)

    raw_df = pd.DataFrame([patient])
    df = full_preprocess(raw_df)
    Xn = build_numeric_matrix(df).values
    corpus = build_text_corpus(df)
    Xt1 = pipeline.vectorizers.transform_model1(corpus)
    word_vocab = pipeline.vectorizers.word_tfidf.get_feature_names_out()

    exp1 = shap_explain_model1(pipeline.model1, Xn, Xt1, patient, word_vocab)
    exp2 = clinical_evidence_model2(patient)

    check("Model 1 SHAP explanation available", exp1["available"] is True)
    check("Model 1 explanation has factors", len(exp1["factors"]) > 0)
    check("Model 1 factors are human-readable (no raw feature names)",
          all("shock_index" not in f["text"] and "_missing" not in f["text"] for f in exp1["factors"]))
    check("Model 2 evidence extraction found clinically relevant phrases",
          "chest pain" in exp2["evidence"] or "shortness of breath" in exp2["evidence"])

    if decision.agreement:
        lines = build_agreement_explanation(decision.final_esi, exp1["factors"], exp2["evidence"])
        check("agreement explanation mentions both models agreeing",
              any("independently classified" in l for l in lines))
        check("agreement explanation is non-empty", len(lines) > 0)


# ---------------------------------------------------------------------
# TEST 4 — DISAGREEMENT EXPLANATION (never fabricates a reason)
# ---------------------------------------------------------------------

def test_disagreement_explanation_never_fabricates():
    m1_result = {"esi_prediction": 2, "confidence": 0.7}
    m2_result = {"esi_prediction": 3, "confidence": 0.6}
    factors = [{"text": "Heart rate (HR 132 bpm) increased the likelihood of this ESI 2 assessment."}]
    evidence = ["abdominal pain"]

    dis = build_disagreement_explanation(m1_result, m2_result, factors, evidence)

    check("disagreement explanation shows model1 side ESI", dis["model1_side"]["esi"] == 2)
    check("disagreement explanation shows model2 side ESI", dis["model2_side"]["esi"] == 3)
    check("disagreement reason does NOT fabricate a specific causal claim",
          "ignored" not in dis["reason"].lower() and "because model" not in dis["reason"].lower())
    check("disagreement reason explicitly states no single cause can be established",
          "no single causal reason" in dis["reason"].lower())


# ---------------------------------------------------------------------
# TEST 5 — CLINICIAN OVERRIDE CONTROLS WAITING-ROOM ESI
# ---------------------------------------------------------------------

def test_override_controls_displayed_esi():
    pipeline = get_pipeline()
    patient = make_patient()
    decision = pipeline.assess_patient(patient)
    rec = ps.create_patient_record("TEST-P3", patient, decision)

    # force a known AI value for a deterministic test regardless of what the model predicted
    rec["assessments"][0]["ai_recommended_esi"] = 3

    check("before any decision, displayed_esi falls back to AI recommendation",
          ps.get_displayed_esi(rec) == 3)

    rec = ps.apply_override(rec, new_esi=2, reason="Patient appears clinically worse than intake suggests.")

    check("waiting room displays clinician-overridden ESI, not AI ESI", ps.get_displayed_esi(rec) == 2)
    check("AI recommendation remains stored and unmodified", rec["assessments"][-1]["ai_recommended_esi"] == 3)
    check("override reason is stored", rec["assessments"][-1]["override_reason"] == "Patient appears clinically worse than intake suggests.")
    check("override flag is True", ps.is_overridden(rec) is True)


# ---------------------------------------------------------------------
# TEST 6 — REASSESSMENT AFTER OVERRIDE PRESERVES HISTORY
# ---------------------------------------------------------------------

def test_reassessment_after_override_preserves_audit_history():
    pipeline = get_pipeline()
    patient = make_patient()
    decision1 = pipeline.assess_patient(patient)
    rec = ps.create_patient_record("TEST-P4", patient, decision1)
    rec["assessments"][0]["ai_recommended_esi"] = 3
    rec = ps.apply_override(rec, new_esi=2, reason="Clinical judgment override.")

    check("first assessment override preserved before reassessment",
          rec["assessments"][0]["override"] is True and rec["assessments"][0]["clinician_final_esi"] == 2)

    patient2 = dict(patient)
    patient2["heart_rate"] = 115
    decision2 = pipeline.assess_patient(patient2)
    rec = ps.add_reassessment(rec, patient2, decision2)

    check("reassessment adds new entry without deleting the override history",
          len(rec["assessments"]) == 2 and rec["assessments"][0]["override"] is True)
    check("waiting room reflects the LATEST assessment's clinician-final ESI (pre-decision fallback)",
          ps.get_displayed_esi(rec) == rec["assessments"][1]["ai_recommended_esi"])

    timeline = ps.assessment_timeline(rec)
    check("assessment_timeline exposes both entries for audit", len(timeline) == 2)
    check("timeline entry 1 still shows override=True", timeline[0]["override"] is True)


# ---------------------------------------------------------------------

def main():
    test_reassessment_workflow()
    test_history_preserved_across_reassessment()
    test_agreement_explanation()
    test_disagreement_explanation_never_fabricates()
    test_override_controls_displayed_esi()
    test_reassessment_after_override_preserves_audit_history()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
