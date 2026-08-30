"""
app.py
======
PatientTriage.ai — Streamlit dashboard.

DISCLAIMER shown throughout the app:
"This prototype is an AI decision-support system for research/demo
purposes and is not a medical device or a substitute for clinical
judgment."
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import streamlit as st

from src.pipeline import get_pipeline
from src import audit
from src import patient_state as ps
from src.explainability import (
    shap_explain_model1, clinical_evidence_model2,
    build_agreement_explanation, build_disagreement_explanation,
    extract_history_evidence,
)
from src.preprocessing import full_preprocess, build_numeric_matrix, build_text_corpus
from src.monitoring import simulate_wait_drift, detect_deterioration, assess_waiting_patient, MAX_SAFE_WAIT_MINUTES
from src.surge_simulation import run_surge_simulation
from src.verification import PRIORITY_GROUP
from demo_patients import DEMO_PATIENTS

st.set_page_config(page_title="PatientTriage.ai", layout="wide", page_icon="🏥")

DISCLAIMER = ("⚠️ **This prototype is an AI decision-support system for research/demo purposes and is "
              "not a medical device or a substitute for clinical judgment.** All patient data on this "
              "page is synthetic/simulated.")


# ---------------------------------------------------------------- state ---
def init_state():
    if "patients" not in st.session_state:
        st.session_state.patients = {}   # patient_id -> record (see src/patient_state.py)
    if "surge_active" not in st.session_state:
        st.session_state.surge_active = False
    if "total_main_ed_beds" not in st.session_state:
        st.session_state.total_main_ed_beds = 4
    if "pipeline_loaded" not in st.session_state:
        st.session_state.pipeline_loaded = False


init_state()

MODEL1_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model1", "model1.joblib")
MODEL2_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model2", "model2.joblib")
VEC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "text_vectorizers.joblib")


def models_exist() -> bool:
    return os.path.exists(MODEL1_PATH) and os.path.exists(MODEL2_PATH) and os.path.exists(VEC_PATH)


if not models_exist():
    # First-run convenience: train automatically instead of erroring and
    # telling the user to run train.py themselves first. This runs once —
    # after training, models/ is populated and every subsequent launch
    # skips straight to loading them (see models_exist() check above).
    with st.spinner("First-time setup: training Model 1 and Model 2 on the synthetic dataset "
                     "(only needed once — this takes about a minute)..."):
        import train as _train_module
        _train_module.main()
    st.toast("Models trained and saved. Loading dashboard...", icon="✅")

try:
    pipeline = get_pipeline()
    st.session_state.pipeline_loaded = True
except Exception as e:
    st.error(f"Could not load models even after training. Try running `python train.py` manually "
             f"from the patienttriage folder and check for errors. ({e})")
    st.stop()


# ------------------------------------------------------------- sidebar ---
PAGES = ["Dashboard", "New Patient", "Waiting Room", "Audit Log"]
PAGE_SLUGS = {"Dashboard": "dashboard", "New Patient": "new_patient", "Waiting Room": "waiting_room",
            "Audit Log": "audit_log"}
SLUG_TO_PAGE = {v: k for k, v in PAGE_SLUGS.items()}


def navigate_to(page_name: str):
    # Stash the request instead of writing st.session_state["nav_page"]
    # directly here — that key belongs to the radio widget below, and
    # Streamlit raises an exception if you assign to a widget's key
    # AFTER that widget has already run earlier in the same script pass.
    st.session_state["nav_page_request"] = page_name
    st.query_params["page"] = PAGE_SLUGS[page_name]


# Apply any pending navigation request BEFORE the radio widget is created
if "nav_page_request" in st.session_state:
    st.session_state["nav_page"] = st.session_state.pop("nav_page_request")
elif "nav_page" not in st.session_state:
    slug = st.query_params.get("page")
    st.session_state["nav_page"] = SLUG_TO_PAGE.get(slug, "Dashboard")

st.sidebar.title("🏥 PatientTriage.ai")
page = st.sidebar.radio("Navigate", PAGES, key="nav_page")
st.query_params["page"] = PAGE_SLUGS[page]  # keep the URL in sync with manual sidebar clicks too
st.sidebar.markdown("---")
st.sidebar.caption(DISCLAIMER)


def priority_color(group):
    return {"CRITICAL": "🔴", "URGENT": "🟠", "NON-URGENT": "🟢"}.get(group, "⚪")


def latest_decision(rec):
    """The TriageDecision object from the most recent assessment."""
    return ps.latest_assessment(rec)["decision"]


def displayed_priority_group(rec):
    """Spec Section 3A: priority group must follow the DISPLAYED (clinician-
    final if set, else AI) ESI — not always the raw AI recommendation."""
    return PRIORITY_GROUP[ps.get_displayed_esi(rec)]


def render_decision(decision, patient):
    col1, col2, col3 = st.columns(3)
    col1.metric("Final ESI (AI-assessed triage priority)", f"ESI {decision.final_esi}")
    col2.metric("Priority Group", f"{priority_color(decision.priority_group)} {decision.priority_group}")
    col3.metric("Uncertainty", decision.uncertainty.level)

    st.markdown(f"**Verification status:** `{decision.verification_status}`  "
                f"(agreement tier: `{decision.agreement_tier}`)"
                + ("  — 🔎 **NURSE REVIEW REQUIRED**" if decision.nurse_review_required else ""))
    if decision.agreement_tier == "MAJOR_DISAGREE":
        st.error("⚠️ MAJOR DISAGREEMENT — the two models differ by 2+ ESI levels. "
                 "This is a stronger signal than an ordinary disagreement and always forces nurse review.")

    # ---- build real explanations (SHAP for Model 1, evidence-presence for Model 2) ----
    try:
        raw_df = pd.DataFrame([patient])
        df = full_preprocess(raw_df)
        Xn = build_numeric_matrix(df).values
        corpus = build_text_corpus(df)
        Xt1 = pipeline.vectorizers.transform_model1(corpus)
        word_vocab = pipeline.vectorizers.word_tfidf.get_feature_names_out()
        exp1 = shap_explain_model1(pipeline.model1, Xn, Xt1, patient, word_vocab)
    except Exception as e:
        exp1 = {"available": False, "factors": [], "limitation": f"Explanation unavailable: {e}"}
    exp2 = clinical_evidence_model2(patient)
    history_evidence = extract_history_evidence(patient.get("history"))

    st.markdown("---")
    if decision.agreement:
        st.success("✓ MODELS AGREE")
        st.write(f"**Why the system recommends ESI {decision.final_esi}:**")
        for line in build_agreement_explanation(decision.final_esi, exp1["factors"], exp2["evidence"]):
            st.write(f"- {line}")
    else:
        st.warning("⚠ MODEL DISAGREEMENT")
        dis = build_disagreement_explanation(decision.model1_result, decision.model2_result,
                                              exp1["factors"], exp2["evidence"])
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.write(f"**Model 1 — Physiological/structured assessment: ESI {dis['model1_side']['esi']}**")
            for f in dis["model1_side"]["factors"]:
                st.write(f"- {f}")
        with dcol2:
            st.write(f"**Model 2 — Clinical/NLP assessment: ESI {dis['model2_side']['esi']}**")
            for e in dis["model2_side"]["evidence"]:
                st.write(f'- narrative includes "{e}"')
        st.info(f"**Why they disagree:** {dis['reason']}")

    if history_evidence:
        st.write("**Relevant historical evidence:**")
        for h in history_evidence:
            st.write(f"- {h}")
        st.caption("Historical findings above contributed in combination with the current presentation — "
                   "not as a standalone cause of this assessment.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("MODEL 1 EXPLANATION — XGBoost")
        st.write(f"Backend: {decision.model1_result['backend']}")
        st.write(f"Prediction: **ESI {decision.model1_result['esi_prediction']}**  "
                 f"(probability {decision.model1_result['confidence']:.0%}, "
                 f"top-2 margin {decision.model1_result.get('top2_margin', 0):.2f})")
        st.bar_chart(pd.Series(decision.model1_result["probabilities"], name="P(ESI)"))
        # if exp1["available"]:
        #     st.write("**Top contributing factors (real SHAP attribution):**")
        #     for f in exp1["factors"]:
        #         st.write(f"- {f['text']}")
        # else:
        #     st.caption(f"⚠ {exp1['limitation']}")
    with c2:
        st.subheader("MODEL 2 EXPLANATION — Clinical NLP")
        st.write(f"Backend: {decision.model2_result['backend']}")
        st.write(f"Prediction: **ESI {decision.model2_result['esi_prediction']}**  "
                 f"(probability {decision.model2_result['confidence']:.0%}, "
                 f"top-2 margin {decision.model2_result.get('top2_margin', 0):.2f})")
        st.bar_chart(pd.Series(decision.model2_result["probabilities"], name="P(ESI)"))
        # st.write("**Important clinical evidence (phrases detected in narrative):**")
        # if exp2["evidence"]:
        #     for e in exp2["evidence"]:
        #         st.write(f'- "{e}"')
        # else:
        #     st.write("- No curated clinical-evidence phrases matched in the narrative text.")
        # st.caption(f"⚠ {exp2['limitation']}")

    if decision.safety_flags:
        st.error("**Safety layer flags triggered:**")
        for f in decision.safety_flags:
            st.write(f"- `{f.rule_id}`: {f.description}")

    if decision.missing_fields:
        st.warning(f"**Missing intake information:** {', '.join(decision.missing_fields)}")
        st.caption("Missing data reduces confidence in this assessment — see uncertainty reasons below.")

    st.info("**Uncertainty reasons:**\n" + "\n".join(f"- {r}" for r in decision.uncertainty.reasons))
    st.caption("The AI never provides a disease diagnosis — only an AI-assessed triage priority "
               "and the intake factors that most influenced it.")


# ------------------------------------------------------------ Dashboard ---
if page == "Dashboard":
    st.title("Dashboard")
    st.caption(DISCLAIMER)

    all_recs = list(st.session_state.patients.values())
    all_decisions = [latest_decision(r) for r in all_recs]
    total = len(all_decisions)
    critical = sum(1 for r in all_recs if displayed_priority_group(r) == "CRITICAL")
    urgent = sum(1 for r in all_recs if displayed_priority_group(r) == "URGENT")
    nonurgent = sum(1 for r in all_recs if displayed_priority_group(r) == "NON-URGENT")
    uncertain = sum(1 for d in all_decisions if d.uncertainty.level == "HIGH")
    disagreements = sum(1 for d in all_decisions if not d.agreement)
    overrides = sum(1 for e in audit.read_audit_log() if e.get("event_type") == "CLINICIAN_OVERRIDE")

    cols = st.columns(7)
    for c, (label, val) in zip(cols, [
        ("Total patients", total), ("🔴 Critical", critical), ("🟠 Urgent", urgent),
        ("🟢 Non-urgent", nonurgent), ("High uncertainty", uncertain),
        ("Model disagreements", disagreements), ("Clinician overrides", overrides),
    ]):
        c.metric(label, val)

    if total == 0:
        st.info("No patients assessed yet this session. Go to **New Patient** to assess one, "
                "or **Surge Simulation** to load the demonstration cohort.")
    else:
        df_view = pd.DataFrame([{
            "Patient ID": pid, "Final ESI": ps.get_displayed_esi(rec),
            "Override?": "✓" if ps.is_overridden(rec) else "",
            "AI recommendation": ps.latest_assessment(rec)["ai_recommended_esi"],
            "Priority": displayed_priority_group(rec),
            "Status": latest_decision(rec).verification_status,
            "Uncertainty": latest_decision(rec).uncertainty.level,
            "Reassessments": sum(1 for a in rec["assessments"] if a["reassessment"]),
            "Nurse review?": "YES" if latest_decision(rec).nurse_review_required else "no",
        } for pid, rec in st.session_state.patients.items()])
        st.dataframe(df_view, use_container_width=True)


# ------------------------------------------------------------ New Patient
elif page == "New Patient":
    st.title("New Patient Intake")
    st.caption(DISCLAIMER)

    st.subheader("1. Patient ID")
    st.caption("Enter a Patient ID and search — history is retrieved automatically if it exists. "
               "Try `PT-1001` (relevant cardiac history), `PT-1002` (unrelated history), `PT-1005` "
               "(frequent ED visits), a made-up ID like `PT-9999` (no record), or "
               "`SIMULATE-DB-OUTAGE` (simulated retrieval failure).")
    id_col1, id_col2 = st.columns([3, 1])
    with id_col1:
        pid_search = st.text_input("Patient ID", value=st.session_state.get("last_pid_search", ""),
                                    label_visibility="collapsed", placeholder="e.g. PT-1001")
    with id_col2:
        search_clicked = st.button("🔍 SEARCH", type="primary", use_container_width=True)

    if search_clicked and pid_search.strip():
        from database.repository import PatientRepository, HistoryStatus
        from src.history_context import build_clinical_text_block, summarize_temporal_features
        repo = PatientRepository()
        bundle = repo.get_history(pid_search.strip())
        st.session_state["last_pid_search"] = pid_search.strip()
        st.session_state["history_bundle"] = bundle
        st.session_state["history_text_prefill"] = build_clinical_text_block(bundle)
        st.session_state["history_temporal"] = summarize_temporal_features(bundle)

    bundle = st.session_state.get("history_bundle")
    if bundle is not None and bundle.patient_id == st.session_state.get("last_pid_search", ""):
        from database.repository import HistoryStatus
        if bundle.history_status == HistoryStatus.FOUND:
            st.success(f"✓ Patient found — Medical history: **AVAILABLE**  \n"
                       f"Previous encounters: {len(bundle.encounters)} · "
                       f"Active conditions: {len([d for d in bundle.diagnoses if d.get('status') in ('active','chronic')])} · "
                       f"Last recorded encounter: {bundle.history_last_updated or 'n/a'}  \n"
                       f"History automatically loaded below.")
            with st.expander("📋 VIEW FULL HISTORY (timeline)"):
                timeline = []
                for e in bundle.encounters:
                    timeline.append((e.get("event_date", ""), f"Encounter: {e.get('chief_complaint','')}"
                                      + (f" (ESI {e['esi_recorded']})" if e.get("esi_recorded") else "")))
                for d in bundle.diagnoses:
                    timeline.append((d.get("event_date", ""), f"Diagnosis: {d['diagnosis']} ({d.get('status','')})"))
                for n in bundle.notes:
                    timeline.append((n.get("event_date", ""), f"Note: {n.get('note','')}"))
                for row_date, label in sorted(timeline, reverse=True):
                    st.write(f"**{row_date}** — {label}")
        # elif bundle.history_status == HistoryStatus.NO_RECORD:
            # st.info("ℹ️ Patient found, but no historical clinical records are available "
            #         "(confirmed no-history, not a retrieval failure). Proceeding using current "
            #         "encounter information.")
        else:  # UNAVAILABLE
            # st.warning("⚠ Historical record unavailable — retrieval failed (simulated database/service "
            #            "outage). Proceeding using current encounter information only. This is NOT the "
                    #    "same as the patient having no history — it means we don't know.")
            st.warning("⚠ New Patient! No historical record available.")

    st.subheader("2. Current Encounter")

    reassess_prefill = st.session_state.get("reassess_prefill")
    is_reassessment = reassess_prefill is not None
    if is_reassessment:
        prev = reassess_prefill["previous_input"]
        st.info(f"🔁 **Reassessing patient {reassess_prefill['patient_id']}** — previous assessment at "
                f"{reassess_prefill['previous_timestamp']} recommended ESI {reassess_prefill['previous_ai_esi']}"
                + (f" (clinician set ESI {reassess_prefill['previous_clinician_esi']})"
                   if reassess_prefill['previous_clinician_esi'] is not None else "")
                + ". Fields below are pre-filled with the previous values — update anything that has changed.")
        if st.button("✖ Cancel reassessment / start a new patient instead"):
            del st.session_state["reassess_prefill"]
            st.rerun()
    else:
        prev = {}

    def _prefill(field, default):
        return prev.get(field, default)

    with st.form("intake_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            default_pid = reassess_prefill["patient_id"] if is_reassessment else (
                st.session_state.get("last_pid_search") or f"P-{len(st.session_state.patients) + 1:03d}")
            pid = st.text_input("Patient ID" + (" (locked — reassessment)" if is_reassessment else " (auto-filled from search)"),
                                 value=default_pid, disabled=is_reassessment)
            age = st.number_input("Age", 0, 120, int(_prefill("age", 40)))
            hr = st.number_input("Heart rate (bpm)", 0, 250, int(_prefill("heart_rate", 80) or 80))
            rr = st.number_input("Respiratory rate (/min)", 0, 60, int(_prefill("resp_rate", 16) or 16))
        with c2:
            sbp = st.number_input("Systolic BP (mmHg)", 0, 260, int(_prefill("sbp", 118) or 118))
            dbp = st.number_input("Diastolic BP (mmHg)", 0, 160, int(_prefill("dbp", 76) or 76))
            spo2 = st.number_input("SpO2 (%)", 0, 100, int(_prefill("spo2", 98) or 98))
            temp = st.number_input("Temperature (°C)", 30.0, 43.0, float(_prefill("temperature", 37.0) or 37.0), step=0.1)
        with c3:
            pain = st.slider("Pain score (0-10)", 0, 10, int(_prefill("pain_score", 2) or 2))
            complaint = st.text_area("Chief complaint", _prefill("chief_complaint", "mild sore throat"))
            symptoms = st.text_area("Symptoms", _prefill("symptoms", "none reported"))
            default_history = _prefill("history", st.session_state.get("history_text_prefill", "")) or ""
            history = st.text_area("Medical history (auto-filled, editable)", value=default_history)

        missing_toggle = st.multiselect(
            "Mark fields as UNAVAILABLE at intake (simulates real-world missing data)",
            ["Heart rate", "Respiratory rate", "Systolic BP", "Diastolic BP", "SpO2", "Temperature", "History"],
        )
        submit_label = "RUN REASSESSMENT" if is_reassessment else "ASSESS PATIENT"
        submitted = st.form_submit_button(submit_label, type="primary")

    if submitted:
        patient = dict(
            patient_id=pid, age=age,
            heart_rate=None if "Heart rate" in missing_toggle else hr,
            resp_rate=None if "Respiratory rate" in missing_toggle else rr,
            sbp=None if "Systolic BP" in missing_toggle else sbp,
            dbp=None if "Diastolic BP" in missing_toggle else dbp,
            spo2=None if "SpO2" in missing_toggle else spo2,
            temperature=None if "Temperature" in missing_toggle else temp,
            pain_score=pain, chief_complaint=complaint, symptoms=symptoms,
            history=None if ("History" in missing_toggle or not history.strip()) else history,
            has_history_info=0 if ("History" in missing_toggle or not history.strip()) else 1,
        )
        decision = pipeline.assess_patient(patient)

        if is_reassessment and pid in st.session_state.patients:
            rec = st.session_state.patients[pid]
            rec = ps.add_reassessment(rec, patient, decision)
            st.session_state.patients[pid] = rec
            del st.session_state["reassess_prefill"]
            st.success(f"Reassessment complete for {pid} — both models re-run on updated data.")
        else:
            rec = ps.create_patient_record(pid, patient, decision,
                                            all_patient_records=st.session_state.patients,
                                            total_main_ed_beds=st.session_state.total_main_ed_beds)
            st.session_state.patients[pid] = rec
            st.success(f"Assessment complete for {pid}.")

        latest = ps.latest_assessment(rec)
        audit.log_triage_decision(decision, patient, assessment_id=latest["assessment_id"],
                                   previous_assessment_id=latest["previous_assessment_id"],
                                   reassessment=latest["reassessment"])
        st.session_state["last_assessed_pid"] = pid

    # show result + accept/override UI for the most recently assessed patient
    last_pid = st.session_state.get("last_assessed_pid")
    if last_pid and last_pid in st.session_state.patients:
        rec = st.session_state.patients[last_pid]
        decision = latest_decision(rec)
        st.markdown("---")
        st.header(f"TRIAGE RESULT — {last_pid}")
        if len(rec["assessments"]) > 1:
            st.caption(f"This is assessment #{len(rec['assessments'])} for this patient "
                       f"(encounter {rec['encounter_id']}) — see the Waiting Room page for the full timeline.")
        render_decision(decision, ps.latest_assessment(rec)["input_snapshot"])

        st.markdown("---")
        st.subheader("AI Recommendation")
        st.write(f"AI: **ESI {decision.final_esi}**")
        colA, colB = st.columns(2)
        with colA:
            if st.button("✅ Accept", key=f"accept_{last_pid}_{len(rec['assessments'])}"):
                previous_pathway = rec["pathway"]
                rec = ps.apply_accept(rec, all_patient_records=st.session_state.patients,
                                       total_main_ed_beds=st.session_state.total_main_ed_beds)
                st.session_state.patients[last_pid] = rec
                latest = ps.latest_assessment(rec)
                audit.log_clinician_override(
                    patient_id=last_pid, ai_esi=latest["ai_recommended_esi"],
                    ai_confidence=decision.model1_result["confidence"],
                    model1_result=decision.model1_result, model2_result=decision.model2_result,
                    clinician_esi=latest["clinician_final_esi"], reason=latest["override_reason"],
                    input_completeness=1.0 - len(decision.missing_fields) / 7.0,
                    safety_flags=[f.rule_id for f in decision.safety_flags],
                    assessment_id=latest["assessment_id"],
                )
                st.success(f"Accepted and logged. Patient moved to {ps.PATHWAY_LABELS[rec['pathway']]} "
                           f"— Waiting Room now shows ESI {latest['clinician_final_esi']}.")
                if previous_pathway != rec["pathway"]:
                    audit.log_pathway_change(last_pid, previous_pathway, rec["pathway"],
                                              reason="Final ESI accepted — auto-moved to matching pathway.",
                                              assessment_id=latest["assessment_id"])
                    if previous_pathway == ps.PATHWAY_MAIN_ED:
                        audit.log_main_ed_bed_released(last_pid, rec["pathway"], latest["clinician_final_esi"])
                        next_candidate = ps.find_next_main_ed_candidate(st.session_state.patients, exclude_patient_id=last_pid)
                        if next_candidate:
                            st.info(f"📣 Main ED bed freed. Next highest-priority eligible patient: "
                                    f"**{next_candidate}** — visit the Waiting Room to move them in.")
        with colB:
            with st.form(f"override_form_{last_pid}_{len(rec['assessments'])}"):
                new_esi = st.selectbox("Override ESI level", [1, 2, 3, 4, 5], index=decision.final_esi - 1)
                reason = st.text_input("Reason for override", "")
                override_submit = st.form_submit_button("Override")
                if override_submit:
                    if not reason.strip():
                        st.error("A reason is required to log an override.")
                    else:
                        previous_pathway = rec["pathway"]
                        rec = ps.apply_override(rec, new_esi, reason,
                                                 all_patient_records=st.session_state.patients,
                                                 total_main_ed_beds=st.session_state.total_main_ed_beds)
                        st.session_state.patients[last_pid] = rec
                        latest = ps.latest_assessment(rec)
                        audit.log_clinician_override(
                            patient_id=last_pid, ai_esi=latest["ai_recommended_esi"],
                            ai_confidence=decision.model1_result["confidence"],
                            model1_result=decision.model1_result, model2_result=decision.model2_result,
                            clinician_esi=new_esi, reason=reason,
                            input_completeness=1.0 - len(decision.missing_fields) / 7.0,
                            safety_flags=[f.rule_id for f in decision.safety_flags],
                            assessment_id=latest["assessment_id"],
                        )
                        st.success(f"Override to ESI {new_esi} logged. Patient moved to "
                                   f"{ps.PATHWAY_LABELS[rec['pathway']]} (AI recommendation ESI "
                                   f"{latest['ai_recommended_esi']} remains stored for audit).")
                        if previous_pathway != rec["pathway"]:
                            audit.log_pathway_change(last_pid, previous_pathway, rec["pathway"],
                                                      reason="Final ESI overridden — auto-moved to matching pathway.",
                                                      assessment_id=latest["assessment_id"])
                            if previous_pathway == ps.PATHWAY_MAIN_ED:
                                audit.log_main_ed_bed_released(last_pid, rec["pathway"], new_esi)
                                next_candidate = ps.find_next_main_ed_candidate(st.session_state.patients, exclude_patient_id=last_pid)
                                if next_candidate:
                                    st.info(f"📣 Main ED bed freed. Next highest-priority eligible patient: "
                                            f"**{next_candidate}** — visit the Waiting Room to move them in.")

        with st.expander("📜 Assessment / reassessment history for this patient"):
            for i, a in enumerate(ps.assessment_timeline(rec), 1):
                tag = "REASSESSMENT" if a["reassessment"] else "INITIAL"
                override_tag = f" — clinician override to ESI {a['clinician_final_esi']}" if a["override"] else (
                    f" — accepted ESI {a['clinician_final_esi']}" if a["clinician_final_esi"] is not None else " — pending nurse decision")
                st.write(f"**#{i} [{tag}]** {a['timestamp']} — AI: ESI {a['ai_recommended_esi']}{override_tag}")


# --------------------------------------------------------------- Waiting Room
elif page == "Waiting Room":
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=1000, key="waiting_room_clock_tick")  # live timer tick, no button needed

    st.title("Waiting Room")
    st.caption(DISCLAIMER)
    st.caption("⏱ Waiting times below are REAL elapsed time (current time minus real arrival time) — "
               "not a manually-advanced counter. They update automatically and are unaffected by page "
               "navigation or refresh. Safe-wait ceilings: "
               + ", ".join(f"ESI {k} \u2192 {v} min" for k, v in MAX_SAFE_WAIT_MINUTES.items()))

    if "total_main_ed_beds" not in st.session_state:
        st.session_state.total_main_ed_beds = 4

    top1, top2 = st.columns([1, 2])
    with top1:
        st.session_state.surge_active = st.checkbox("🚨 Surge Mode Active", value=st.session_state.surge_active)
    with top2:
        st.session_state.total_main_ed_beds = st.number_input(
            "Total Main ED beds (operational capacity)", 1, 30, st.session_state.total_main_ed_beds)

    bed_state = ps.main_ed_bed_state(st.session_state.patients, st.session_state.total_main_ed_beds)
    bc1, bc2, bc3 = st.columns(3)
    bc1.metric("Total Main ED beds", bed_state["total_main_ed_beds"])
    bc2.metric("Occupied", bed_state["occupied_main_ed_beds"])
    bc3.metric("Available", bed_state["available_main_ed_beds"])
    if st.session_state.surge_active and bed_state["available_main_ed_beds"] == 0:
        st.error("🚨 SURGE — Main ED at full capacity. Reassessing Main ED patients may free a bed, "
                 "but ESI must be genuinely reassessed — surge pressure alone never changes ESI.")

    if not st.session_state.patients:
        st.info("No patients currently tracked. Assess a patient on **New Patient**, "
                "or load the demonstration cohort via **Surge Simulation**.")
    else:
        def render_patient_row(pid, rec, section_label, show_bed_action=False):
            latest = ps.latest_assessment(rec)
            decision = latest["decision"]
            displayed_esi = ps.get_displayed_esi(rec)
            vitals_worse, vitals_reasons = ps.check_vitals_deterioration(rec)
            wait_breached, wait_reason = ps.check_real_wait_breach(displayed_esi, rec, MAX_SAFE_WAIT_MINUTES)
            # Product requirement: the wait-time-ceiling "timer" alert should
            # only ever surface for ESI-3 patients (Vertical Care) — ESI 1/2
            # patients are expected to already be in Main ED being treated,
            # so flagging them for an unmet "safe wait" ceiling is just noise.
            # This ONLY suppresses the on-screen alert for other ESI levels;
            # it does not touch ps.check_real_wait_breach()/MAX_SAFE_WAIT_MINUTES
            # itself, since Surge Simulation's per-ESI breach stats and
            # tests/test_realtime_and_pathways.py both depend on that function
            # working across all ESI levels.
            if displayed_esi != 3:
                wait_breached = False
            vtimer = ps.check_vertical_care_timer(rec)
            needs_attention = wait_breached or vitals_worse or vtimer["breached"]

            header = f"{'🚨 ' if needs_attention else ''}{pid} — ESI {displayed_esi} — {ps.format_elapsed(rec)}" + (
                " (Clinician Override)" if ps.is_overridden(rec) else "")

            # NOTE on the row toggle below: this page runs st_autorefresh
            # every second to keep wait timers live, which forces a full
            # script rerun on that cadence. Streamlit's built-in st.expander
            # only keeps its open/closed state in the browser by default, so
            # every autorefresh-driven rerun was recreating it from scratch
            # and snapping it shut (or back open) regardless of what the
            # nurse had just clicked. Rather than lean on st.expander's own
            # state handling, we track "open or closed" ourselves in
            # st.session_state — the same proven pattern already used for
            # the "Modify ESI" toggle a few lines down in this file — so a
            # click is the ONLY thing that can change it, immune to the
            # 1-second refresh tick.
            row_open_key = f"row_open_{pid}"
            if row_open_key not in st.session_state:
                st.session_state[row_open_key] = needs_attention  # auto-open on first sight only

            arrow = "🔽" if st.session_state[row_open_key] else "▶️"
            if st.button(f"{arrow} {header}", key=f"toggle_row_{pid}", use_container_width=True):
                st.session_state[row_open_key] = not st.session_state[row_open_key]

            if st.session_state[row_open_key]:
              with st.container(border=True):
                vitals = latest["input_snapshot"]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("FINAL ESI", displayed_esi,
                              delta="Clinician Override" if ps.is_overridden(rec) else None, delta_color="off")
                    st.write(f"AI recommendation: **ESI {latest['ai_recommended_esi']}**")
                    st.write(f"Pathway: **{ps.PATHWAY_LABELS[rec['pathway']]}**")
                with col2:
                    st.write(f"**Waiting:** {ps.format_elapsed(rec)}  ({ps.elapsed_seconds(rec)/60:.0f} min)")
                    st.write(f"**Last assessment:** {latest['timestamp']}")
                    st.write(f"**Models:** {'AGREEMENT' if decision.agreement else 'DISAGREEMENT'} "
                             f"({decision.agreement_tier})")
                with col3:
                    st.write(f"**Uncertainty:** {decision.uncertainty.level}")
                    if decision.safety_flags:
                        st.write(f"**Safety alerts:** {', '.join(f.rule_id for f in decision.safety_flags)}")

                st.write(f"**Latest vitals:** HR {vitals.get('heart_rate')}, "
                         f"BP {vitals.get('sbp')}/{vitals.get('dbp')}, SpO2 {vitals.get('spo2')}%, "
                         f"RR {vitals.get('resp_rate')}, Temp {vitals.get('temperature')}\u00b0C")

                if wait_breached:
                    st.error(f"⏰ {wait_reason}")
                if vitals_worse:
                    st.error("⚠️ DETERIORATION DETECTED since last assessment:")
                    for r in vitals_reasons:
                        st.write(f"- {r}")

                # Dedicated Vertical Care 30-min timer — ESI-3 patients in
                # Waiting/Vertical Care ONLY (spec requirement 3)
                if vtimer["applicable"]:
                    tcol1, tcol2 = st.columns([2, 1])
                    with tcol1:
                        mins, secs = divmod(int(vtimer["remaining_seconds"]), 60)
                        if vtimer["breached"]:
                            st.error(f"🔴 Vertical Care 30-min timer EXPIRED — patient needs reassessment. "
                                     f"(elapsed {int(vtimer['elapsed_seconds']//60)} min)")
                        else:
                            st.info(f"⏳ Vertical Care reassessment timer: **{mins:02d}:{secs:02d}** remaining")
                    with tcol2:
                        if st.button("🔄 Reset Timer", key=f"reset_vtimer_{pid}"):
                            rec = ps.reset_vertical_timer(rec)
                            st.session_state.patients[pid] = rec
                            audit.log_event({"event_type": "VERTICAL_TIMER_RESET", "patient_id": pid})
                            st.rerun()

                if len(rec["assessments"]) > 1:
                    st.caption(f"{len(rec['assessments'])} assessments on record for this encounter "
                               f"({rec['encounter_id']}).")

                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    btn_label = "🔁 UPDATE / REASSESS ESI" if show_bed_action else "🔁 REASSESS PATIENT"
                    if st.button(btn_label, key=f"reassess_btn_{pid}", type="primary", use_container_width=True):
                        st.session_state["reassess_prefill"] = {
                            "patient_id": pid,
                            "previous_input": dict(latest["input_snapshot"]),
                            "previous_timestamp": latest["timestamp"],
                            "previous_ai_esi": latest["ai_recommended_esi"],
                            "previous_clinician_esi": latest["clinician_final_esi"],
                        }
                        st.session_state["last_pid_search"] = pid
                        navigate_to("New Patient")
                        st.rerun()
                with btn_col2:
                    modify_open_key = f"modify_esi_open_{pid}"
                    if st.button("✏️ Modify ESI", key=f"modify_esi_btn_{pid}", use_container_width=True):
                        st.session_state[modify_open_key] = not st.session_state.get(modify_open_key, False)

                if st.session_state.get(f"modify_esi_open_{pid}", False):
                    with st.form(f"modify_esi_form_{pid}"):
                        st.caption("Enter the ESI directly from your own clinical judgment — this skips "
                                   "re-running the AI models.")
                        manual_esi = st.selectbox("New ESI", [1, 2, 3, 4, 5], index=displayed_esi - 1,
                                                   key=f"manual_esi_select_{pid}")
                        manual_reason = st.text_input("Reason", key=f"manual_esi_reason_{pid}")
                        save_clicked = st.form_submit_button("💾 Save ESI")
                        if save_clicked:
                            if not manual_reason.strip():
                                st.error("A reason is required to save a manually-entered ESI.")
                            else:
                                previous_pathway = rec["pathway"]
                                rec = ps.apply_manual_esi_edit(rec, manual_esi, manual_reason)
                                audit.log_clinician_override(
                                    patient_id=pid, ai_esi=latest["ai_recommended_esi"],
                                    ai_confidence=decision.model1_result["confidence"],
                                    model1_result=decision.model1_result, model2_result=decision.model2_result,
                                    clinician_esi=manual_esi, reason=f"[MANUAL ESI EDIT] {manual_reason}",
                                    input_completeness=1.0 - len(decision.missing_fields) / 7.0,
                                    safety_flags=[f.rule_id for f in decision.safety_flags],
                                    assessment_id=latest["assessment_id"],
                                )
                                if st.session_state.surge_active:
                                    # SURGE: move immediately, no separate confirm step
                                    rec = ps.auto_assign_pathway_from_esi(
                                        rec, all_patient_records=st.session_state.patients,
                                        total_main_ed_beds=st.session_state.total_main_ed_beds)
                                    st.session_state.patients[pid] = rec
                                    audit.log_pathway_change(pid, previous_pathway, rec["pathway"],
                                                              reason="Manual ESI edit during surge — auto-moved.",
                                                              assessment_id=latest["assessment_id"])
                                    st.session_state[modify_open_key] = False
                                    st.success(f"ESI updated to {manual_esi}. Patient moved to "
                                               f"{ps.PATHWAY_LABELS[rec['pathway']]} (surge mode: automatic).")
                                else:
                                    # QUIET SHIFT: save the ESI but wait for
                                    # explicit confirmation before moving depts
                                    st.session_state.patients[pid] = rec
                                    st.session_state[modify_open_key] = False
                                    st.success(f"ESI updated to {manual_esi}. Confirm below to move departments.")
                                st.rerun()

                # Quiet-shift explicit "Send to new Dept" — appears only
                # when the saved final ESI no longer matches the dept the
                # patient is still physically shown in (i.e. right after a
                # quiet-shift Modify-ESI save that hasn't been confirmed yet)
                # Bed-capacity aware, same rule apply_accept/apply_override/
                # auto_assign_pathway_from_esi use — so this banner/button
                # always matches where the patient will actually land
                # (e.g. it says "SEND TO WAITING/VERTICAL", not "MAIN ED",
                # when Main ED is already full for an ESI-1/2 patient).
                suggested_now = ps.resolve_pathway(displayed_esi, all_patient_records=st.session_state.patients,
                                                    total_main_ed_beds=st.session_state.total_main_ed_beds,
                                                    exclude_patient_id=pid)
                if not st.session_state.surge_active and suggested_now != rec["pathway"]:
                    st.warning(f"Final ESI ({displayed_esi}) suggests **{ps.PATHWAY_LABELS[suggested_now]}**, "
                               f"but the patient is still shown in **{ps.PATHWAY_LABELS[rec['pathway']]}**.")
                    if st.button(f"➡️ SEND TO {ps.PATHWAY_LABELS[suggested_now].upper()}", key=f"send_dept_{pid}"):
                        previous_pathway = rec["pathway"]
                        rec = ps.auto_assign_pathway_from_esi(
                            rec, all_patient_records=st.session_state.patients,
                            total_main_ed_beds=st.session_state.total_main_ed_beds)
                        st.session_state.patients[pid] = rec
                        audit.log_pathway_change(pid, previous_pathway, rec["pathway"],
                                                  reason="Nurse-confirmed department move.",
                                                  assessment_id=latest["assessment_id"])
                        if previous_pathway == ps.PATHWAY_MAIN_ED:
                            audit.log_main_ed_bed_released(pid, rec["pathway"], displayed_esi)
                            next_candidate = ps.find_next_main_ed_candidate(st.session_state.patients, exclude_patient_id=pid)
                            if next_candidate:
                                st.info(f"📣 Main ED bed freed. Next highest-priority eligible patient: "
                                        f"**{next_candidate}** — visit their row to move them in.")
                        st.success(f"{pid} moved to {ps.PATHWAY_LABELS[rec['pathway']]}.")
                        st.rerun()

        sections = [
            (ps.PATHWAY_MAIN_ED, "🏥 MAIN ED", True),
            (ps.PATHWAY_WAITING_VERTICAL, "🪑 WAITING / VERTICAL CARE", False),
            (ps.PATHWAY_FAST_NORMAL, "⚡ FAST / NORMAL TREATMENT", False),
        ]
        for pathway_key, label, show_bed_action in sections:
            members = {pid: rec for pid, rec in st.session_state.patients.items() if rec["pathway"] == pathway_key}
            st.subheader(f"{label}  ({len(members)})")
            if not members:
                st.caption("No patients currently in this pathway.")
                continue
            summary_rows = []
            for pid, rec in members.items():
                displayed_esi = ps.get_displayed_esi(rec)
                vitals_worse, _ = ps.check_vitals_deterioration(rec)
                wait_breached, _ = ps.check_real_wait_breach(displayed_esi, rec, MAX_SAFE_WAIT_MINUTES)
                if displayed_esi != 3:
                    wait_breached = False  # "Reassess due" ⏰ column: ESI-3-only, see render_patient_row
                summary_rows.append({
                    "Patient ID": pid, "Final ESI": displayed_esi,
                    "Override?": "✓" if ps.is_overridden(rec) else "",
                    "Waiting": ps.format_elapsed(rec),
                    "Agreement": "AGREE" if ps.latest_assessment(rec)["decision"].agreement else "DISAGREE",
                    "Uncertainty": ps.latest_assessment(rec)["decision"].uncertainty.level,
                    "Deterioration": "⚠️" if vitals_worse else "",
                    "Reassess due": "⏰" if wait_breached else "",
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
            for pid, rec in members.items():
                render_patient_row(pid, rec, label, show_bed_action=show_bed_action)


# ---------------------------------------------------------- Surge Simulation
# elif page == "Surge Simulation":
#     st.title("Surge Simulation")
#     st.caption(DISCLAIMER)
#     st.write("Runs a real discrete-event queueing simulation — patients arrive randomly over an "
#              "8-hour shift and compete for a **fixed number of treatment slots**, pulled by priority "
#              "(most urgent first). Compares **Normal** volume (100 patients/day) against "
#              "**3× Surge** (300 patients/day) using the SAME number of slots in both runs, so you can "
#              "see how a fixed-capacity system behaves as load triples.")
#     st.warning("Note: the system **never lowers medical safety thresholds** under surge. "
#                "The Safety Layer and ESI logic are byte-for-byte identical in both modes — only "
#                "the queueing/workflow outcome differs.")

#     n_servers = st.slider("Treatment slots available (nurses/bays) — same in both modes", 2, 12, 5)

#     if st.button("▶️ START 3× SURGE SIMULATION", type="primary"):
#         # First, run the ML batch (same as before) to show model behavior at volume
#         base_n = len(DEMO_PATIENTS)

#         def run_batch(n):
#             results = []
#             for i in range(n):
#                 base = DEMO_PATIENTS[i % base_n]
#                 p = {k: v for k, v in base.items() if k != "category"}
#                 p = {**p, "patient_id": f"{p['patient_id']}-B{i}"}
#                 d = pipeline.assess_patient(p)
#                 results.append(d)
#             return results

#         with st.spinner("Running normal-volume ML batch..."):
#             normal_results = run_batch(base_n)
#         with st.spinner("Running 3× surge-volume ML batch..."):
#             surge_results = run_batch(base_n * 3)

#         esi_pool = [d.final_esi for d in normal_results] or [3]

#         with st.spinner("Simulating queueing under normal load..."):
#             queue_normal = run_surge_simulation(patients_per_day=100, esi_pool=esi_pool, n_servers=n_servers, seed=1)
#         with st.spinner("Simulating queueing under 3× surge load..."):
#             queue_surge = run_surge_simulation(patients_per_day=300, esi_pool=esi_pool, n_servers=n_servers, seed=1)

#         st.subheader("Queueing behavior: Normal vs. 3× Surge (fixed capacity)")
#         queue_df = pd.DataFrame([
#             {"Mode": "Normal (1×)", "Patients": queue_normal["n_patients"],
#              "Avg wait (min)": round(queue_normal["avg_wait_overall"], 1),
#              "Max wait (min)": round(queue_normal["max_wait_overall"], 1),
#              "Safety wait-breaches": queue_normal["safety_breaches_total"],
#              "Critical": queue_normal["critical_count"], "Urgent": queue_normal["urgent_count"],
#              "Non-urgent": queue_normal["non_urgent_count"]},
#             {"Mode": "Surge (3×)", "Patients": queue_surge["n_patients"],
#              "Avg wait (min)": round(queue_surge["avg_wait_overall"], 1),
#              "Max wait (min)": round(queue_surge["max_wait_overall"], 1),
#              "Safety wait-breaches": queue_surge["safety_breaches_total"],
#              "Critical": queue_surge["critical_count"], "Urgent": queue_surge["urgent_count"],
#              "Non-urgent": queue_surge["non_urgent_count"]},
#         ]).set_index("Mode")
#         st.dataframe(queue_df, use_container_width=True)

#         st.write("**Average wait time by ESI level (minutes) — safe ceiling shown for reference:**")
#         wait_rows = []
#         for esi in range(1, 6):
#             wait_rows.append({
#                 "ESI": esi, "Safe ceiling (min)": MAX_SAFE_WAIT_MINUTES[esi],
#                 "Normal avg wait": round(queue_normal["avg_wait_by_esi"][esi], 1) if queue_normal["avg_wait_by_esi"][esi] else None,
#                 "Surge avg wait": round(queue_surge["avg_wait_by_esi"][esi], 1) if queue_surge["avg_wait_by_esi"][esi] else None,
#                 "Surge breaches": queue_surge["safety_breaches_by_esi"][esi],
#             })
#         st.dataframe(pd.DataFrame(wait_rows), use_container_width=True)

#         if queue_surge["safety_breaches_total"] > queue_normal["safety_breaches_total"]:
#             st.error(f"🚨 Under 3× surge with the SAME {n_servers} treatment slots, "
#                      f"**{queue_surge['safety_breaches_total']} patients** breached their safe-wait "
#                      f"ceiling (vs {queue_normal['safety_breaches_total']} at normal volume) — including "
#                      f"{queue_surge['safety_breaches_by_esi'][1] + queue_surge['safety_breaches_by_esi'][2]} "
#                      f"CRITICAL-priority patients. This is the queueing consequence of fixed capacity "
#                      f"under 3× load, not a change in triage thresholds.")

#         st.subheader("Model behavior at volume (ML batch)")
#         ml_df = pd.DataFrame([
#             {"Mode": "Normal (1×)", "n": len(normal_results),
#              "critical": sum(1 for d in normal_results if d.priority_group == "CRITICAL"),
#              "urgent": sum(1 for d in normal_results if d.priority_group == "URGENT"),
#              "non_urgent": sum(1 for d in normal_results if d.priority_group == "NON-URGENT"),
#              "uncertain": sum(1 for d in normal_results if d.uncertainty.level == "HIGH"),
#              "reassessments_flagged": sum(1 for d in normal_results if d.nurse_review_required)},
#             {"Mode": "Surge (3×)", "n": len(surge_results),
#              "critical": sum(1 for d in surge_results if d.priority_group == "CRITICAL"),
#              "urgent": sum(1 for d in surge_results if d.priority_group == "URGENT"),
#              "non_urgent": sum(1 for d in surge_results if d.priority_group == "NON-URGENT"),
#              "uncertain": sum(1 for d in surge_results if d.uncertainty.level == "HIGH"),
#              "reassessments_flagged": sum(1 for d in surge_results if d.nurse_review_required)},
#         ]).set_index("Mode")
#         st.dataframe(ml_df, use_container_width=True)
#         st.bar_chart(ml_df[["critical", "urgent", "non_urgent"]])

#         for d in normal_results + surge_results:
#             audit.log_triage_decision(d, {"patient_id": d.patient_id})
#         audit.log_event({
#             "event_type": "SURGE_SIMULATION_RUN", "n_servers": n_servers,
#             "normal_avg_wait": queue_normal["avg_wait_overall"], "surge_avg_wait": queue_surge["avg_wait_overall"],
#             "normal_breaches": queue_normal["safety_breaches_total"], "surge_breaches": queue_surge["safety_breaches_total"],
#         })


# --------------------------------------------------------------- Audit Log
elif page == "Audit Log":
    st.title("Audit Log")
    st.caption(DISCLAIMER)
    st.write("Every triage decision, clinician accept/override, and deterioration event is "
             "recorded here (append-only), including model version, both model outputs, "
             "safety flags, and input completeness.")

    events = audit.read_audit_log(limit=1000)
    if not events:
        st.info("No audit events yet.")
    else:
        event_types = sorted(set(e.get("event_type", "UNKNOWN") for e in events))
        filt = st.multiselect("Filter by event type", event_types, default=event_types)
        filtered = [e for e in events if e.get("event_type") in filt]
        st.write(f"Showing {len(filtered)} of {len(events)} events.")
        st.dataframe(pd.DataFrame(filtered), use_container_width=True)

        with st.expander("Raw JSON (most recent 20 events)"):
            for e in filtered[-20:][::-1]:
                st.json(e)

        if st.button("Clear audit log (demo reset)"):
            audit.clear_audit_log()
            st.rerun()
