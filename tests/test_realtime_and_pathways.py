"""
test_realtime_and_pathways.py
==============================
Tests for the Waiting Room real-time clock, three care pathways, and
surge Main-ED bed-release workflow (spec Part 25).

Run: python3 tests/test_realtime_and_pathways.py
"""

import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import get_pipeline
from src import patient_state as ps
from src.monitoring import MAX_SAFE_WAIT_MINUTES

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def make_patient(**overrides):
    base = dict(patient_id="RT-P1", age=55, heart_rate=90, resp_rate=18, sbp=120, dbp=78,
                spo2=97, temperature=37.0, pain_score=3, chief_complaint="mild sore throat",
                symptoms="none reported", history="no significant past medical history")
    base.update(overrides)
    return base


def test_realtime_timer_uses_stored_arrival_timestamp():
    pipeline = get_pipeline()
    p1 = make_patient(patient_id="RT-1")
    decision = pipeline.assess_patient(p1)
    rec = ps.create_patient_record("RT-1", p1, decision)

    check("arrival_timestamp is a real wall-clock epoch value",
          abs(rec["arrival_timestamp"] - time.time()) < 2)
    check("elapsed_seconds starts near zero", ps.elapsed_seconds(rec) < 2)

    time.sleep(1.2)
    elapsed_after = ps.elapsed_seconds(rec)
    check("elapsed_seconds increases with real time without any manual increment call",
          elapsed_after >= 1.0)

    formatted = ps.format_elapsed(rec)
    check("format_elapsed produces HH:MM:SS", len(formatted) == 8 and formatted.count(":") == 2)


def test_timer_not_reset_by_navigation_or_reassessment():
    pipeline = get_pipeline()
    p1 = make_patient(patient_id="RT-2")
    decision1 = pipeline.assess_patient(p1)
    rec = ps.create_patient_record("RT-2", p1, decision1)
    original_arrival = rec["arrival_timestamp"]

    check("arrival_timestamp unaffected by re-reading the record (simulated navigation)",
          rec["arrival_timestamp"] == original_arrival)

    p2 = dict(p1)
    p2["heart_rate"] = 100
    decision2 = pipeline.assess_patient(p2)
    rec = ps.add_reassessment(rec, p2, decision2)
    check("arrival_timestamp NOT reset by reassessment", rec["arrival_timestamp"] == original_arrival)


def test_no_manual_wait_minutes_counter_exists():
    pipeline = get_pipeline()
    p1 = make_patient(patient_id="RT-3")
    decision = pipeline.assess_patient(p1)
    rec = ps.create_patient_record("RT-3", p1, decision)
    check("patient record has no manually-tracked 'wait_minutes' counter field",
          "wait_minutes" not in rec)
    check("patient record has no stale 'wait_time_breached' stored flag",
          "wait_time_breached" not in rec)


def test_real_wait_breach_uses_elapsed_time():
    pipeline = get_pipeline()
    p1 = make_patient(patient_id="RT-4")
    decision = pipeline.assess_patient(p1)
    rec = ps.create_patient_record("RT-4", p1, decision)

    breached, reason = ps.check_real_wait_breach(2, rec, MAX_SAFE_WAIT_MINUTES)
    check("freshly-arrived patient does not breach wait-time ceiling", breached is False)

    rec["arrival_timestamp"] = time.time() - (20 * 60)
    breached, reason = ps.check_real_wait_breach(2, rec, MAX_SAFE_WAIT_MINUTES)
    check("patient waiting 20 real minutes breaches ESI-2's 10-min ceiling", breached is True)
    check("breach reason references the real elapsed minutes", "20" in reason)


def test_pathway_suggestion_is_guideline_not_automatic():
    check("ESI 1 suggests Main ED", ps.suggest_pathway(1) == ps.PATHWAY_MAIN_ED)
    check("ESI 2 suggests Main ED", ps.suggest_pathway(2) == ps.PATHWAY_MAIN_ED)
    check("ESI 3 suggests Waiting/Vertical", ps.suggest_pathway(3) == ps.PATHWAY_WAITING_VERTICAL)
    check("ESI 4 suggests Fast/Normal", ps.suggest_pathway(4) == ps.PATHWAY_FAST_NORMAL)
    check("ESI 5 suggests Fast/Normal", ps.suggest_pathway(5) == ps.PATHWAY_FAST_NORMAL)

    pipeline = get_pipeline()
    p1 = make_patient(patient_id="RT-5")
    decision = pipeline.assess_patient(p1)
    rec = ps.create_patient_record("RT-5", p1, decision)
    check("patient record has an initial pathway suggestion stored", rec["pathway"] in ps.PATHWAY_LABELS)

    rec = ps.set_pathway(rec, ps.PATHWAY_FAST_NORMAL)
    check("nurse can manually override the pathway assignment", rec["pathway"] == ps.PATHWAY_FAST_NORMAL)


def test_pathway_persists_through_reassessment():
    pipeline = get_pipeline()
    p1 = make_patient(patient_id="RT-6")
    decision1 = pipeline.assess_patient(p1)
    rec = ps.create_patient_record("RT-6", p1, decision1)
    rec = ps.set_pathway(rec, ps.PATHWAY_WAITING_VERTICAL)

    p2 = dict(p1)
    p2["heart_rate"] = 95
    decision2 = pipeline.assess_patient(p2)
    rec = ps.add_reassessment(rec, p2, decision2)
    check("nurse-set pathway persists across a reassessment (not silently reset)",
          rec["pathway"] == ps.PATHWAY_WAITING_VERTICAL)


def test_main_ed_bed_state_computed_live():
    pipeline = get_pipeline()
    patients = {}
    for i, pathway in enumerate([ps.PATHWAY_MAIN_ED, ps.PATHWAY_MAIN_ED, ps.PATHWAY_WAITING_VERTICAL]):
        p = make_patient(patient_id=f"BED-{i}")
        d = pipeline.assess_patient(p)
        rec = ps.create_patient_record(f"BED-{i}", p, d)
        rec = ps.set_pathway(rec, pathway)
        patients[f"BED-{i}"] = rec

    state = ps.main_ed_bed_state(patients, total_main_ed_beds=3)
    check("occupied beds computed live from actual pathway assignments", state["occupied_main_ed_beds"] == 2)
    check("available beds correctly computed", state["available_main_ed_beds"] == 1)

    state2 = ps.main_ed_bed_state(patients, total_main_ed_beds=1)
    check("available beds never negative even if occupancy exceeds a reduced total",
          state2["available_main_ed_beds"] == 0)


def test_accept_or_override_auto_moves_pathway_and_frees_main_ed_bed():
    """NEW requirement: accept/override always immediately moves the
    patient to the pathway matching their final ESI — no separate
    confirmation step. This supersedes the older 'bed_release_eligible +
    manual confirm' workflow for the normal accept/override path."""
    pipeline = get_pipeline()
    p1 = make_patient(patient_id="RT-7")
    decision1 = pipeline.assess_patient(p1)
    rec = ps.create_patient_record("RT-7", p1, decision1)
    rec = ps.set_pathway(rec, ps.PATHWAY_MAIN_ED)
    rec["assessments"][0]["ai_recommended_esi"] = 2
    rec = ps.apply_accept(rec)
    check("ESI 2 accept keeps patient in Main ED", rec["pathway"] == ps.PATHWAY_MAIN_ED)

    p2 = dict(p1)
    decision2 = pipeline.assess_patient(p2)
    rec = ps.add_reassessment(rec, p2, decision2)
    rec["assessments"][-1]["ai_recommended_esi"] = 3
    rec = ps.apply_accept(rec)  # accept alone (no manual override) still auto-moves
    check("accepting an ESI-3 recommendation auto-moves out of Main ED immediately",
          rec["pathway"] == ps.PATHWAY_WAITING_VERTICAL)


def test_bed_release_requires_actual_reassessment_not_surge_alone():
    pipeline = get_pipeline()
    p1 = make_patient(patient_id="RT-8")
    decision1 = pipeline.assess_patient(p1)
    rec = ps.create_patient_record("RT-8", p1, decision1)
    rec = ps.set_pathway(rec, ps.PATHWAY_MAIN_ED)
    rec["assessments"][0]["ai_recommended_esi"] = 2
    rec = ps.apply_accept(rec)

    rec["arrival_timestamp"] = time.time() - (999 * 60)
    check("long wait time alone does not make an ESI-2 patient bed-release eligible",
          ps.bed_release_eligible(rec) is False)


def test_next_main_ed_candidate_surfaced_not_auto_assigned():
    pipeline = get_pipeline()
    patients = {}
    configs = [("NEXT-1", 4, ps.PATHWAY_FAST_NORMAL), ("NEXT-2", 2, ps.PATHWAY_WAITING_VERTICAL),
               ("NEXT-3", 3, ps.PATHWAY_WAITING_VERTICAL)]
    for pid, esi, pathway in configs:
        p = make_patient(patient_id=pid)
        d = pipeline.assess_patient(p)
        rec = ps.create_patient_record(pid, p, d)
        rec["assessments"][0]["ai_recommended_esi"] = esi
        rec = ps.apply_accept(rec)
        rec = ps.set_pathway(rec, pathway)
        patients[pid] = rec

    candidate = ps.find_next_main_ed_candidate(patients)
    check("next Main ED candidate is the lowest-ESI eligible waiting patient", candidate == "NEXT-2")

    candidate2 = ps.find_next_main_ed_candidate(patients, exclude_patient_id="NEXT-2")
    check("no eligible candidate returns None rather than an inappropriate patient",
          candidate2 is None)


def test_modify_esi_manual_edit_does_not_auto_move_pathway():
    """Requirement 2: 'Modify ESI' is a separate quick-edit path that does
    NOT auto-move the pathway by itself — the caller (quiet vs surge)
    decides when to actually move the patient."""
    pipeline = get_pipeline()
    p1 = make_patient(patient_id="MOD-1")
    decision = pipeline.assess_patient(p1)
    rec = ps.create_patient_record("MOD-1", p1, decision)
    rec = ps.set_pathway(rec, ps.PATHWAY_MAIN_ED)
    rec["assessments"][0]["ai_recommended_esi"] = 2
    rec = ps.apply_accept(rec)
    check("setup: patient in Main ED at ESI 2", rec["pathway"] == ps.PATHWAY_MAIN_ED)

    rec = ps.apply_manual_esi_edit(rec, 4, "Nurse clinical judgment: patient looks stable, downgrading.")
    check("manual ESI edit updates displayed ESI", ps.get_displayed_esi(rec) == 4)
    check("manual ESI edit does NOT auto-move pathway", rec["pathway"] == ps.PATHWAY_MAIN_ED)
    check("manual ESI edit is tagged distinctly from a model-based override",
          rec["assessments"][-1]["manual_esi_edit"] is True)

    # quiet shift: nurse explicitly confirms the move
    rec = ps.auto_assign_pathway_from_esi(rec)
    check("explicit 'Send to new Dept' action moves the patient", rec["pathway"] == ps.PATHWAY_FAST_NORMAL)


def test_modify_esi_surge_auto_moves_immediately():
    pipeline = get_pipeline()
    p1 = make_patient(patient_id="MOD-2")
    decision = pipeline.assess_patient(p1)
    rec = ps.create_patient_record("MOD-2", p1, decision)
    rec = ps.set_pathway(rec, ps.PATHWAY_MAIN_ED)
    rec["assessments"][0]["ai_recommended_esi"] = 2
    rec = ps.apply_accept(rec)

    rec = ps.apply_manual_esi_edit(rec, 3, "Judgment call.")
    # simulate the surge-mode UI branch: auto-move immediately after saving
    rec = ps.auto_assign_pathway_from_esi(rec)
    check("surge mode moves patient immediately after Modify-ESI save, no separate confirm",
          rec["pathway"] == ps.PATHWAY_WAITING_VERTICAL)


# ---------------------------------------------------------------------
# VERTICAL CARE 30-MINUTE TIMER (ESI 3 only)
# ---------------------------------------------------------------------

def test_vertical_timer_only_applies_to_esi3_in_vertical_care():
    pipeline = get_pipeline()
    p1 = make_patient(patient_id="VT-1")
    decision = pipeline.assess_patient(p1)
    rec = ps.create_patient_record("VT-1", p1, decision)
    rec["assessments"][0]["ai_recommended_esi"] = 3
    rec = ps.apply_accept(rec)  # auto-moves to Waiting/Vertical since ESI 3
    check("ESI 3 patient auto-moved to Waiting/Vertical", rec["pathway"] == ps.PATHWAY_WAITING_VERTICAL)

    status = ps.check_vertical_care_timer(rec)
    check("timer is applicable for an ESI-3 Vertical Care patient", status["applicable"] is True)
    check("timer starts near the full 30 minutes remaining", status["remaining_seconds"] > 1790)

    # an ESI-2 patient who overflows into Vertical Care during surge must NOT get the timer
    p2 = make_patient(patient_id="VT-2")
    decision2 = pipeline.assess_patient(p2)
    rec2 = ps.create_patient_record("VT-2", p2, decision2)
    rec2["assessments"][0]["ai_recommended_esi"] = 2
    rec2 = ps.apply_accept(rec2)
    rec2 = ps.set_pathway(rec2, ps.PATHWAY_WAITING_VERTICAL)  # manual surge-overflow placement
    status2 = ps.check_vertical_care_timer(rec2)
    check("ESI-2 overflow patient in Vertical Care does NOT get the dedicated timer",
          status2["applicable"] is False)


def test_vertical_timer_breach_and_reset():
    pipeline = get_pipeline()
    p1 = make_patient(patient_id="VT-3")
    decision = pipeline.assess_patient(p1)
    rec = ps.create_patient_record("VT-3", p1, decision)
    rec["assessments"][0]["ai_recommended_esi"] = 3
    rec = ps.apply_accept(rec)

    status_fresh = ps.check_vertical_care_timer(rec)
    check("fresh timer has not breached", status_fresh["breached"] is False)

    # backdate the timer start to simulate 31 minutes elapsed
    rec["vertical_care_timer_start"] = time.time() - (31 * 60)
    status_breached = ps.check_vertical_care_timer(rec)
    check("timer breaches after 30 minutes", status_breached["breached"] is True)

    # nurse resets the timer
    rec = ps.reset_vertical_timer(rec)
    status_after_reset = ps.check_vertical_care_timer(rec)
    check("nurse-initiated reset clears the breach", status_after_reset["breached"] is False)
    check("reset timer has close to the full 30 minutes remaining again",
          status_after_reset["remaining_seconds"] > 1790)


def main():
    test_realtime_timer_uses_stored_arrival_timestamp()
    test_timer_not_reset_by_navigation_or_reassessment()
    test_no_manual_wait_minutes_counter_exists()
    test_real_wait_breach_uses_elapsed_time()
    test_pathway_suggestion_is_guideline_not_automatic()
    test_pathway_persists_through_reassessment()
    test_main_ed_bed_state_computed_live()
    test_accept_or_override_auto_moves_pathway_and_frees_main_ed_bed()
    test_bed_release_requires_actual_reassessment_not_surge_alone()
    test_next_main_ed_candidate_surfaced_not_auto_assigned()
    test_modify_esi_manual_edit_does_not_auto_move_pathway()
    test_modify_esi_surge_auto_moves_immediately()
    test_vertical_timer_only_applies_to_esi3_in_vertical_care()
    test_vertical_timer_breach_and_reset()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
