"""
patient_state.py
=================
The underlying patient/encounter state model (spec Section 4, extended
by the Waiting Room / Care Pathways / Surge Bed-Release spec) — kept
separate from app.py so the state transitions are testable without
Streamlit and so the UI can't accidentally desync from the data model.

A "patient record" (what app.py stores per patient_id in
st.session_state.patients) is a dict with:

    patient_id, encounter_id
    assessments: [Assessment, ...]     # full longitudinal history, oldest first
    arrival_timestamp: float           # time.time() epoch — REAL wall-clock,
                                        # authoritative source for waiting time
    pathway: str                       # "MAIN_ED" | "WAITING_VERTICAL" | "FAST_NORMAL"
    status: str

An Assessment is a dict with exactly the fields spec Section 4 asks for:
    assessment_id, patient_id, encounter_id, timestamp
    input_snapshot                         # the raw patient dict submitted
    model1_prediction, model1_probabilities
    model2_prediction, model2_probabilities
    agreement_status, uncertainty
    safety_flags
    ai_recommended_esi
    clinician_final_esi                    # None until accept/override
    override, override_reason
    reassessment, previous_assessment_id
    decision                               # the full TriageDecision object (for rendering)

CENTRAL RULE (spec Section 3A / Part 22): the ESI shown anywhere
operational (Waiting Room, Dashboard, pathway logic) is ALWAYS:

    displayed_esi = clinician_final_esi if set, else ai_recommended_esi

Never the raw AI recommendation once a clinician has made a decision.

REAL-TIME WAITING TIME (Part 1): waiting time is NEVER a counter that
gets manually incremented. It is always computed on demand as
`time.time() - arrival_timestamp`. This means page navigation, a
browser refresh, or restarting the UI (as long as session_state
survives) can never desync the displayed time from reality — there is
no stored "minutes waited" value to go stale.

CARE PATHWAYS (Parts 3-5): `suggest_pathway(esi)` returns a GUIDELINE,
never an automatic assignment. The nurse always has the final say via
the UI, and `pathway` is stored explicitly per patient rather than
derived implicitly from ESI every time — so a nurse's manual pathway
choice persists even if a later reassessment would suggest something
else, until the nurse acts on that new suggestion.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
import time
import uuid

PATHWAY_MAIN_ED = "MAIN_ED"
PATHWAY_WAITING_VERTICAL = "WAITING_VERTICAL"
PATHWAY_FAST_NORMAL = "FAST_NORMAL"

PATHWAY_LABELS = {
    PATHWAY_MAIN_ED: "Main ED",
    PATHWAY_WAITING_VERTICAL: "Waiting / Vertical Care",
    PATHWAY_FAST_NORMAL: "Fast / Normal Treatment",
}


def suggest_pathway(esi: int) -> str:
    """GUIDELINE ONLY (spec Part 3/16) — not an automatic clinical
    assignment. ESI 1-2 -> Main ED, ESI 3 -> Waiting/Vertical,
    ESI 4-5 -> Fast/Normal. The nurse can always choose differently.
    This is the pure ESI-only guideline with NO bed-capacity awareness —
    used for messaging (e.g. the "Send to Dept" suggestion banner) and
    kept exactly as-is for backward compatibility. Actual placement into
    Main ED additionally requires a free bed — see resolve_pathway()."""
    if esi in (1, 2):
        return PATHWAY_MAIN_ED
    if esi == 3:
        return PATHWAY_WAITING_VERTICAL
    return PATHWAY_FAST_NORMAL


def resolve_pathway(esi: int, all_patient_records: Optional[dict] = None,
                     total_main_ed_beds: Optional[int] = None,
                     exclude_patient_id: Optional[str] = None) -> str:
    """Bed-capacity-aware pathway resolution.

    ESI 1/2 remain the ONLY ESI levels ever eligible for Main ED (ESI 3
    always goes to Waiting/Vertical Care, ESI 4-5 to Fast/Normal — that
    part of suggest_pathway() is unchanged). What's new: an ESI-1/2
    patient is only actually PLACED in Main ED if a bed is free right
    now. If Main ED is already at (or over) its operational capacity —
    i.e. occupied beds == total_main_ed_beds — the patient is instead
    routed to Waiting/Vertical Care until a bed frees up, rather than
    over-filling Main ED. As soon as a bed opens up, the next ESI-1/2
    patient resolved (on their own reassess/accept/override/modify-ESI
    action) will be placed into Main ED again.

    `exclude_patient_id` should be the patient's own ID so that a
    patient who is already occupying a Main ED bed doesn't get counted
    against their own availability check.

    Falls back to the plain ESI-only suggest_pathway() guideline when no
    bed context is supplied, so existing callers/tests that don't pass
    bed info behave exactly as before."""
    guideline = suggest_pathway(esi)
    if guideline != PATHWAY_MAIN_ED or all_patient_records is None or total_main_ed_beds is None:
        return guideline
    occupied = sum(
        1 for pid, r in all_patient_records.items()
        if r.get("pathway") == PATHWAY_MAIN_ED and pid != exclude_patient_id
    )
    if occupied < total_main_ed_beds:
        return PATHWAY_MAIN_ED
    return PATHWAY_WAITING_VERTICAL  # Main ED full — hold in Waiting/Vertical Care instead


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def new_assessment(patient_id: str, encounter_id: str, input_snapshot: dict, decision,
                    reassessment: bool = False, previous_assessment_id: Optional[str] = None) -> dict:
    """Builds a fresh Assessment dict from a TriageDecision. Does NOT set
    clinician_final_esi — that's only set later via apply_accept/apply_override,
    matching the spec's explicit "nurse decision" step in the workflow."""
    return {
        "assessment_id": _new_id("ASMT"),
        "patient_id": patient_id,
        "encounter_id": encounter_id,
        "timestamp": _now_iso(),
        "input_snapshot": dict(input_snapshot),
        "model1_prediction": decision.model1_result["esi_prediction"],
        "model1_probabilities": decision.model1_result["probabilities"],
        "model2_prediction": decision.model2_result["esi_prediction"],
        "model2_probabilities": decision.model2_result["probabilities"],
        "agreement_status": decision.verification_status,
        "agreement_tier": decision.agreement_tier,
        "uncertainty": decision.uncertainty.level,
        "safety_flags": [f.rule_id for f in decision.safety_flags],
        "ai_recommended_esi": decision.final_esi,
        "clinician_final_esi": None,
        "override": False,
        "override_reason": None,
        "reassessment": reassessment,
        "previous_assessment_id": previous_assessment_id,
        "decision": decision,   # kept for UI rendering; not written to the audit log verbatim
    }


def create_patient_record(patient_id: str, input_snapshot: dict, decision,
                           all_patient_records: Optional[dict] = None,
                           total_main_ed_beds: Optional[int] = None) -> dict:
    """First-ever assessment for a patient in this session — creates the
    patient record AND its first assessment. arrival_timestamp is the
    REAL wall-clock time (time.time()) — the sole authoritative source
    for how long this patient has been waiting.

    `all_patient_records`/`total_main_ed_beds` are optional bed-capacity
    context (see resolve_pathway()) — pass the caller's current
    session_state.patients / session_state.total_main_ed_beds so a
    fresh ESI-1/2 patient isn't placed straight into Main ED if it's
    already full."""
    encounter_id = _new_id("ENC")
    assessment = new_assessment(patient_id, encounter_id, input_snapshot, decision, reassessment=False)
    rec = {
        "patient_id": patient_id,
        "encounter_id": encounter_id,
        "assessments": [assessment],
        "arrival_timestamp": time.time(),
        "pathway": None,
        "status": "WAITING",
    }
    # initial guideline suggestion, bed-capacity aware; nurse can change
    set_pathway(rec, resolve_pathway(decision.final_esi, all_patient_records, total_main_ed_beds,
                                      exclude_patient_id=patient_id))
    return rec


def add_reassessment(rec: dict, input_snapshot: dict, decision) -> dict:
    """Appends a new assessment to an EXISTING patient record. Does NOT
    create a new patient (spec Section 7 / Part 21) — same patient_id,
    same encounter_id, linked via previous_assessment_id. The old
    assessment is never removed or overwritten. arrival_timestamp is
    NEVER reset by a reassessment — spec Part 13: 'do not reset the
    timer just because their pathway changes' (and the same applies to
    a reassessment; the patient's total time in the department is a
    real-world fact, not something a reassessment should erase)."""
    previous = rec["assessments"][-1]
    assessment = new_assessment(rec["patient_id"], rec["encounter_id"], input_snapshot, decision,
                                 reassessment=True, previous_assessment_id=previous["assessment_id"])
    rec["assessments"].append(assessment)
    rec["status"] = "WAITING"
    return rec


def elapsed_seconds(rec: dict) -> float:
    """REAL elapsed waiting time — always computed as current time minus
    the stored arrival timestamp (spec Part 1). Never a counter."""
    return max(0.0, time.time() - rec["arrival_timestamp"])


def format_elapsed(rec: dict) -> str:
    """HH:MM:SS display format, as shown in the spec's own examples."""
    total_seconds = int(elapsed_seconds(rec))
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def check_vitals_deterioration(rec: dict) -> tuple:
    """Compares the two most recent assessments' vitals (spec Part 15) —
    computed live from the assessment history, never a stored flag that
    could go stale. Returns (deteriorated: bool, reasons: list). Needs
    at least 2 assessments to have anything to compare."""
    from .monitoring import detect_deterioration
    if len(rec["assessments"]) < 2:
        return False, []
    previous = rec["assessments"][-2]["input_snapshot"]
    current = rec["assessments"][-1]["input_snapshot"]
    return detect_deterioration(previous, current)


def check_real_wait_breach(esi: int, rec: dict, max_safe_wait_minutes: dict) -> tuple:
    """Real-time equivalent of the old manual wait-time check — uses
    elapsed_seconds() instead of a manually-tracked counter."""
    elapsed_min = elapsed_seconds(rec) / 60.0
    limit = max_safe_wait_minutes.get(esi, 120)
    if elapsed_min >= limit:
        return True, f"Patient has waited {elapsed_min:.0f} min, exceeding the {limit}-min safe wait ceiling for ESI {esi}."
    return False, None
    """Real-time equivalent of the old manual wait-time check — uses
    elapsed_seconds() instead of a manually-tracked counter."""
    elapsed_min = elapsed_seconds(rec) / 60.0
    limit = max_safe_wait_minutes.get(esi, 120)
    if elapsed_min >= limit:
        return True, f"Patient has waited {elapsed_min:.0f} min, exceeding the {limit}-min safe wait ceiling for ESI {esi}."
    return False, None


def latest_assessment(rec: dict) -> dict:
    return rec["assessments"][-1]


def apply_accept(rec: dict, all_patient_records: Optional[dict] = None,
                  total_main_ed_beds: Optional[int] = None) -> dict:
    """Nurse accepts the AI recommendation as-is (spec: clinician_final_esi
    = ai_recommended_esi, override = False). The patient is automatically
    moved to the pathway matching this final ESI (bed-capacity aware,
    see resolve_pathway()) — a clinician decision (accept counts as one)
    always determines dept placement."""
    a = latest_assessment(rec)
    a["clinician_final_esi"] = a["ai_recommended_esi"]
    a["override"] = False
    a["override_reason"] = "Nurse accepted AI recommendation."
    auto_assign_pathway_from_esi(rec, all_patient_records, total_main_ed_beds)
    return rec


def apply_override(rec: dict, new_esi: int, reason: str, all_patient_records: Optional[dict] = None,
                    total_main_ed_beds: Optional[int] = None) -> dict:
    """Nurse overrides to a different ESI via the full model-based
    assessment/reassessment flow: stores BOTH ai_recommended_esi
    (untouched) and clinician_final_esi (the new value) — never
    overwrites the original AI recommendation. Per explicit
    requirement: an override is ALWAYS treated as the final ESI, and
    the patient is ALWAYS immediately moved to the matching pathway
    (Main ED / Waiting-Vertical / Fast-Normal) — unconditionally, in
    both quiet and surge conditions, and bed-capacity aware (see
    resolve_pathway()). (Contrast with apply_manual_esi_edit(), the
    separate "Modify ESI" quick-edit path, where dept movement is
    conditional on quiet/surge — see that function's docstring.)"""
    a = latest_assessment(rec)
    a["clinician_final_esi"] = new_esi
    a["override"] = (new_esi != a["ai_recommended_esi"])
    a["override_reason"] = reason
    auto_assign_pathway_from_esi(rec, all_patient_records, total_main_ed_beds)
    return rec


def apply_manual_esi_edit(rec: dict, new_esi: int, reason: str) -> dict:
    """The separate "Modify ESI" quick-edit path: a nurse can set the
    final ESI directly from her own clinical judgment, without running
    the full model-based (re)assessment flow. Updates clinician_final_esi
    exactly like apply_override, but is tagged manual_esi_edit=True so
    the audit trail and UI can distinguish "nurse judgment call" from
    "nurse reviewed a fresh model assessment."

    UNLIKE apply_override(), this does NOT automatically move the
    patient's pathway — the caller (app.py) decides:
      - During SURGE: the caller should immediately call
        auto_assign_pathway_from_esi() right after this, no confirmation.
      - During a quiet shift: the caller should leave the pathway as-is
        and let the nurse explicitly confirm via a "Send to new Dept"
        action, which then calls auto_assign_pathway_from_esi()."""
    a = latest_assessment(rec)
    a["clinician_final_esi"] = new_esi
    a["override"] = (new_esi != a["ai_recommended_esi"])
    a["override_reason"] = reason
    a["manual_esi_edit"] = True
    return rec


def auto_assign_pathway_from_esi(rec: dict, all_patient_records: Optional[dict] = None,
                                  total_main_ed_beds: Optional[int] = None) -> dict:
    """Computes the pathway guideline for the patient's CURRENT displayed
    ESI and applies it immediately via set_pathway() (which also handles
    the Vertical Care timer reset-on-entry). This is the single place
    where "final ESI determines dept" is enforced, so
    apply_accept/apply_override, the surge branch of the Modify-ESI
    flow, and the quiet-shift "Send to Dept" confirmation all share
    identical, bed-capacity-aware behavior (see resolve_pathway()):
    ESI 1/2 only actually go to Main ED if a bed is free there right
    now; otherwise they're held in Waiting/Vertical Care."""
    new_pathway = resolve_pathway(get_displayed_esi(rec), all_patient_records, total_main_ed_beds,
                                   exclude_patient_id=rec["patient_id"])
    return set_pathway(rec, new_pathway)


def get_displayed_esi(rec: dict) -> int:
    """Spec Section 3A — the single rule for what the Waiting Room and
    Dashboard must show: clinician_final_esi if a decision has been
    made, otherwise the AI's recommendation as a provisional value."""
    a = latest_assessment(rec)
    return a["clinician_final_esi"] if a["clinician_final_esi"] is not None else a["ai_recommended_esi"]


def is_overridden(rec: dict) -> bool:
    a = latest_assessment(rec)
    return bool(a["override"])


def assessment_timeline(rec: dict) -> list:
    """Returns a compact, display-ready summary of every assessment for
    this patient, oldest first — for an "assessment history" UI panel."""
    out = []
    for a in rec["assessments"]:
        out.append({
            "timestamp": a["timestamp"],
            "assessment_id": a["assessment_id"],
            "reassessment": a["reassessment"],
            "ai_recommended_esi": a["ai_recommended_esi"],
            "clinician_final_esi": a["clinician_final_esi"],
            "override": a["override"],
            "override_reason": a["override_reason"],
            "agreement_status": a["agreement_status"],
            "uncertainty": a["uncertainty"],
        })
    return out


# ---------------------------------------------------------------------
# Care pathways & Main ED bed capacity (spec Parts 3-9, 12, 16-18, 23)
# ---------------------------------------------------------------------

VERTICAL_CARE_TIMER_SECONDS = 30 * 60  # 30-minute dedicated reassessment timer for ESI-3 patients in Vertical Care


def set_pathway(rec: dict, pathway: str):
    """Nurse-confirmed (or auto-assigned-from-final-ESI) pathway
    assignment — always explicit, never silently derived on every
    render. suggest_pathway() is only ever a suggestion; this function
    is what actually applies it.

    Freshly ENTERING Waiting/Vertical Care starts (or restarts) the
    dedicated 30-minute Vertical Care timer — but only on a genuine
    transition into that pathway, not on every redundant call, so a
    patient who was already in Vertical Care doesn't get their timer
    silently reset by an unrelated state update."""
    entering_vertical = (pathway == PATHWAY_WAITING_VERTICAL and rec.get("pathway") != PATHWAY_WAITING_VERTICAL)
    rec["pathway"] = pathway
    if entering_vertical:
        rec["vertical_care_timer_start"] = time.time()
    return rec


def reset_vertical_timer(rec: dict) -> dict:
    """Nurse-initiated reset of the dedicated Vertical Care 30-minute
    timer — independent of the overall ED wait-time tracking. Used when
    the nurse has checked on the patient and judges they're fine to
    keep waiting a while longer."""
    rec["vertical_care_timer_start"] = time.time()
    return rec


def check_vertical_care_timer(rec: dict) -> dict:
    """Only applicable for patients CURRENTLY in Waiting/Vertical Care
    WITH a final ESI of exactly 3 (spec requirement: not for ESI 1-2
    overflow patients who may land in Vertical Care during surge).
    Returns {'applicable': bool, 'elapsed_seconds', 'remaining_seconds', 'breached': bool}."""
    if rec.get("pathway") != PATHWAY_WAITING_VERTICAL or get_displayed_esi(rec) != 3:
        return {"applicable": False, "elapsed_seconds": 0, "remaining_seconds": VERTICAL_CARE_TIMER_SECONDS, "breached": False}
    start = rec.get("vertical_care_timer_start", time.time())
    elapsed = max(0.0, time.time() - start)
    remaining = max(0.0, VERTICAL_CARE_TIMER_SECONDS - elapsed)
    return {"applicable": True, "elapsed_seconds": elapsed, "remaining_seconds": remaining,
            "breached": elapsed >= VERTICAL_CARE_TIMER_SECONDS}


def main_ed_bed_state(all_patient_records: dict, total_main_ed_beds: int) -> dict:
    """Computed live from actual current pathway assignments rather than
    a manually incremented/decremented counter (spec Part 23) — this
    avoids the whole class of bugs where a counter drifts out of sync
    with reality (e.g. a patient record is removed without the counter
    being updated). occupied is simply "how many waiting patients are
    currently assigned to the Main ED pathway right now"."""
    occupied = sum(1 for rec in all_patient_records.values() if rec.get("pathway") == PATHWAY_MAIN_ED)
    occupied = min(occupied, total_main_ed_beds)  # display never goes negative/over capacity
    return {
        "total_main_ed_beds": total_main_ed_beds,
        "occupied_main_ed_beds": occupied,
        "available_main_ed_beds": max(0, total_main_ed_beds - occupied),
    }


def bed_release_eligible(rec: dict) -> bool:
    """Spec Part 8: a Main ED patient becomes eligible for a pathway
    change (and thus bed release) ONLY when their CURRENT final
    clinician ESI is >= 3, and ONLY after an actual reassessment/nurse
    decision — never automatically from surge pressure alone."""
    if rec.get("pathway") != PATHWAY_MAIN_ED:
        return False
    return get_displayed_esi(rec) >= 3


def find_next_main_ed_candidate(all_patient_records: dict, exclude_patient_id: Optional[str] = None) -> Optional[str]:
    """Spec Part 9: after a Main ED bed frees up, surface (NOT
    automatically assign) the highest-priority eligible waiting patient
    currently outside the Main ED pathway — lowest ESI number first,
    longest-waiting as a tiebreaker. Returns a patient_id or None."""
    candidates = [
        (pid, rec) for pid, rec in all_patient_records.items()
        if rec.get("pathway") != PATHWAY_MAIN_ED and pid != exclude_patient_id
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda pr: (get_displayed_esi(pr[1]), -elapsed_seconds(pr[1])))
    best_esi = get_displayed_esi(candidates[0][1])
    if best_esi > 2:
        return None  # nothing clinically warrants Main ED right now
    return candidates[0][0]
