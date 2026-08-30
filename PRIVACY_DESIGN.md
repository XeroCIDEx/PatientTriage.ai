# Patient Data Protection — Design Notes

> This document describes how patient data WOULD be protected in a real
> deployment, and states plainly what is (and isn't) actually implemented
> in this hackathon prototype. It is a design specification, not a
> completed security implementation — a functioning access-control and
> encryption layer is out of scope for a prototype running on synthetic
> data with no real users, but the design is spelled out here so the
> gap is explicit rather than silent.

## 1. Assumed regulatory jurisdiction

This prototype assumes a **United States deployment governed by HIPAA**
(the Health Insurance Portability and Accountability Act), as the
Round 2 brief explicitly asks teams to state a jurisdiction. This is a
stated assumption, not a claim of actual compliance — nothing in this
codebase has been reviewed by a compliance officer or legal counsel.
The design choices below are written to be *directionally consistent*
with HIPAA's Privacy Rule and Security Rule expectations; a real
deployment in the EU would instead need to map the same design onto
GDPR + relevant national health-data law (e.g. explicit consent basis,
right-to-erasure handling, EU data residency), which is a different
enough legal framework that it isn't simply a drop-in swap.

## 2. What data this system touches, and its sensitivity

| Data | Sensitivity | Where it lives in this prototype |
|---|---|---|
| Vitals, symptoms, chief complaint | Protected Health Information (PHI) under HIPAA | In-memory during a session (`st.session_state`); written to `logs/audit_log.jsonl` |
| Medical history text | PHI | Same as above |
| Patient ID | Direct identifier if it maps to a real person (e.g. MRN) | Used as a key throughout; in this prototype it's an arbitrary demo string, never a real medical record number |
| Clinician override reasoning | PHI (clinical judgment about a specific patient) | Logged in `audit_log.jsonl` |
| Synthetic training data | **Not PHI** — entirely fabricated, no real patients | `data/synthetic/synthetic_patients.csv` |

## 3. What's actually implemented right now (prototype scope)

- The audit log stores only a `patient_id` string, not a real name, DOB,
  SSN, or MRN — the prototype never asks for or handles real
  direct identifiers.
- No PHI ever leaves the local process — there's no external API call,
  telemetry, or third-party service that patient data is sent to.
- The audit log is a local, append-only JSONL file — nothing is
  transmitted over a network.

## 4. What a real deployment would add (not built here — explicitly out of scope for this prototype)

- **Access control**: role-based access so only authorized clinical
  staff can view PHI, with every access itself logged (HIPAA's
  "minimum necessary" principle — a floor nurse doesn't need the same
  visibility as, say, a hospital administrator).
- **Encryption at rest and in transit**: the audit log and any patient
  data store would need field-level or full-disk encryption; any
  network transmission would need TLS.
- **De-identification for model training**: this prototype trains only
  on synthetic data, so there's no real de-identification problem yet
  — but a real deployment retraining on actual ED data would need a
  formal de-identification or safe-harbor process before any of that
  data could be used to improve the models.
- **Retention policy**: a defined retention period for audit logs
  (HIPAA generally expects 6 years for certain records) with automatic
  purge/archival — this prototype's log simply grows indefinitely
  until manually cleared from the Audit Log page.
- **Business Associate Agreements (BAAs)**: required with any
  third-party vendor (cloud hosting, any external AI API) that would
  touch real PHI — not applicable here since everything runs locally
  on synthetic data, but would be a hard requirement before this
  system could touch a real hospital's data.
- **Consent model**: documentation of what a patient (or their
  guardian, for pediatric cases) is told about an AI-assisted triage
  tool being used on their intake, and any opt-out mechanism.
- **Breach notification plan**: a defined process for detecting and
  reporting unauthorized access, as HIPAA's Breach Notification Rule
  requires.

## 5. What the audit trail is designed to support either way

Regardless of jurisdiction, the audit log (`src/audit.py`) already
captures the fields a compliance review would look for in an override
event: timestamp, patient ID, both models' outputs and confidence,
the clinician's final decision, their typed reason, model version,
input completeness, and any safety flags. That structure is
jurisdiction-agnostic — what changes between HIPAA and GDPR is mostly
*retention, access control, and consent*, not *what gets logged*.
