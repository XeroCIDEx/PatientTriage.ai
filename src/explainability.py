"""
explainability.py
==================
Turns raw model internals into nurse-readable explanations. Implements
spec Section 2 in full:

  - Model 1 (XGBoost): REAL SHAP (TreeExplainer) feature attribution for
    the predicted class, translated from raw feature names into
    human-readable sentences with the patient's actual values. Both
    numeric features (HR, SpO2, etc.) and word-TFIDF text features
    (readable terms like "chest pain") get attributions, since Model 1's
    word-level vectorizer produces genuinely readable vocabulary.
  - Model 2 (MLP + character n-gram TF-IDF): character n-grams are NOT
    human-readable (a SHAP attribution on "hest p" means nothing to a
    nurse), so true per-feature attribution is not shown for Model 2.
    Per the spec's own instruction ("if a particular model cannot
    support a reliable explanation, clearly state the limitation and
    show the strongest defensible explanation available"), Model 2's
    explanation instead lists which known clinically-relevant phrases
    actually appear in the patient's own narrative — this is REAL
    evidence the model's input text contained, honestly labeled as
    evidence-present rather than as a model attribution claim.
  - Agreement / disagreement narrative builders that never fabricate a
    causal reason for disagreement (Section 2B): when no reliable reason
    can be established, they say so explicitly, as instructed.
  - History evidence extraction from the retrieved history text block,
    with a combination-only claim (never "history alone caused this").

No explanation here is invented — every line traces back to an actual
model output, an actual patient value, or an actual retrieved record.
"""

from __future__ import annotations
import numpy as np
from typing import Optional

# ---------------------------------------------------------------------
# Human-readable labels for structured features
# ---------------------------------------------------------------------

_NUMERIC_LABELS = {
    "age": ("Age", "{v:.0f}", ""),
    "heart_rate": ("Heart rate", "HR {v:.0f}", " bpm"),
    "resp_rate": ("Respiratory rate", "RR {v:.0f}", "/min"),
    "sbp": ("Systolic blood pressure", "SBP {v:.0f}", " mmHg"),
    "dbp": ("Diastolic blood pressure", "DBP {v:.0f}", " mmHg"),
    "spo2": ("Oxygen saturation", "SpO2 {v:.0f}", "%"),
    "temperature": ("Temperature", "{v:.1f}", "\u00b0C"),
    "pain_score": ("Reported pain score", "{v:.0f}", "/10"),
    "shock_index": ("Shock index (HR/SBP ratio)", "{v:.2f}", ""),
    "pulse_pressure": ("Pulse pressure", "{v:.0f}", " mmHg"),
    "missing_field_count": ("Amount of missing intake data", "{v:.0f}", " fields missing"),
}

_BOOLEAN_LABELS = {
    "is_pediatric": "Patient is in the pediatric age band",
    "is_geriatric": "Patient is in the geriatric age band",
    "has_history_info": "Medical history was available at intake",
}

_MISSING_FIELD_READABLE = {
    "heart_rate_missing": "heart rate", "resp_rate_missing": "respiratory rate",
    "sbp_missing": "systolic BP", "dbp_missing": "diastolic BP",
    "spo2_missing": "SpO2", "temperature_missing": "temperature", "pain_score_missing": "pain score",
}


_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "but", "no", "not",
    "is", "was", "are", "were", "be", "been", "with", "without", "as", "by", "from", "it",
    "this", "that", "these", "those", "has", "have", "had", "reported", "none",
}


def _humanize_feature(name: str, raw_value: float, patient: dict) -> Optional[str]:
    """Returns a human-readable label+value string for one structured
    feature, or None if this feature isn't worth surfacing (e.g. a
    'not missing' flag)."""
    if name in _NUMERIC_LABELS:
        label, fmt, unit = _NUMERIC_LABELS[name]
        return f"{label} ({fmt.format(v=raw_value)}{unit})"
    if name in _MISSING_FIELD_READABLE:
        if raw_value >= 1:
            return f"Missing {_MISSING_FIELD_READABLE[name]} at intake"
        return None
    if name in _BOOLEAN_LABELS:
        if name == "has_history_info":
            return _BOOLEAN_LABELS[name] if raw_value >= 1 else "Medical history was NOT available at intake"
        return _BOOLEAN_LABELS[name] if raw_value >= 1 else None
    return None


def shap_explain_model1(model1_wrapper, x_numeric_row, x_text_row, patient: dict,
                         word_vocab: list, top_k: int = 5) -> dict:
    """Real SHAP TreeExplainer attribution for Model 1's predicted class.
    Returns {"available": bool, "factors": [{"text":.., "direction":.., "shap_value":..}], "limitation": str|None}
    """
    try:
        import shap
        import scipy.sparse as sp
    except ImportError:
        return {"available": False, "factors": [],
                "limitation": "SHAP library not available in this environment."}

    Xn_scaled = model1_wrapper.scaler.transform(x_numeric_row)
    X = sp.hstack([sp.csr_matrix(Xn_scaled), x_text_row]).tocsr()

    try:
        explainer = shap.TreeExplainer(model1_wrapper.clf)
        sv = explainer.shap_values(X)  # shape (1, n_features, n_classes)
    except Exception as e:
        return {"available": False, "factors": [], "limitation": f"SHAP computation failed: {e}"}

    probs = model1_wrapper.predict_proba(x_numeric_row, x_text_row)[0]
    predicted_class_idx = int(np.argmax(probs))
    esi = model1_wrapper.ESI_CLASSES[predicted_class_idx]

    row_shap = sv[0, :, predicted_class_idx]  # per-feature attribution toward the predicted class
    numeric_names = model1_wrapper.feature_names_numeric or []
    n_numeric = len(numeric_names)
    all_feature_names = list(numeric_names) + list(word_vocab)

    if len(all_feature_names) != len(row_shap):
        # dimension mismatch guard — don't silently misattribute
        return {"available": False, "factors": [],
                "limitation": "Feature/name alignment mismatch — attribution withheld to avoid mislabeling."}

    x_numeric_flat = np.asarray(x_numeric_row).flatten()
    x_text_flat = np.asarray(x_text_row.todense()).flatten()

    factors = []
    for i, shap_val in enumerate(row_shap):
        if abs(shap_val) < 1e-4:
            continue
        if i < n_numeric:
            raw_value = x_numeric_flat[i]
            label = _humanize_feature(numeric_names[i], raw_value, patient)
        else:
            text_idx = i - n_numeric
            term_present = x_text_flat[text_idx] > 0
            if not term_present:
                continue  # only surface text terms actually present in this patient's narrative
            term = all_feature_names[i]
            # skip uninformative stopword-only terms — this filters the
            # DISPLAY only; the SHAP value itself is unchanged/unfabricated,
            # we just don't surface tokens too generic to mean anything to
            # a nurse (e.g. "of", "no" scoring high due to sparse-vocab noise)
            if all(tok in _STOPWORDS for tok in term.split()):
                continue
            label = f'Clinical narrative mentions "{term}"'
        if label is None:
            continue
        direction = "increased" if shap_val > 0 else "decreased"
        text = (f"{label} {direction} the likelihood of this ESI {esi} assessment"
                + ("." if shap_val > 0 else " (outweighed by other factors)."))
        factors.append({"text": text, "direction": direction, "shap_value": float(shap_val), "abs": abs(float(shap_val))})

    factors.sort(key=lambda f: f["abs"], reverse=True)
    return {"available": True, "factors": factors[:top_k], "limitation": None}


# ---------------------------------------------------------------------
# Model 2 — evidence-presence extraction (NOT a feature attribution
# claim; char n-gram + MLP has no human-readable per-feature attribution)
# ---------------------------------------------------------------------

_CLINICAL_EVIDENCE_TERMS = [
    "chest pain", "shortness of breath", "difficulty breathing", "facial droop",
    "slurred speech", "severe bleeding", "uncontrolled bleeding", "unresponsive",
    "seizure", "head trauma", "allergic reaction", "abdominal pain", "vomiting",
    "diarrhea", "fever", "confusion", "dizziness", "wheezing", "rash",
    "sore throat", "ear pain", "back pain", "headache", "swelling", "cough",
    "fatigue", "syncope", "palpitations", "radiating to left arm", "diaphoresis",
]


def clinical_evidence_model2(patient: dict, top_k: int = 6) -> dict:
    """NOT SHAP — a defensible, non-fabricated alternative for an
    architecture (char-ngram TF-IDF + MLP) where per-feature attribution
    would not be human-readable. Lists curated clinical phrases that
    genuinely appear in the patient's own narrative, which is real
    evidence the model actually received as input — just not a claim
    about which specific characters the network weighted most."""
    text = " ".join([
        str(patient.get("chief_complaint", "") or ""),
        str(patient.get("symptoms", "") or ""),
    ]).lower()
    present = [term for term in _CLINICAL_EVIDENCE_TERMS if term in text]
    return {
        "available": True,
        "evidence": present[:top_k],
        "limitation": ("Model 2 uses character-level text features and a neural network, which do not "
                       "produce human-readable per-feature attributions the way Model 1's tree-based "
                       "model does. The items below are clinically relevant phrases confirmed present "
                       "in the patient's own narrative (i.e., evidence the model actually saw), not a "
                       "ranked attribution of which words most influenced its specific prediction."),
    }


# ---------------------------------------------------------------------
# Agreement / disagreement narrative builders
# ---------------------------------------------------------------------

def build_agreement_explanation(esi: int, model1_factors: list, model2_evidence: list) -> list:
    lines = []
    for f in model1_factors[:3]:
        lines.append(f"XGBoost: {f['text']}")
    for e in model2_evidence[:3]:
        lines.append(f'Clinical NLP model: narrative includes "{e}".')
    lines.append(f"Both models independently classified this patient as ESI {esi}.")
    return lines


def build_disagreement_explanation(model1_result: dict, model2_result: dict,
                                    model1_factors: list, model2_evidence: list) -> dict:
    """Per spec Section 2B: explain both sides separately, and NEVER
    invent a specific causal reason for the disagreement unless it can
    actually be established from the model outputs. Here it cannot (the
    two models use different architectures and representations), so we
    say that plainly rather than fabricate a mechanism."""
    return {
        "model1_side": {
            "esi": model1_result["esi_prediction"],
            "confidence": model1_result["confidence"],
            "factors": [f["text"] for f in model1_factors[:4]],
        },
        "model2_side": {
            "esi": model2_result["esi_prediction"],
            "confidence": model2_result["confidence"],
            "evidence": model2_evidence[:4],
        },
        "reason": ("The models produced different assessments from the same available evidence. "
                   "No single causal reason can be established from the model outputs alone — "
                   "Model 1 and Model 2 use different algorithms and different representations of "
                   "the same patient information, so a direct 'why' cannot be attributed with "
                   "confidence. Treat this as a genuine signal for nurse review, not as an error."),
    }


def extract_history_evidence(history_text: Optional[str]) -> list:
    """Pulls the [ACTIVE HISTORY] lines out of the clinical text block
    built by history_context.build_clinical_text_block(), if present.
    Returns [] if there's no meaningful history (NO_RECORD/UNAVAILABLE
    placeholder text, or no active-history section) — we do not pretend
    history contributed when there wasn't any to contribute."""
    if not history_text:
        return []
    if "[ACTIVE HISTORY]" not in history_text:
        return []
    lines = history_text.split("\n")
    try:
        start = lines.index("[ACTIVE HISTORY]") + 1
    except ValueError:
        return []
    evidence = []
    for line in lines[start:]:
        if line.startswith("[") or not line.strip():
            break
        evidence.append(line.strip())
    return evidence
