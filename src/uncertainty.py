"""
uncertainty.py
==============
Computes an uncertainty assessment that is NOT just "1 - max probability".

Combines:
  1. Margin  = P(top class) - P(second class)   (small margin = ambiguous)
  2. Entropy = normalized Shannon entropy of the probability vector
  3. Missing-data score = fraction of expected fields that were missing
  4. Model disagreement = whether Model 1 and Model 2 predict different ESI

Thresholds are simple, documented prototype constants — not derived from
any clinical calibration study.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

MARGIN_LOW_THRESHOLD = 0.15       # margin below this -> ambiguous top-2 classes
ENTROPY_HIGH_THRESHOLD = 0.75     # normalized entropy above this -> high uncertainty
MISSING_HIGH_THRESHOLD = 0.30     # >30% of key fields missing -> high uncertainty


@dataclass
class UncertaintyAssessment:
    margin: float
    entropy_norm: float
    missing_score: float
    model_disagreement: bool
    level: str          # "LOW" | "MODERATE" | "HIGH"
    reasons: list


def normalized_entropy(probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-12, 1.0)
    ent = -np.sum(probs * np.log(probs))
    max_ent = np.log(len(probs))
    return float(ent / max_ent) if max_ent > 0 else 0.0


def top2_margin(probs: np.ndarray) -> float:
    sorted_probs = np.sort(probs)[::-1]
    return float(sorted_probs[0] - sorted_probs[1])


def assess_uncertainty(probs_model1: np.ndarray, missing_score: float, model_disagreement: bool) -> UncertaintyAssessment:
    margin = top2_margin(probs_model1)
    entropy_norm = normalized_entropy(probs_model1)

    reasons = []
    score = 0
    if margin < MARGIN_LOW_THRESHOLD:
        reasons.append(f"Top-two predicted classes are close (margin={margin:.2f} < {MARGIN_LOW_THRESHOLD}).")
        score += 1
    if entropy_norm > ENTROPY_HIGH_THRESHOLD:
        reasons.append(f"Probability distribution is spread across classes (normalized entropy={entropy_norm:.2f}).")
        score += 1
    if missing_score > MISSING_HIGH_THRESHOLD:
        reasons.append(f"A significant fraction of intake fields were missing ({missing_score:.0%}).")
        score += 1
    if model_disagreement:
        reasons.append("Model 1 and Model 2 disagree on the predicted ESI level.")
        score += 2  # weighted higher — disagreement is a strong signal

    if score >= 3:
        level = "HIGH"
    elif score >= 1:
        level = "MODERATE"
    else:
        level = "LOW"

    if not reasons:
        reasons.append("No ambiguity signals detected; prediction is well-separated and inputs are complete.")

    return UncertaintyAssessment(margin, entropy_norm, missing_score, model_disagreement, level, reasons)
