"""
evaluate.py
===========
Full evaluation of Model 1 and Model 2 on the held-out synthetic test set,
with explicit focus on UNDER-TRIAGE (predicting a LESS urgent ESI than the
true label) because under-triage is the more dangerous failure mode in a
real ED (spec section 19).

Run: python3 evaluate.py   (after train.py)
Writes: models/evaluation_report.json and prints a human-readable summary.
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

BASE = os.path.dirname(os.path.abspath(__file__))


def under_triage_rate(y_true, y_pred):
    """Under-triage = predicted ESI number is HIGHER than true ESI number
    (i.e. predicted LESS urgent than reality). ESI 1 = most urgent."""
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    under = (y_pred > y_true)
    return {
        "under_triage_rate_overall": float(under.mean()),
        "under_triage_count": int(under.sum()),
        "over_triage_rate_overall": float((y_pred < y_true).mean()),
        "exact_match_rate": float((y_pred == y_true).mean()),
    }


def critical_class_recall(y_true, y_pred):
    """Recall for ESI 1-2 (CRITICAL group) treated as a binary detection
    problem: did we correctly flag a truly-critical patient as critical?"""
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    true_critical = np.isin(y_true, [1, 2])
    pred_critical = np.isin(y_pred, [1, 2])
    if true_critical.sum() == 0:
        return {"critical_recall": None, "critical_missed_count": 0, "n_true_critical": 0}
    recall = (true_critical & pred_critical).sum() / true_critical.sum()
    missed = int((true_critical & ~pred_critical).sum())
    return {"critical_recall": float(recall), "critical_missed_count": missed, "n_true_critical": int(true_critical.sum())}


def evaluate_model(name, y_true, y_pred, probs=None):
    report = {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_recall": {str(k): v for k, v in zip(
            [1, 2, 3, 4, 5], recall_score(y_true, y_pred, average=None, labels=[1, 2, 3, 4, 5], zero_division=0))},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5]).tolist(),
        "confusion_matrix_labels": ["ESI1", "ESI2", "ESI3", "ESI4", "ESI5"],
        **under_triage_rate(y_true, y_pred),
        **{f"critical_{k}": v for k, v in critical_class_recall(y_true, y_pred).items()},
    }
    if probs is not None:
        conf = probs.max(axis=1)
        correct = (np.array(y_true) == np.array(y_pred))
        report["mean_confidence_correct"] = float(conf[correct].mean()) if correct.sum() else None
        report["mean_confidence_incorrect"] = float(conf[~correct].mean()) if (~correct).sum() else None
    return report


def main():
    bundle = joblib.load(os.path.join(BASE, "models", "test_bundle.joblib"))
    m1 = joblib.load(os.path.join(BASE, "models", "model1", "model1.joblib"))
    m2 = joblib.load(os.path.join(BASE, "models", "model2", "model2.joblib"))

    y_test = bundle["y_test"]
    probs1 = m1.predict_proba(bundle["Xn_test"], bundle["Xt1_test"])
    probs2 = m2.predict_proba(bundle["Xn_test"], bundle["Xt2_test"])
    pred1 = probs1.argmax(axis=1) + 1
    pred2 = probs2.argmax(axis=1) + 1

    report1 = evaluate_model("Model1_XGBoost", y_test, pred1, probs1)
    report2 = evaluate_model("Model2_MLP", y_test, pred2, probs2)

    disagreement_rate = float((pred1 != pred2).mean())
    # "insufficient data" proxy: count of missing fields in the raw df slice
    df_test = bundle["df_test_raw"]
    missing_cols = [c for c in df_test.columns if c.endswith("_missing")]
    insufficient_rate = float((df_test[missing_cols].sum(axis=1) >= 3).mean())

    combined = {
        "note": ("All results below are computed on a SYNTHETIC, rule-labeled test set "
                 "(see src/data_generation.py). These numbers describe how well each model "
                 "recovers the synthetic labeling rule, NOT real-world clinical accuracy. "
                 "No claim of clinical validity is made."),
        "model1": report1,
        "model2": report2,
        "model_disagreement_rate": disagreement_rate,
        "insufficient_data_rate_proxy": insufficient_rate,
        "n_test_patients": int(len(y_test)),
    }

    out_path = os.path.join(BASE, "models", "evaluation_report.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)

    print("=" * 70)
    print("EVALUATION SUMMARY (synthetic test set — not clinical validation)")
    print("=" * 70)
    for r in (report1, report2):
        print(f"\n--- {r['model']} ---")
        print(f"Accuracy: {r['accuracy']:.3f}   Macro-F1: {r['macro_f1']:.3f}")
        print(f"Per-class recall: {r['per_class_recall']}")
        print(f"UNDER-triage rate (predicted LESS urgent than truth): {r['under_triage_rate_overall']:.3f}  "
              f"({r['under_triage_count']} patients)")
        print(f"Critical-class (ESI1-2) recall: {r['critical_critical_recall']}  "
              f"missed={r['critical_critical_missed_count']}/{r['critical_n_true_critical']}")
        print("Confusion matrix (rows=true ESI1..5, cols=pred ESI1..5):")
        for row in r["confusion_matrix"]:
            print("  ", row)
    print(f"\nModel1/Model2 disagreement rate on test set: {disagreement_rate:.1%}")
    print(f"Proxy insufficient-data rate (>=3 missing fields) in test set: {insufficient_rate:.1%}")
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
