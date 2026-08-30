"""
model1.py
=========
Model 1 — primary ML triage classifier.

Architecture: XGBoost multiclass classifier over
  [structured numeric features + missingness indicators + derived features]
  concatenated with
  [word-level TF-IDF features over chief complaint / symptoms / history].

Falls back to RandomForest if xgboost is unavailable in the environment
(spec section 4).
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import joblib

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from .version import MODEL_VERSION, PREPROCESSING_VERSION


class Model1:
    ESI_CLASSES = [1, 2, 3, 4, 5]  # label order

    def __init__(self):
        self.scaler = StandardScaler()
        if HAS_XGB:
            self.clf = XGBClassifier(
                n_estimators=250, max_depth=5, learning_rate=0.08,
                subsample=0.85, colsample_bytree=0.85,
                objective="multi:softprob", num_class=5,
                eval_metric="mlogloss", random_state=42, n_jobs=-1,
            )
            self.backend = "XGBoost (gradient-boosted trees)"
        else:
            self.clf = RandomForestClassifier(n_estimators=400, max_depth=12, random_state=42, n_jobs=-1)
            self.backend = "RandomForest (xgboost unavailable, fallback)"
        self.feature_names_numeric = None

    def _label_to_idx(self, y):
        return np.array([self.ESI_CLASSES.index(v) for v in y])

    def _idx_to_label(self, idx):
        return self.ESI_CLASSES[idx]

    def fit(self, X_numeric, X_text_sparse, y, feature_names=None):
        self.feature_names_numeric = feature_names
        Xn = self.scaler.fit_transform(X_numeric)
        X = sp.hstack([sp.csr_matrix(Xn), X_text_sparse]).tocsr()
        y_idx = self._label_to_idx(y)
        self.clf.fit(X, y_idx)
        return self

    def _build_X(self, X_numeric, X_text_sparse):
        Xn = self.scaler.transform(X_numeric)
        return sp.hstack([sp.csr_matrix(Xn), X_text_sparse]).tocsr()

    def predict_proba(self, X_numeric, X_text_sparse) -> np.ndarray:
        X = self._build_X(X_numeric, X_text_sparse)
        return self.clf.predict_proba(X)

    def predict_single(self, x_numeric_row, x_text_row) -> dict:
        """x_numeric_row: (1, n_features) array. x_text_row: (1, n_text_features) sparse."""
        probs = self.predict_proba(x_numeric_row, x_text_row)[0]
        top_idx = int(np.argmax(probs))
        esi = self._idx_to_label(top_idx)
        sorted_probs = np.sort(probs)[::-1]
        margin = float(sorted_probs[0] - sorted_probs[1])
        return {
            "esi_prediction": esi,
            "probabilities": {self.ESI_CLASSES[i]: float(probs[i]) for i in range(len(self.ESI_CLASSES))},
            "confidence": float(probs[top_idx]),  # NOTE: this is the model's raw predicted
            # probability, not "clinical confidence" — see verification.py / uncertainty.py
            # for the separate system-level uncertainty assessment (spec Section 23).
            "max_probability": float(probs[top_idx]),
            "top2_margin": margin,
            "backend": self.backend,
            "model_version": MODEL_VERSION,
            "preprocessing_version": PREPROCESSING_VERSION,
        }

    def feature_importance(self, top_k: int = 8) -> list:
        if self.feature_names_numeric is None:
            return []
        try:
            importances = self.clf.feature_importances_[: len(self.feature_names_numeric)]
        except Exception:
            return []
        order = np.argsort(importances)[::-1][:top_k]
        return [(self.feature_names_numeric[i], float(importances[i])) for i in order if importances[i] > 0]

    def save(self, path):
        from .atomic_save import atomic_joblib_dump
        atomic_joblib_dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
