"""
pipeline.py
===========
Loads trained artifacts once and exposes `assess_patient(patient_dict)` to
run a single patient through: preprocessing -> Model 1 -> Model 2 ->
verification engine -> (safety layer + uncertainty already inside
verification.run_verification). Used by both app.py and demo_patients.py
so there is exactly one code path from raw intake to a TriageDecision.
"""

from __future__ import annotations
import os
import joblib
import pandas as pd

from .preprocessing import full_preprocess, build_numeric_matrix, build_text_corpus
from .verification import run_verification

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TriagePipeline:
    def __init__(self):
        self.model1 = joblib.load(os.path.join(BASE, "models", "model1", "model1.joblib"))
        self.model2 = joblib.load(os.path.join(BASE, "models", "model2", "model2.joblib"))
        self.vectorizers = joblib.load(os.path.join(BASE, "models", "text_vectorizers.joblib"))
        self.numeric_columns = None  # set after first call for UI display

    def assess_patient(self, patient: dict):
        """patient: dict with raw intake fields (may contain None for
        missing values). Returns a TriageDecision."""
        raw_df = pd.DataFrame([patient])
        df = full_preprocess(raw_df)
        Xn_df = build_numeric_matrix(df)
        self.numeric_columns = list(Xn_df.columns)
        corpus = build_text_corpus(df)

        Xn = Xn_df.values
        Xt1 = self.vectorizers.transform_model1(corpus)
        Xt2 = self.vectorizers.transform_model2(corpus)

        decision = run_verification(patient, self.model1, self.model2, Xn, Xt1, Xt2)
        return decision


_PIPELINE_SINGLETON = None


def get_pipeline() -> TriagePipeline:
    global _PIPELINE_SINGLETON
    if _PIPELINE_SINGLETON is None:
        _PIPELINE_SINGLETON = TriagePipeline()
    return _PIPELINE_SINGLETON
