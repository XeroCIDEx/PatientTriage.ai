"""
preprocessing.py
=================
Turns raw patient records into feature matrices for Model 1 and Model 2.

Key design point (Section 9 of spec — Missing Data):
  We explicitly separate "value" from "was this value available".
  For every numeric vital we add a `<field>_missing` indicator (1 = the
  value was not available at intake, not "the value was normal").
  Missing numeric values are imputed with an age-band median SOLELY so the
  model has a number to compute on — the missingness indicator is what
  tells the model (and the uncertainty engine) that the value is not
  trustworthy.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .age_bands import age_band as _shared_age_band

NUMERIC_FIELDS = ["age", "heart_rate", "resp_rate", "sbp", "dbp", "spo2", "temperature", "pain_score"]

# Simplified age-band medians used ONLY for imputing missing values before
# feeding a classical ML model (which cannot natively take NaN + xgboost
# actually can, but we impute explicitly so behavior is transparent and the
# missingness indicators carry the real signal).
IMPUTE_MEDIANS = {
    "pediatric": {"heart_rate": 115, "resp_rate": 24, "sbp": 97, "dbp": 60, "spo2": 98, "temperature": 37.2},
    "adult": {"heart_rate": 80, "resp_rate": 16, "sbp": 118, "dbp": 75, "spo2": 98, "temperature": 36.8},
    "geriatric": {"heart_rate": 75, "resp_rate": 17, "sbp": 125, "dbp": 75, "spo2": 96, "temperature": 36.8},
}


def _age_band(age):
    # delegates to the single shared definition in age_bands.py so
    # preprocessing, safety rules, and data generation never disagree
    # about what counts as pediatric/adult/geriatric
    if pd.isna(age):
        return "adult"
    return _shared_age_band(age)


def add_missingness_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_band"] = df["age"].apply(_age_band)

    for field in ["heart_rate", "resp_rate", "sbp", "dbp", "spo2", "temperature", "pain_score"]:
        miss_col = f"{field}_missing"
        df[miss_col] = df[field].isna().astype(int)
        if field == "pain_score":
            df[field] = df[field].fillna(0)  # pain unknown -> not assumed severe, flagged separately
        else:
            df[field] = df.apply(
                lambda r: IMPUTE_MEDIANS[r["age_band"]][field] if pd.isna(r[field]) else r[field], axis=1
            )

    # history: distinguish "confirmed no history" vs "not available"
    if "has_history_info" not in df.columns:
        df["has_history_info"] = df["history"].notna().astype(int)
    df["history_missing"] = (df["has_history_info"] == 0).astype(int)
    df["history"] = df["history"].fillna("history not available")

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Shock index = HR / SBP, a simple, well-known prototype-level derived
    # physiological feature (elevated values are a recognized red flag).
    df["shock_index"] = (df["heart_rate"] / df["sbp"].replace(0, np.nan)).clip(0, 5).fillna(0)
    df["pulse_pressure"] = (df["sbp"] - df["dbp"]).clip(lower=0)
    df["missing_field_count"] = df[[c for c in df.columns if c.endswith("_missing")]].sum(axis=1)
    df["is_pediatric"] = (df["age_band"] == "pediatric").astype(int)
    df["is_geriatric"] = (df["age_band"] == "geriatric").astype(int)
    return df


def build_numeric_matrix(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = (
        ["age", "heart_rate", "resp_rate", "sbp", "dbp", "spo2", "temperature", "pain_score",
         "shock_index", "pulse_pressure", "missing_field_count", "is_pediatric", "is_geriatric",
         "has_history_info"]
        + [c for c in df.columns if c.endswith("_missing")]
    )
    return df[feature_cols].astype(float)


def build_text_corpus(df: pd.DataFrame) -> pd.Series:
    return (df["chief_complaint"].fillna("") + " . " + df["symptoms"].fillna("") + " . " + df["history"].fillna(""))


def full_preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = add_missingness_and_impute(df_raw)
    df = add_derived_features(df)
    return df


class TextVectorizers:
    """Holds two DIFFERENT text vectorizers on purpose:
    - `word_tfidf` (word n-grams) used by Model 1
    - `char_tfidf` (character n-grams) used by Model 2, so Model 2's text
      representation is genuinely different, not a copy of Model 1's.
    """

    def __init__(self):
        self.word_tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 2), min_df=2)
        self.char_tfidf = TfidfVectorizer(max_features=300, analyzer="char_wb", ngram_range=(3, 5), min_df=2)

    def fit(self, corpus: pd.Series):
        self.word_tfidf.fit(corpus)
        self.char_tfidf.fit(corpus)
        return self

    def transform_model1(self, corpus: pd.Series):
        return self.word_tfidf.transform(corpus)

    def transform_model2(self, corpus: pd.Series):
        return self.char_tfidf.transform(corpus)
