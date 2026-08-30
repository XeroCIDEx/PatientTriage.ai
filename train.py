"""
train.py
========
End-to-end training pipeline:
  1. Generate (or load) dataset
  2. Preprocess -> numeric matrix + two text representations
  3. Train Model 1 (XGBoost) and Model 2 (MLP, independent representation)
  4. Save models + vectorizers + scalers to models/model1, models/model2
  5. Print quick sanity metrics (full evaluation lives in evaluate.py)

Run: python3 train.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from src.data_generation import generate_dataset, load_real_dataset, RANDOM_SEED
from src.preprocessing import full_preprocess, build_numeric_matrix, build_text_corpus, TextVectorizers
from src.model1 import Model1
from src.model2 import Model2
from src.atomic_save import atomic_joblib_dump

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL1_DIR = os.path.join(BASE, "models", "model1")
MODEL2_DIR = os.path.join(BASE, "models", "model2")
DATA_DIR = os.path.join(BASE, "data", "synthetic")


def main():
    os.makedirs(MODEL1_DIR, exist_ok=True)
    os.makedirs(MODEL2_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    real_df = load_real_dataset()
    if real_df is not None:
        print("Using real dataset.")
        df_raw = real_df
    else:
        print("No real clinical dataset available in this environment. "
              "Generating clearly-labeled SYNTHETIC dataset (see src/data_generation.py docstring).")
        df_raw = generate_dataset(n=3000, seed=RANDOM_SEED)
        df_raw.to_csv(os.path.join(DATA_DIR, "synthetic_patients.csv"), index=False)

    df = full_preprocess(df_raw)
    X_numeric_df = build_numeric_matrix(df)
    corpus = build_text_corpus(df)
    y = df["esi_label"].values

    idx_train, idx_test = train_test_split(
        df.index, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    vec = TextVectorizers().fit(corpus.loc[idx_train])
    atomic_joblib_dump(vec, os.path.join(BASE, "models", "text_vectorizers.joblib"))

    Xn_train = X_numeric_df.loc[idx_train].values
    Xn_test = X_numeric_df.loc[idx_test].values
    y_train = y[df.index.get_indexer(idx_train)]
    y_test = y[df.index.get_indexer(idx_test)]

    Xt1_train = vec.transform_model1(corpus.loc[idx_train])
    Xt1_test = vec.transform_model1(corpus.loc[idx_test])
    Xt2_train = vec.transform_model2(corpus.loc[idx_train])
    Xt2_test = vec.transform_model2(corpus.loc[idx_test])

    print(f"\nTraining set: {len(idx_train)} | Test set: {len(idx_test)}")
    print("Training Model 1 (XGBoost, structured + word-TFIDF)...")
    m1 = Model1()
    m1.fit(Xn_train, Xt1_train, y_train, feature_names=list(X_numeric_df.columns))
    m1.save(os.path.join(MODEL1_DIR, "model1.joblib"))

    print("Training Model 2 (MLP neural net, structured + char-TFIDF)...")
    m2 = Model2()
    m2.fit(Xn_train, Xt2_train, y_train)
    m2.save(os.path.join(MODEL2_DIR, "model2.joblib"))

    # quick sanity accuracy (full report in evaluate.py)
    from sklearn.metrics import accuracy_score, f1_score
    p1 = m1.predict_proba(Xn_test, Xt1_test).argmax(axis=1) + 1
    p2 = m2.predict_proba(Xn_test, Xt2_test).argmax(axis=1) + 1
    print(f"\n[Model 1] test accuracy={accuracy_score(y_test, p1):.3f}  macro-F1={f1_score(y_test, p1, average='macro'):.3f}")
    print(f"[Model 2] test accuracy={accuracy_score(y_test, p2):.3f}  macro-F1={f1_score(y_test, p2, average='macro'):.3f}")
    agree_rate = (p1 == p2).mean()
    print(f"Model 1 / Model 2 agreement rate on held-out test set: {agree_rate:.1%}")

    # persist test split for evaluate.py
    test_bundle = {
        "idx_test": idx_test, "y_test": y_test,
        "Xn_test": Xn_test, "Xt1_test": Xt1_test, "Xt2_test": Xt2_test,
        "df_test_raw": df.loc[idx_test],
    }
    atomic_joblib_dump(test_bundle, os.path.join(BASE, "models", "test_bundle.joblib"))
    print("\nSaved: models/model1/model1.joblib, models/model2/model2.joblib, "
          "models/text_vectorizers.joblib, models/test_bundle.joblib")


if __name__ == "__main__":
    main()
