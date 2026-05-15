"""
02_healthcare_baseline.py - Healthcare Domain
==============================================
Trains standard (unmitigated) models on the biased healthcare insurance dataset.
These models absorb racial access disparities as genuine health signal, producing
a system that systematically under-predicts cost risk for non-white patients.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'models', 'healthcare'
)
os.makedirs(MODEL_DIR, exist_ok=True)


def train_healthcare_baseline(df, target_col='high_cost', protected_attr='race_binary'):
    print("\n--- Training Healthcare Baseline Models (No Fairness Constraint) ---")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

    mask_w  = X_test[protected_attr] == 1
    mask_nw = X_test[protected_attr] == 0
    print(f"Test set - White patients: {mask_w.sum()}  |  Non-White: {mask_nw.sum()}")

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest':       RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)

        # Racial disparity in positive (high-cost) prediction rates
        rate_w  = preds[mask_w].mean()
        rate_nw = preds[mask_nw].mean()
        gap     = rate_w - rate_nw

        print(f"  Accuracy: {acc:.4f}  |  AUC: {auc:.4f}")
        print(f"  High-cost prediction rate -> White: {rate_w:.3f}  |  Non-White: {rate_nw:.3f}")
        print(f"  Racial Gap (DPD proxy): {gap:+.3f}  <- positive means White patients flagged MORE as high-cost")
        print(f"  [!] Non-white patients are UNDER-identified as high-risk -> denied access to richer coverage tiers")

        joblib.dump(model, os.path.join(MODEL_DIR, f"baseline_{name}.pkl"))
        print(f"  Model saved -> models/healthcare/baseline_{name}.pkl")

        results[name] = {
            'accuracy': acc, 'auc': auc, 'model': model,
            'rate_w': rate_w, 'rate_nw': rate_nw, 'gap': gap
        }

    return results, X_test, y_test


if __name__ == "__main__":
    from importlib.machinery import SourceFileLoader
    preproc = SourceFileLoader("preproc", os.path.join(os.path.dirname(__file__), "01_healthcare_preprocessing.py")).load_module()

    df = preproc.load_healthcare_data()
    df_processed = preproc.preprocess_healthcare(df)
    train_healthcare_baseline(df_processed)
