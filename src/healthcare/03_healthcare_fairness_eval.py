"""
03_healthcare_fairness_eval.py - Healthcare Domain
===================================================
Multi-metric fairness evaluation with RACE as the protected attribute.

Metrics computed:
  1. Demographic Parity Difference (DPD)     — equal positive prediction rates
  2. Equalized Odds Difference (EOD)         — equal TPR & FPR across groups
  3. Disparate Impact (DI)                   — ratio of positive rates (80% rule)
  4. Matched Counterpart Discrepancy Rate    — KNN-based individual fairness audit
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Native fairness metrics (no fairlearn dependency)
_src = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, _src)
from fairness_metrics import demographic_parity_difference, equalized_odds_difference

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'models', 'healthcare'
)


def calculate_disparate_impact(y_pred, sensitive_features, privileged=1, unprivileged=0):
    """
    DI = P(ŷ=1 | Non-White) / P(ŷ=1 | White)
    80% rule: DI < 0.8 indicates illegal disparate impact in US law.
    """
    df = pd.DataFrame({'y_pred': y_pred, 'group': sensitive_features})
    rate_priv   = df[df['group'] == privileged]['y_pred'].mean()
    rate_unpriv = df[df['group'] == unprivileged]['y_pred'].mean()
    if rate_priv == 0:
        return np.inf
    return rate_unpriv / rate_priv


def evaluate_healthcare_fairness(model, X_test, y_test, protected_attr='race_binary'):
    print(f"\n--- Healthcare Fairness Evaluation: {type(model).__name__} ---")

    sensitive = X_test[protected_attr]
    y_pred    = model.predict(X_test)

    dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive)
    eod = equalized_odds_difference(y_test,    y_pred, sensitive_features=sensitive)
    di  = calculate_disparate_impact(y_pred, sensitive)

    print(f"  Demographic Parity Difference : {dpd:+.4f}  (Ideal: 0)")
    print(f"  Equalized Odds Difference     : {eod:+.4f}  (Ideal: 0)")
    print(f"  Disparate Impact              :  {di:.4f}  (Ideal: 1.0 | Legal threshold: >=0.8)")

    if di < 0.8:
        print(f"  [!] DISPARATE IMPACT VIOLATION - Non-white patients are high-cost-flagged at")
        print(f"      only {100*di:.1f}% the rate of White patients. This fails the 80% legal rule.")

    return {'dpd': dpd, 'eod': eod, 'di': di}


def matched_counterpart_audit(X_test, y_pred, protected_attr='race_binary',
                               privileged=1, unprivileged=0):
    """
    KNN-based individual fairness audit.
    Matches each non-white patient to their nearest white counterpart
    (same age, bmi, conditions, income) and checks if the model gave them
    a DIFFERENT prediction — pure discriminatory effect.
    """
    print("\n--- Matched Counterpart Audit (Individual Fairness) ---")

    X_priv   = X_test[X_test[protected_attr] == privileged].drop(columns=[protected_attr])
    X_unpriv = X_test[X_test[protected_attr] == unprivileged].drop(columns=[protected_attr])

    y_priv   = np.array(y_pred)[X_test[protected_attr] == privileged]
    y_unpriv = np.array(y_pred)[X_test[protected_attr] == unprivileged]

    if len(X_priv) == 0 or len(X_unpriv) == 0:
        print("  Insufficient data for matching.")
        return None

    nn = NearestNeighbors(n_neighbors=1).fit(X_priv)
    _, indices = nn.kneighbors(X_unpriv)

    matched_priv_preds   = y_priv[indices.flatten()]
    discrepancy_rate     = np.mean(y_unpriv != matched_priv_preds)

    print(f"  Matched pairs analysed: {len(X_unpriv)}")
    print(f"  Prediction discrepancy rate: {discrepancy_rate:.4f}")
    print(f"  -> {100*discrepancy_rate:.1f}% of non-white patients received a DIFFERENT")
    print(f"    prediction than their demographically identical white counterpart.")

    return discrepancy_rate


if __name__ == "__main__":
    from importlib.machinery import SourceFileLoader
    preproc  = SourceFileLoader("preproc",   os.path.join(os.path.dirname(__file__), "01_healthcare_preprocessing.py")).load_module()
    baseline = SourceFileLoader("baseline",  os.path.join(os.path.dirname(__file__), "02_healthcare_baseline.py")).load_module()

    df           = preproc.load_healthcare_data()
    df_processed = preproc.preprocess_healthcare(df)
    results, X_test, y_test = baseline.train_healthcare_baseline(df_processed)

    for name, res in results.items():
        print(f"\n=== {name} ===")
        metrics = evaluate_healthcare_fairness(res['model'], X_test, y_test)
        y_pred  = pd.Series(res['model'].predict(X_test), index=X_test.index)
        matched_counterpart_audit(X_test, y_pred.values)
