"""
fairness_metrics.py
===================
Native implementations of fairness metrics, replacing the fairlearn dependency.
Used by both the salary and healthcare pipeline domains.

Implements:
  - demographic_parity_difference  (DPD)
  - equalized_odds_difference       (EOD)
"""

import numpy as np


def demographic_parity_difference(y_true, y_pred, sensitive_features):
    """
    Demographic Parity Difference (DPD).
    DPD = max_group P(ŷ=1|A=g) − min_group P(ŷ=1|A=g)
    Ideal value: 0 (equal positive prediction rates across all groups).
    """
    y_pred = np.array(y_pred)
    sf     = np.array(sensitive_features)
    groups = np.unique(sf)
    rates  = [y_pred[sf == g].mean() for g in groups]
    return float(max(rates) - min(rates))


def equalized_odds_difference(y_true, y_pred, sensitive_features):
    """
    Equalized Odds Difference (EOD).
    EOD = max(|TPR_A − TPR_B|, |FPR_A − FPR_B|) across all group pairs.
    Ideal value: 0 (equal TPR and FPR across all groups).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sf     = np.array(sensitive_features)
    groups = np.unique(sf)

    tprs, fprs = [], []
    for g in groups:
        mask = sf == g
        tp = ((y_pred[mask] == 1) & (y_true[mask] == 1)).sum()
        fn = ((y_pred[mask] == 0) & (y_true[mask] == 1)).sum()
        fp = ((y_pred[mask] == 1) & (y_true[mask] == 0)).sum()
        tn = ((y_pred[mask] == 0) & (y_true[mask] == 0)).sum()

        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)

    tpr_gap = max(tprs) - min(tprs)
    fpr_gap = max(fprs) - min(fprs)
    return float(max(tpr_gap, fpr_gap))
