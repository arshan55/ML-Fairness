"""
03_fairness_evaluation.py
Implements Multi-Metric Fairness Evaluation.
Calculates Demographic Parity, Equalized Odds, Disparate Impact natively and using fairlearn.
Includes concepts like Propensity Score Matching for Matched Counterpart Auditing.
"""
import os
import joblib
import pandas as pd
import numpy as np
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.neighbors import NearestNeighbors

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

def calculate_disparate_impact(y_true, y_pred, sensitive_features, group_1, group_0):
    """
    Calculates Disparate Impact (DI). 
    DI = P(y_pred=1 | group=group_1) / P(y_pred=1 | group=group_0)
    """
    df = pd.DataFrame({'y_pred': y_pred, 'group': sensitive_features})
    prob_1 = df[df['group'] == group_1]['y_pred'].mean()
    prob_0 = df[df['group'] == group_0]['y_pred'].mean()
    
    if prob_0 == 0:
        return np.inf
    return prob_1 / prob_0

def evaluate_model_fairness(model, X_test, y_test, protected_attr='sex', group_1=1, group_0=0):
    print(f"\n--- Fairness Evaluation for Model ---")
    sensitive_features = X_test[protected_attr]
    y_pred = model.predict(X_test)
    
    dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_features)
    eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_features)
    di = calculate_disparate_impact(y_test, y_pred, sensitive_features, group_1, group_0)
    
    print(f"Demographic Parity Difference: {dp_diff:.4f} (Ideal: 0)")
    print(f"Equalized Odds Difference:     {eo_diff:.4f} (Ideal: 0)")
    print(f"Disparate Impact:              {di:.4f} (Ideal: 1)")
    
    return {'dp_diff': dp_diff, 'eo_diff': eo_diff, 'di': di}

def matched_counterpart_auditing(X_test, y_pred, protected_attr='sex', group_1=1, group_0=0):
    """
    Simulates Propensity Score Matching / Matched Counterparts auditing from Objective 2.
    We match individuals from group_1 to individuals in group_0 based on their covariates (excluding the protected attr).
    """
    print("\n--- Auditing with Matched Counterparts ---")
    X_g1 = X_test[X_test[protected_attr] == group_1].drop(columns=[protected_attr])
    X_g0 = X_test[X_test[protected_attr] == group_0].drop(columns=[protected_attr])
    
    y_pred_g1 = y_pred[X_test[protected_attr] == group_1]
    y_pred_g0 = y_pred[X_test[protected_attr] == group_0]
    
    # Simple K-NN matching (k=1)
    if len(X_g1) > 0 and len(X_g0) > 0:
        nn = NearestNeighbors(n_neighbors=1).fit(X_g0)
        distances, indices = nn.kneighbors(X_g1)
        
        # Compare outcomes of matched pairs
        matched_g0_preds = y_pred_g0.iloc[indices.flatten()].values
        matching_discrepancy = np.mean(y_pred_g1 != matched_g0_preds)
        
        print(f"Matched Counterpart Discrepancy Rate: {matching_discrepancy:.4f}")
        print(f"Measures systematic bias ignoring covariate differences.")
        return matching_discrepancy
    return None

if __name__ == "__main__":
    from importlib.machinery import SourceFileLoader
    preproc = SourceFileLoader("preproc", os.path.join(os.path.dirname(__file__), "01_preprocessing_and_symptoms.py")).load_module()
    baseline = SourceFileLoader("baseline", os.path.join(os.path.dirname(__file__), "02_baseline_training.py")).load_module()
    
    df = preproc.load_data()
    if df is not None:
        df_processed = preproc.preprocess_data(df)
        _, X_test, y_test = baseline.train_baseline(df_processed)
        
        # Load the baseline random forest to evaluate
        model_path = os.path.join(MODEL_DIR, "baseline_RandomForest.pkl")
        if os.path.exists(model_path):
            rf_model = joblib.load(model_path)
            evaluate_model_fairness(rf_model, X_test, y_test)
            y_pred = pd.Series(rf_model.predict(X_test), index=X_test.index)
            matched_counterpart_auditing(X_test, y_pred)
