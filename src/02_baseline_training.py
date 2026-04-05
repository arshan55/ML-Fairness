"""
02_baseline_training.py
Trains standard (Unfair) models.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets', 'raw_data')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

def train_baseline(df, target_col='income', protected_attr='sex'):
    print("\n--- Training Baseline Models ---")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")
    
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        preds_proba = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, preds_proba)
        
        print(f"{name} Results:")
        print(f"Accuracy: {acc:.4f} | ROC AUC: {auc:.4f}")
        
        # Save model
        joblib.dump(model, os.path.join(MODEL_DIR, f"baseline_{name}.pkl"))
        
        results[name] = {
            'accuracy': acc,
            'auc': auc,
            'model': model
        }
        
    # We return the test sets to be used by the fairness evaluation
    return results, X_test, y_test

if __name__ == "__main__":
    from importlib.machinery import SourceFileLoader
    preproc = SourceFileLoader("preproc", os.path.join(os.path.dirname(__file__), "01_preprocessing_and_symptoms.py")).load_module()
    
    df = preproc.load_data()
    if df is not None:
        df_processed = preproc.preprocess_data(df)
        train_baseline(df_processed)
