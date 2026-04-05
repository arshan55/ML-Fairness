"""
01_preprocessing_and_symptoms.py
Handles Data Preprocessing and initial Bias Detection using dataset symptoms.
Provides dataset diagnostic functionality.
"""
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import wasserstein_distance

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets', 'raw_data')

def load_data(dataset_name='adult.csv'):
    path = os.path.join(DATA_DIR, dataset_name)
    if not os.path.exists(path):
        print(f"Data not found at {path}. Please run datasets/fetch_data.py first.")
        return None
    
    df = pd.read_csv(path)
    # Basic cleaning
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def preprocess_data(df, target_col='income', protected_attr='sex'):
    print(f"--- Preprocessing Data ---")
    le = LabelEncoder()
    # Encode target
    df[target_col] = le.fit_transform(df[target_col].astype(str))
    
    # Encode categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Numerical normalization
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    num_cols = [c for c in num_cols if c not in [target_col, protected_attr] and c in df.columns]
    
    if len(num_cols) > 0:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        
    print(f"Preprocessed DataFrame shape: {df.shape}")
    return df

def calculate_bias_symptoms(df, target_col='income', protected_attr='sex'):
    """
    Quantifies intrinsic dataset bias before model training.
    Uses Mutual Information and Earth Mover's Distance (Wasserstein Distance).
    """
    print(f"\n--- Calculating Bias Symptoms ---")
    
    # 1. Mutual Information between Protected Attribute and Target
    if df[protected_attr].dtype == 'float64':
        df[protected_attr] = df[protected_attr].astype(int)
    
    mi = mutual_info_classif(df[[protected_attr]], df[target_col], discrete_features=True)
    print(f"Mutual Information ({protected_attr} -> {target_col}): {mi[0]:.4f}")
    
    # 2. Earth Mover's Distance (EMD / Wasserstein Distance) 
    # Compare the distribution of the target based on the protected attribute groups
    groups = df[protected_attr].unique()
    if len(groups) >= 2:
        group_0_target = df[df[protected_attr] == groups[0]][target_col].values
        group_1_target = df[df[protected_attr] == groups[1]][target_col].values
        
        emd = wasserstein_distance(group_0_target, group_1_target)
        print(f"Earth Mover's Distance (EMD) between group {groups[0]} and {groups[1]} for {target_col}: {emd:.4f}")
    
    return mi[0], emd if len(groups) >= 2 else 0

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        df_processed = preprocess_data(df, target_col='income', protected_attr='sex')
        calculate_bias_symptoms(df_processed, target_col='income', protected_attr='sex')
        
        # Save processed data explicitly for the user's research
        processed_dir = os.path.join(os.path.dirname(DATA_DIR), 'processed_data')
        os.makedirs(processed_dir, exist_ok=True)
        out_path = os.path.join(processed_dir, 'adult_processed.csv')
        df_processed.to_csv(out_path, index=False)
        print(f"Processed data saved securely to: {out_path}")

