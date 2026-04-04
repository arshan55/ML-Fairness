"""
Script to dynamically fetch and cache datasets from the UCI ML Repository.
Focuses on Adult Income (id=2) and Heart Disease (id=45).
"""
import pandas as pd
from ucimlrepo import fetch_ucirepo
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), 'raw_data')
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_adult_income():
    print("Fetching Adult Income dataset...")
    try:
        adult = fetch_ucirepo(id=2) 
        X = adult.data.features 
        y = adult.data.targets 
        df = pd.concat([X, y], axis=1)
        path = os.path.join(DATA_DIR, 'adult.csv')
        df.to_csv(path, index=False)
        print(f"Adult Income saved to {path}")
        return df
    except Exception as e:
        print(f"Error fetching Adult Income: {e}")
        return None

def fetch_heart_disease():
    print("Fetching Heart Disease dataset...")
    try:
        heart_disease = fetch_ucirepo(id=45) 
        X = heart_disease.data.features 
        y = heart_disease.data.targets 
        df = pd.concat([X, y], axis=1)
        path = os.path.join(DATA_DIR, 'heart_disease.csv')
        df.to_csv(path, index=False)
        print(f"Heart Disease saved to {path}")
        return df
    except Exception as e:
        print(f"Error fetching Heart Disease: {e}")
        return None

if __name__ == "__main__":
    df_adult = fetch_adult_income()
    df_heart = fetch_heart_disease()
