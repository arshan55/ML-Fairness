"""
fetch_data.py
=============
Downloads the UCI Adult Income dataset for salary/gender bias analysis.
"""
import os
import urllib.request
import pandas as pd
from sklearn.datasets import fetch_openml

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'raw_data')


def fetch_adult_dataset():
    """Download and save the Adult Income dataset from OpenML."""
    print("Fetching Adult Income dataset from OpenML...")

    # Fetch the dataset
    adult = fetch_openml('adult', version=2, as_frame=True, parser='auto')

    df = adult.frame

    # Save to CSV
    output_path = os.path.join(DATA_DIR, 'adult.csv')
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Adult Income dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    return df


if __name__ == "__main__":
    fetch_adult_dataset()
