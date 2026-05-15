"""
01_healthcare_preprocessing.py - Healthcare Domain
===================================================
Dataset  : Synthetic Healthcare Insurance dataset (calibrated to MEPS Panel-21 statistics)
Target   : high_cost - binary flag (annual charges > $7,000)
           Used by insurers to place patients into premium tiers.
Protected: race_binary - White=1 (privileged), Non-White=0 (unprivileged)

Bias Story:
  Non-white patients historically have LOWER recorded healthcare utilization
  not because they are healthier, but due to systemic access barriers
  (under-insurance, cost avoidance, geographic deserts).
  A naive model learns this pattern and classifies non-white patients as
  "low cost" -> insurer under-prices preventive care and denies high-tier
  coverage -> when they eventually seek care it is acute/expensive ->
  premiums spike dramatically in subsequent cycles (discriminatory feedback loop).
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import wasserstein_distance

HEALTHCARE_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'datasets', 'healthcare'
)


def generate_healthcare_data(n_samples=12000, random_state=42, save=True):
    """
    Generates a synthetic healthcare insurance dataset calibrated to MEPS statistics.
    Introduces realistic racial disparities reflecting documented access barriers.
    """
    np.random.seed(random_state)

    # -- Demographics ----
    race_cats  = ['White', 'Black', 'Hispanic', 'Asian']
    race_probs = [0.60,    0.15,    0.20,       0.05]
    race = np.random.choice(race_cats, size=n_samples, p=race_probs)

    age    = np.random.normal(45, 14, n_samples).clip(18, 85).astype(int)
    sex    = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.49, 0.51])
    bmi    = np.random.normal(28.5, 6.0, n_samples).clip(15, 55)
    smoker = np.random.choice([0, 1], size=n_samples, p=[0.82, 0.18])

    # Income - correlated with race (a classic proxy-discrimination variable)
    income_base = {'White': 72000, 'Asian': 85000, 'Black': 48000, 'Hispanic': 45000}
    income = np.array([
        np.random.normal(income_base[r], 18000) for r in race
    ]).clip(15000, 250000)

    region = np.random.choice(['Northeast', 'Midwest', 'South', 'West'], size=n_samples)

    # Chronic conditions - driven by actual health factors
    num_conditions = np.random.poisson(
        0.8 + 0.02 * (age - 30) + 0.01 * bmi + 1.2 * smoker
    ).clip(0, 8)

    # Insurance type - reflects documented access disparities by race (MEPS statistics)
    ins_probs = {
        'White':    [0.65, 0.10, 0.20, 0.05],  # employer, private, medicaid, uninsured
        'Black':    [0.45, 0.08, 0.35, 0.12],
        'Hispanic': [0.38, 0.07, 0.35, 0.20],
        'Asian':    [0.70, 0.15, 0.10, 0.05],
    }
    insurance_type = np.array([
        np.random.choice(
            ['Employer', 'Private', 'Medicaid', 'Uninsured'], p=ins_probs[r]
        ) for r in race
    ])

    # -- Charge Generation (the bias mechanism) --
    # True health-need-based charges
    base_charges = (
        3000
        + 80  * age
        + 120 * num_conditions
        + 5000 * smoker
        + 50  * bmi
        + np.random.normal(0, 2500, n_samples)
    ).clip(500, 80000)

    # BIAS: Non-white patients have lower OBSERVED utilization due to access barriers.
    # These multipliers reflect published MEPS disparity findings.
    access_multiplier = np.where(race == 'White',    1.00,
                        np.where(race == 'Asian',    0.95,
                        np.where(race == 'Black',    0.72,   # 28% lower recorded utilization
                                                     0.68))) # Hispanic: 32% lower

    observed_charges = (base_charges * access_multiplier).clip(500, 80000)

    # Binary target: high-cost flag (>$7,000/year = "high-risk" premium tier)
    actual_high_cost   = (base_charges       > 7000).astype(int)  # true medical need
    observed_high_cost = (observed_charges   > 7000).astype(int)  # biased label model sees

    df = pd.DataFrame({
        'age':              age,
        'sex':              sex,
        'race':             race,
        'bmi':              bmi,
        'smoker':           smoker,
        'region':           region,
        'income':           income,
        'num_conditions':   num_conditions,
        'insurance_type':   insurance_type,
        'observed_charges': observed_charges,
        'actual_charges':   base_charges,
        'high_cost':        observed_high_cost,   # biased training target
        'actual_high_cost': actual_high_cost,     # ground-truth health need
    })

    if save:
        os.makedirs(HEALTHCARE_DATA_DIR, exist_ok=True)
        path = os.path.join(HEALTHCARE_DATA_DIR, 'healthcare_insurance.csv')
        df.to_csv(path, index=False)
        print(f"Healthcare dataset saved to: {path}  ({len(df):,} patients)")

    return df


def load_healthcare_data():
    path = os.path.join(HEALTHCARE_DATA_DIR, 'healthcare_insurance.csv')
    if not os.path.exists(path):
        print("Healthcare data not found. Generating synthetic dataset...")
        return generate_healthcare_data()
    df = pd.read_csv(path)
    print(f"Healthcare data loaded: {len(df):,} patients")
    return df


def preprocess_healthcare(df, target_col='high_cost', protected_attr='race_binary'):
    print("\n--- Preprocessing Healthcare Data ---")
    df = df.copy()

    # Binarize race: White=1 (privileged), Non-White=0 (unprivileged)
    df['race_binary'] = (df['race'] == 'White').astype(int)

    # Encode categoricals
    le = LabelEncoder()
    for col in ['sex', 'region', 'insurance_type']:
        df[col] = le.fit_transform(df[col])

    # Drop columns irrelevant to modelling phase
    drop_cols = ['race', 'observed_charges', 'actual_charges', 'actual_high_cost']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Standardise numerics
    num_cols = ['age', 'bmi', 'income', 'num_conditions']
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    print(f"Preprocessed shape: {df.shape}")
    print(f"High-cost rate -> White: {100*df[df['race_binary']==1][target_col].mean():.1f}%  |  Non-White: {100*df[df['race_binary']==0][target_col].mean():.1f}%")
    return df


def calculate_healthcare_bias_symptoms(df, target_col='high_cost', protected_attr='race_binary'):
    """
    Quantifies intrinsic dataset bias before any model is trained.
    Uses Mutual Information and Wasserstein Distance — same methodology as salary domain.
    """
    print("\n--- Healthcare Bias Symptom Detection ---")

    mi = mutual_info_classif(df[[protected_attr]], df[target_col], discrete_features=True)
    print(f"Mutual Information (race -> high_cost): {mi[0]:.4f}")

    g_white    = df[df[protected_attr] == 1][target_col].values
    g_nonwhite = df[df[protected_attr] == 0][target_col].values
    emd = wasserstein_distance(g_white, g_nonwhite)
    print(f"Earth Mover's Distance (White vs Non-White): {emd:.4f}")
    print(f"  -> Non-zero EMD confirms racial disparity is baked into the data distribution.")

    return mi[0], emd


if __name__ == "__main__":
    df = load_healthcare_data()
    df_processed = preprocess_healthcare(df)
    calculate_healthcare_bias_symptoms(df_processed)
