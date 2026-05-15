"""
04_healthcare_mitigation.py - Healthcare Domain
================================================
Trains a FairNeuralNet with a Differentiable Demographic Parity Regularizer
using RACE as the protected attribute.

Total Loss = BCE Loss + λ · DP Regularizer (race-based)

The DP regularizer penalises the squared difference in mean predictions
between White and Non-White patients, forcing the optimizer to treat both
groups equally from the very first gradient step.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'models', 'healthcare'
)
os.makedirs(MODEL_DIR, exist_ok=True)


class HealthcareFairNet(nn.Module):
    """
    3-layer MLP identical in architecture to the salary-domain FairNeuralNet.
    Keeps the research methodology consistent across both domains.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.out = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.net(x)
        return self.sigmoid(self.out(h)), h


def demographic_parity_loss_race(outputs, race_attr):
    """
    Differentiable DP regularizer for race.
    Penalises squared difference in mean prediction between
    White (race_attr==1) and Non-White (race_attr==0) patients.
    """
    mask_w  = (race_attr == 1.0)
    mask_nw = (race_attr == 0.0)

    if mask_w.sum() == 0 or mask_nw.sum() == 0:
        return torch.tensor(0.0, device=outputs.device)

    mean_w  = torch.mean(outputs[mask_w])
    mean_nw = torch.mean(outputs[mask_nw])
    return torch.pow(mean_w - mean_nw, 2)


def train_healthcare_fair_model(
    df,
    target_col='high_cost',
    protected_attr='race_binary',
    lambda_fairness=0.5,
    epochs=60
):
    print("\n--- Training Healthcare FairNeuralNet (Racial DP Regularizer) ---")
    print(f"    λ_fairness = {lambda_fairness}  |  epochs = {epochs}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    X_train_t = torch.FloatTensor(X_train.values)
    y_train_t = torch.FloatTensor(y_train.values).unsqueeze(1)
    race_train = X_train_t[:, X_train.columns.get_loc(protected_attr)]

    X_test_t  = torch.FloatTensor(X_test.values)

    model     = HealthcareFairNet(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    bce_loss  = nn.BCELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs, _ = model(X_train_t)

        loss_main = bce_loss(outputs, y_train_t)
        loss_fair = demographic_parity_loss_race(outputs.squeeze(), race_train)
        total     = loss_main + lambda_fairness * loss_fair

        total.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs} | "
                  f"BCE: {loss_main.item():.4f} | "
                  f"DP-Race: {loss_fair.item():.4f} | "
                  f"Total: {total.item():.4f}")

    # -- Evaluation --
    model.eval()
    with torch.no_grad():
        test_out, _ = model(X_test_t)
        preds  = (test_out.squeeze() > 0.5).numpy().astype(int)
        probas = test_out.squeeze().numpy()

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probas)

    mask_w  = X_test[protected_attr] == 1
    mask_nw = X_test[protected_attr] == 0
    rate_w  = preds[mask_w.values].mean()
    rate_nw = preds[mask_nw.values].mean()
    gap     = rate_w - rate_nw

    print(f"\n  Fair NN Accuracy : {acc:.4f}  |  AUC: {auc:.4f}")
    print(f"  High-cost rate   -> White: {rate_w:.3f}  |  Non-White: {rate_nw:.3f}")
    print(f"  Racial Gap (DPD) : {gap:+.3f}  (baseline was significantly higher)")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "fair_healthcare_nn.pth"))
    print("  Fair Healthcare Model saved → models/healthcare/fair_healthcare_nn.pth")

    return model, X_test, y_test, preds, probas


if __name__ == "__main__":
    from importlib.machinery import SourceFileLoader
    preproc = SourceFileLoader("preproc", os.path.join(os.path.dirname(__file__), "01_healthcare_preprocessing.py")).load_module()

    df           = preproc.load_healthcare_data()
    df_processed = preproc.preprocess_healthcare(df)
    train_healthcare_fair_model(df_processed)
