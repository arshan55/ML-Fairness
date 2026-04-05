"""
04_mitigation.py
Implements advanced mitigation strategies.
1. PyTorch Neural Network with Differentiable Fairness Penalty (Proxy for MI Regularizer)
2. BiasPruner conceptual approach (pruning weights most correlated with protected attribute)
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

class FairNeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(FairNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        out = self.sigmoid(self.fc3(h2))
        return out, h2 # Returning hidden layer for BiasPruner

def demographic_parity_loss(outputs, sensitive_attributes):
    """
    Differentiable proxy for Demographic Parity to use as a regularizer.
    Penalizes the squared difference between the mean prediction for group 0 and group 1.
    """
    mask_g1 = (sensitive_attributes == 1.0)
    mask_g0 = (sensitive_attributes == 0.0)
    
    if mask_g1.sum() == 0 or mask_g0.sum() == 0:
        return torch.tensor(0.0, device=outputs.device)
        
    mean_g1 = torch.mean(outputs[mask_g1])
    mean_g0 = torch.mean(outputs[mask_g0])
    
    return torch.pow(mean_g1 - mean_g0, 2)

def train_fair_model(X, y, protected_attr_idx, lambda_fairness=0.5, epochs=50):
    print("\n--- Training Fair Neural Network (Differentiable Regularizer) ---")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train.values)
    y_train_t = torch.FloatTensor(y_train.values).unsqueeze(1)
    sensitive_train = X_train_t[:, protected_attr_idx]
    
    X_test_t = torch.FloatTensor(X_test.values)
    
    model = FairNeuralNet(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    bce_loss = nn.BCELoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs, _ = model(X_train_t)
        
        # Standard Objective
        loss_main = bce_loss(outputs, y_train_t)
        
        # Fairness Objective (Mutual Information / DP proxy)
        loss_fair = demographic_parity_loss(outputs, sensitive_train)
        
        # Total Loss
        total_loss = loss_main + lambda_fairness * loss_fair
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Main Loss: {loss_main.item():.4f} | Fair Loss: {loss_fair.item():.4f}")
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs, _ = model(X_test_t)
        preds = (test_outputs.squeeze() > 0.5).numpy().astype(int)
        acc = accuracy_score(y_test, preds)
        print(f"Fair NN Accuracy: {acc:.4f}")
        
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "fair_nn.pth"))
    print("Fair Model saved successfully.")
    
    return model, X_test, y_test, preds

if __name__ == "__main__":
    from importlib.machinery import SourceFileLoader
    preproc = SourceFileLoader("preproc", os.path.join(os.path.dirname(__file__), "01_preprocessing_and_symptoms.py")).load_module()
    
    df = preproc.load_data()
    if df is not None:
        df_processed = preproc.preprocess_data(df)
        X = df_processed.drop(columns=['income'])
        y = df_processed['income']
        # Find index of 'sex'
        protected_attr_idx = X.columns.get_loc('sex')
        train_fair_model(X, y, protected_attr_idx)
