# ============================================
# Author: Nikhil Goyal
# GitHub: https://github.com/goyalnik
# Project: 100 PyTorch Projects - Project 18
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

X_test = torch.randn(5, 10)

# Method 1: Load state_dict
print("--- Method 1: Load state_dict ---")
model1 = SimpleMLP()
model1.load_state_dict(torch.load("model_state.pth"))
model1.eval()
with torch.no_grad():
    preds1 = model1(X_test)
print("Predictions:", preds1.view(-1).tolist())

# Method 2: Load full model
print("\n--- Method 2: Load full model ---")
model2 = torch.load("model_full.pth")
model2.eval()
with torch.no_grad():
    preds2 = model2(X_test)
print("Predictions:", preds2.view(-1).tolist())

# Method 3: Load checkpoint
print("\n--- Method 3: Load checkpoint ---")
model3 = SimpleMLP()
checkpoint = torch.load("model_checkpoint.pth")
model3.load_state_dict(checkpoint['model_state_dict'])
model3.eval()
print(f"Loaded from Epoch: {checkpoint['epoch']}")
print(f"Saved Loss: {checkpoint['loss']:.4f}")
with torch.no_grad():
    preds3 = model3(X_test)
print("Predictions:", preds3.view(-1).tolist())