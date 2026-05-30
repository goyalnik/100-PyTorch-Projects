# ============================================
# Author: Nikhil Goyal
# GitHub: https://github.com/goyalnik
# Project: 100 PyTorch Projects - Project 8
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.fc1 = nn.Linear(3, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = SmallNet()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Forward hook to capture activations
activations = {}

def forward_hook(module, input, output):
    activations['fc1'] = output

hook = model.fc1.register_forward_hook(forward_hook)

# Test on new data
X_test = torch.randn(5, 3)

with torch.no_grad():
    predictions = model(X_test)

print("Predictions:")
for i, pred in enumerate(predictions):
    print(f"Sample {i+1}: {pred.item():.4f}")

print(f"\nfc1 Activations Shape: {activations['fc1'].shape}")
print(f"fc1 Activations:\n{activations['fc1']}")

hook.remove()