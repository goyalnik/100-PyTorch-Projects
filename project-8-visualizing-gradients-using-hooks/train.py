# ============================================
# Author: Nikhil Goyal
# GitHub: https://github.com/goyalnik
# Project: 100 PyTorch Projects - Project 8
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# Generate synthetic data
X = torch.randn(10, 3)
Y = torch.randn(10, 1)

class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.fc1 = nn.Linear(3, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = SmallNet()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Gradient storage
gradients = {}

def print_grad_hook(module, grad_input, grad_output):
    print(f"\n--- Gradient for {module.__class__.__name__} ---")
    print(f"Grad Output Shape: {grad_output[0].shape}")
    print(f"Grad Output Mean: {grad_output[0].mean().item():.6f}")
    gradients['fc1_grad'] = grad_output[0]

# Register backward hook
hook = model.fc1.register_backward_hook(print_grad_hook)

# Training for 5 epochs
for epoch in range(5):
    output = model(X)
    loss = criterion(output, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Remove hook
hook.remove()
print("\nHook removed!")

torch.save(model.state_dict(), 'model.pth')
print("Model saved!")