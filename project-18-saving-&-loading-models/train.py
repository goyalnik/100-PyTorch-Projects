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

model = SimpleMLP()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

X = torch.randn(100, 10)
Y = torch.randn(100, 1)

for epoch in range(10):
    outputs = model(X)
    loss = criterion(outputs, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 2 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("\nTraining done!")

# Method 1: Save state_dict (Recommended)
torch.save(model.state_dict(), "model_state.pth")
print("✅ state_dict saved as model_state.pth")

# Method 2: Save entire model
torch.save(model, "model_full.pth")
print("✅ Full model saved as model_full.pth")

# Method 3: Save checkpoint (with optimizer state)
torch.save({
    'epoch': 10,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item()
}, "model_checkpoint.pth")
print("✅ Checkpoint saved as model_checkpoint.pth")