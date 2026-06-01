# ============================================
# Author: Nikhil Goyal
# GitHub: https://github.com/goyalnik
# Project: 100 PyTorch Projects - Project 20
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model1 = SimpleNN()
model2 = SimpleNN()

criterion = nn.CrossEntropyLoss()
optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)

# -------- Method 1: Manual Batching --------
print("=" * 40)
print("Method 1: Manual Batching")
print("=" * 40)

X = train_data.data.view(-1, 1, 28, 28).float() / 255.0
Y = train_data.targets
batch_size = 64

start = time.time()
for epoch in range(3):
    total_loss = 0
    for i in range(0, len(X), batch_size):
        x_batch = X[i:i+batch_size]
        y_batch = Y[i:i+batch_size]
        outputs = model1(x_batch)
        loss = criterion(outputs, y_batch)
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(X)*batch_size:.4f}")

manual_time = time.time() - start
print(f"Manual Batching Time: {manual_time:.2f}s\n")

# -------- Method 2: DataLoader Batching --------
print("=" * 40)
print("Method 2: DataLoader Batching")
print("=" * 40)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

start = time.time()
for epoch in range(3):
    total_loss = 0
    for images, labels in train_loader:
        outputs = model2(images)
        loss = criterion(outputs, labels)
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

dataloader_time = time.time() - start
print(f"DataLoader Time: {dataloader_time:.2f}s\n")

# -------- Comparison --------
print("=" * 40)
print("COMPARISON")
print("=" * 40)
print(f"Manual Batching : {manual_time:.2f}s")
print(f"DataLoader      : {dataloader_time:.2f}s")
print(f"DataLoader is {'faster' if dataloader_time < manual_time else 'slower'} ✅")

torch.save(model2.state_dict(), 'model.pth')
print("\nModel saved!")