# ============================================
# Author: Nikhil Goyal
# GitHub: https://github.com/goyalnik
# Project: 100 PyTorch Projects - Project 15
# ============================================

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(-1, 1, 28, 28)

model = Autoencoder()
model.load_state_dict(torch.load('model.pth'))
model.eval()

noisy_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.5 * torch.randn_like(x)),
    transforms.Lambda(lambda x: torch.clamp(x, 0., 1.))
])

test_data = datasets.MNIST(root='./data', train=False, download=True, transform=noisy_transform)
test_loader = DataLoader(test_data, batch_size=5)

images, _ = next(iter(test_loader))

with torch.no_grad():
    denoised = model(images)

print("Denoising complete!")
print(f"Input (noisy) shape: {images.shape}")
print(f"Output (clean) shape: {denoised.shape}")
print(f"Noisy pixel mean: {images.mean().item():.4f}")
print(f"Denoised pixel mean: {denoised.mean().item():.4f}")