import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# CPU Only
device = torch.device("cpu")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Dataset
train_data = datasets.ImageFolder(
    root="./custom_data/train",
    transform=transform
)

test_data = datasets.ImageFolder(
    root="./custom_data/test",
    transform=transform
)

train_loader = DataLoader(
    train_data,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=16
)

# Model
weights = models.EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)

num_classes = len(train_data.classes)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

# Freeze feature extractor
for param in model.features.parameters():
    param.requires_grad = False

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)

epochs = 5

for epoch in range(epochs):

    model.train()
    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {running_loss/len(train_loader):.4f}"
    )

# Save model
torch.save(model.state_dict(), "efficientnet_custom.pth")

print("Model Saved!")

# Evaluation
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")