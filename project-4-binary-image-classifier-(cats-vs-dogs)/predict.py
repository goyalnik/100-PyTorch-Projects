import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        return torch.sigmoid(self.fc1(x))

# Load model
model = CatDogCNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Predict
def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        prob = output.item()
        label = 'Dog' if prob > 0.5 else 'Cat'
        print(f'Image: {image_path} → {label} (Confidence: {prob:.2f})')

# Test karo
predict('./data/train/cat/cat1.jpg')
predict('./data/train/dog/dog1.jpg')