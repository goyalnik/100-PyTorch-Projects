import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_model(model_path='model_amp.pth'):
    model = SmallCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img)
        probabilities = F.softmax(output, dim=1)
        predicted = torch.argmax(output, dim=1)
    
    return CLASSES[predicted.item()], probabilities[0][predicted].item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict CIFAR-10 class with AMP model')
    parser.add_argument('--image', type=str, required=True, help='Image path')
    parser.add_argument('--model', type=str, default='model_amp.pth', help='Model path')
    
    args = parser.parse_args()
    
    model = load_model(args.model)
    class_name, confidence = predict(args.image, model)
    
    print(f"\nPredicted Class: {class_name}")
    print(f"Confidence: {confidence*100:.2f}%\n")