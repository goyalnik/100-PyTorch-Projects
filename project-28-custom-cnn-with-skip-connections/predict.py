import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class SkipCNN(nn.Module):
    def __init__(self):
        super(SkipCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(out1))
        skip = out1 + out2
        out3 = self.pool(F.relu(self.conv3(skip)))
        flat = out3.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(flat))
        return self.fc2(x)

def load_model(model_path='model.pth'):
    model = SkipCNN().to(device)
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
    parser = argparse.ArgumentParser(description='Predict CIFAR-10 class')
    parser.add_argument('--image', type=str, required=True, help='Image path')
    parser.add_argument('--model', type=str, default='model.pth', help='Model path')
    
    args = parser.parse_args()
    
    model = load_model(args.model)
    class_name, confidence = predict(args.image, model)
    
    print(f"\nPredicted Class: {class_name}")
    print(f"Confidence: {confidence*100:.2f}%\n")