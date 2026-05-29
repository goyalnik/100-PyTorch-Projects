import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Load model
model = DigitClassifier()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Test on sample images
transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=10)

images, labels = next(iter(test_loader))

with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

for i in range(10):
    print(f'Actual: {labels[i].item()}, Predicted: {predicted[i].item()}')