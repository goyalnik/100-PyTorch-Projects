import torch
from PIL import Image
from torchvision import transforms

from train import EMNISTClassifier

model = EMNISTClassifier()
model.load_state_dict(torch.load("model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

image = Image.open("sample.png")
image = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(image)
    pred = output.argmax(1).item()

print("Predicted Class:", pred)