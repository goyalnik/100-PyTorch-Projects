import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

device = torch.device("cpu")

# Class names
class_names = ["class1", "class2"]

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Load model
model = models.efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(class_names)
)

model.load_state_dict(
    torch.load(
        "efficientnet_custom.pth",
        map_location=device
    )
)

model.eval()

# Load image
image = Image.open("sample.jpg").convert("RGB")

image = transform(image)
image = image.unsqueeze(0)

with torch.no_grad():

    output = model(image)

    prediction = torch.argmax(output, dim=1)

print(
    "Predicted Class:",
    class_names[prediction.item()]
)