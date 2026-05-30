import torch
import torch.nn as nn

class XORNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

model = XORNet()

model.load_state_dict(
    torch.load("xor_model.pth")
)

model.eval()

X = torch.tensor([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
])

with torch.no_grad():
    predictions = model(X)

print("Predictions:")
print(predictions.round())