import torch
import torch.nn as nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SwishNet(nn.Module):
    def __init__(self):
        super(SwishNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            Swish(),
            nn.Linear(32, 32),
            Swish(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = SwishNet()
model.load_state_dict(torch.load('model.pth'))
model.eval()

test_inputs = torch.tensor([[-3.0], [-1.5], [0.0], [1.5], [3.0]])

with torch.no_grad():
    predictions = model(test_inputs)

for i, (x, y) in enumerate(zip(test_inputs, predictions)):
    print(f'Input: {x.item():.1f}, Predicted: {y.item():.4f}')