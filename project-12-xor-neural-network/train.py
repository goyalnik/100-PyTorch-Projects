import torch
import torch.nn as nn

# XOR Dataset
X = torch.tensor([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
])

Y = torch.tensor([
    [0.],
    [1.],
    [1.],
    [0.]
])

# Neural Network
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

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epochs = 5000

for epoch in range(epochs):
    outputs = model(X)

    loss = criterion(outputs, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "xor_model.pth")

print("\nModel saved as xor_model.pth")

with torch.no_grad():
    predictions = model(X)

    print("\nRaw Predictions:")
    print(predictions)

    print("\nRounded Predictions:")
    print(predictions.round())