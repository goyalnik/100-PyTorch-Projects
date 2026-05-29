import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Generate synthetic data
X = torch.linspace(-3, 3, 100).unsqueeze(1)
Y = torch.sin(X) + 0.1 * torch.randn(X.size())

dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Custom Swish activation
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Neural network with Swish activation
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
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    for batch_X, batch_Y in loader:
        output = model(batch_X)
        loss = criterion(output, batch_Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 40 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'model.pth')
print("Model saved!")