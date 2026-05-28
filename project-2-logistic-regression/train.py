import torch
import torch.nn as nn

# Example input features and binary labels (0 or 1)
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
Y = torch.tensor([[0.0], [0.0], [1.0], [1.0]])

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = LogisticRegression()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1000):
    y_pred = model(X)
    loss = criterion(y_pred, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

params = list(model.parameters())
print(f'Learned weight: {params[0].item():.4f}, bias: {params[1].item():.4f}')

# Save model
torch.save(model.state_dict(), 'model.pth')
print("Model saved!")