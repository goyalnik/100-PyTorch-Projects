import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Load trained model
model = LogisticRegression()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# New data points to predict
X_new = torch.tensor([[1.5], [2.5], [3.5]])

with torch.no_grad():
    predictions = model(X_new)
    predicted_labels = (predictions >= 0.5).float()

for i, (prob, label) in enumerate(zip(predictions, predicted_labels)):
    print(f'Input: {X_new[i].item()}, Probability: {prob.item():.4f}, Label: {int(label.item())}')