import torch
import torch.nn as nn

# Create model structure
model = nn.Linear(1, 1)

# Load saved weights
model.load_state_dict(torch.load("model/linear_model.pth"))

# Set evaluation mode
model.eval()

# Predict
x = torch.tensor([[10.0]])

prediction = model(x)

print(f"Prediction for x=10 : {prediction.item():.4f}")