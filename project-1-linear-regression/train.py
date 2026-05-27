import torch
import torch.nn as nn

# Training Data
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
Y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Model
model = nn.Linear(1, 1)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training Loop
epochs = 1000

for epoch in range(epochs):

    predictions = model(X)

    loss = criterion(predictions, Y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "model/linear_model.pth")

# Final Parameters
w, b = model.parameters()

print("\nTraining Completed!")
print(f"Weight: {w.item():.4f}")
print(f"Bias: {b.item():.4f}")

# Test prediction
test = torch.tensor([[5.0]])

prediction = model(test)

print(f"Prediction for x=5 : {prediction.item():.4f}")