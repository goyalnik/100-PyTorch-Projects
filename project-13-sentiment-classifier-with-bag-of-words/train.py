# ============================================
# Author: Nikhil Goyal
# GitHub: https://github.com/goyalnik
# Project: 100 PyTorch Projects - Project 13
# ============================================

import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle

# Sample dataset
texts = [
    "I loved this movie", "This film was fantastic", "What a great experience",
    "Amazing performances by the cast", "Highly recommend this film",
    "I hated this movie", "This film was terrible", "What a boring experience",
    "Worst movie I have ever seen", "Complete waste of time"
]
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

# Convert text to Bag-of-Words vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()
Y = torch.tensor(labels).float().unsqueeze(1)
X = torch.tensor(X).float()

# Save vectorizer for predict.py
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Split into train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

class SentimentBoW(nn.Module):
    def __init__(self, input_dim):
        super(SentimentBoW, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

model = SentimentBoW(input_dim=X.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    outputs = model(X_train)
    loss = criterion(outputs, Y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Test accuracy
with torch.no_grad():
    preds = model(X_test).round()
    correct = (preds == Y_test).sum().item()
    print(f"\nTest Accuracy: {correct}/{len(Y_test)}")

torch.save(model.state_dict(), 'model.pth')
print("Model saved!")