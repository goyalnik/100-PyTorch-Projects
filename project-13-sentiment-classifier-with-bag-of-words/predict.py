# ============================================
# Author: Nikhil Goyal
# GitHub: https://github.com/goyalnik
# Project: 100 PyTorch Projects - Project 13
# ============================================

import torch
import torch.nn as nn
import pickle

# Load vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

class SentimentBoW(nn.Module):
    def __init__(self, input_dim):
        super(SentimentBoW, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

# Load model
input_dim = len(vectorizer.vocabulary_)
model = SentimentBoW(input_dim=input_dim)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# New reviews to predict
new_reviews = [
    "This movie was absolutely wonderful",
    "I did not enjoy this film at all",
    "Great acting and amazing story",
    "Terrible plot and bad acting"
]

X_new = vectorizer.transform(new_reviews).toarray()
X_new = torch.tensor(X_new).float()

with torch.no_grad():
    predictions = model(X_new)

for review, pred in zip(new_reviews, predictions):
    sentiment = "Positive 😊" if pred.item() > 0.5 else "Negative 😞"
    print(f'Review: "{review}"')
    print(f'Sentiment: {sentiment} (Score: {pred.item():.4f})\n')