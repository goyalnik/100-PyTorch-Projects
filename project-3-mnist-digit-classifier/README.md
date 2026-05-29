# Project 3 - MNIST Digit Classifier

## 📌 Description
Feedforward neural network to classify handwritten digits (0-9) from MNIST dataset.

## 🎯 Goal
Classify 28×28 grayscale images using:
- Input: 784 neurons
- Hidden: 128 neurons (ReLU)
- Output: 10 neurons (digits 0-9)

## 🛠️ Tech Stack
- Python 3.x
- PyTorch
- torchvision

## 🚀 How to Run
pip install -r requirements.txt
python train.py
python predict.py

## 📊 Expected Output
Epoch 1, Loss: 0.2341
...
Epoch 5, Loss: 0.0823
Test Accuracy: 97.85%