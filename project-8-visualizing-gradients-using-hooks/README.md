# Project 8 - Visualizing Gradients using Hooks

## 👨‍💻 Author
**Nikhil Goyal**
- GitHub: [@goyalnik](https://github.com/goyalnik)
- LinkedIn: [Nikhil Goyal](https://www.linkedin.com/in/nikhil-goyal-30aa50124/)

## 📌 Description
Use PyTorch backward hooks to capture and visualize gradients flowing through layers during training.

## 🎯 Goal
- Register backward hooks on layers
- Inspect gradient shapes and values
- Understand gradient flow for debugging

## 🛠️ Tech Stack
- Python 3.x
- PyTorch

## 🚀 How to Run
pip install -r requirements.txt
python train.py
python predict.py

## 📊 Expected Output
--- Gradient for Linear ---
Grad Output Shape: torch.Size([10, 5])
Grad Output Mean: 0.003421
Epoch 1, Loss: 1.2341
...