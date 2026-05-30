# Project 10 - Weight Initialization

## 👨‍💻 Author
**Nikhil Goyal**
- GitHub: [@goyalnik](https://github.com/goyalnik)
- LinkedIn: [Nikhil Goyal](https://www.linkedin.com/in/nikhil-goyal-30aa50124/)

## 📌 Description
Demonstrates different weight initialization techniques in PyTorch MLP.

## 🎯 Initialization Techniques Used
| Layer | Method |
|-------|--------|
| fc1 | Xavier Uniform |
| fc2 | He (Kaiming Normal) |
| out | Constant (0.01) |
| biases | Zero Initialization |

## 🛠️ Tech Stack
- Python 3.x
- PyTorch
- torchvision

## 🚀 How to Run
pip install -r requirements.txt
python train.py
python predict.py

## 📊 Expected Output
Epoch 1, Loss: 0.4123
...
Epoch 5, Loss: 0.2341
Test Accuracy: 87.65%