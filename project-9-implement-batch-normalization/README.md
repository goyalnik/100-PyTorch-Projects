# Project 9 - Implement Batch Normalization

## 👨‍💻 Author
**Nikhil Goyal**
- GitHub: [@goyalnik](https://github.com/goyalnik)
- LinkedIn: [Nikhil Goyal](https://www.linkedin.com/in/nikhil-goyal-30aa50124/)

## 📌 Description
MLP with BatchNorm1d layers to stabilize and speed up training on FashionMNIST.

## 🎯 Architecture
Input(784) → FC(256) → BatchNorm → ReLU → FC(128) → BatchNorm → ReLU → Output(10)

## 🛠️ Tech Stack
- Python 3.x
- PyTorch
- torchvision

## 🚀 How to Run
pip install -r requirements.txt
python train.py
python predict.py

## 📊 Expected Output
Epoch 1, Loss: 0.3201
...
Epoch 5, Loss: 0.1823
Test Accuracy: 88.92%