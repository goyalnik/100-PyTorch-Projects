# Project 7 - Multi-layer Perceptron with Dropout

## 👨‍💻 Author
**Nikhil Goyal**
- GitHub: [@goyalnik](https://github.com/goyalnik)
- LinkedIn: [Nikhil Goyal](https://www.linkedin.com/in/nikhil-goyal-30aa50124/)

## 📌 Description
MLP with dropout regularization to classify FashionMNIST clothing items.

## 🎯 Architecture
Input(784) → FC(256) → Dropout(0.5) → FC(128) → Dropout(0.5) → Output(10)

## 🛠️ Tech Stack
- Python 3.x
- PyTorch
- torchvision

## 🚀 How to Run
pip install -r requirements.txt
python train.py
python predict.py

## 📊 Expected Output
Epoch 5, Loss: 0.3821
Test Accuracy with Dropout: 87.45%