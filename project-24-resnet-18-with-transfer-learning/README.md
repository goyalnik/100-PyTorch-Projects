# Project 24 - ResNet-18 with Transfer Learning

## 👨‍💻 Author
**Nikhil Goyal**
- GitHub: [@goyalnik](https://github.com/goyalnik)
- LinkedIn: [Nikhil Goyal](https://www.linkedin.com/in/nikhil-goyal-30aa50124/)

## 📌 Description
ResNet-18 pretrained on ImageNet, fine-tuned on CIFAR-10 using Transfer Learning.

## 🎯 Transfer Learning Strategy
- Freeze all layers except final FC layer
- Only train the last layer (10 classes)
- Uses ImageNet pretrained weights

## 🛠️ Tech Stack
- Python 3.x
- PyTorch
- torchvision

## 🚀 How to Run
pip install -r requirements.txt
python train.py
python predict.py

## 📊 Expected Output
Epoch 3, Loss: 0.8234
Test Accuracy: 72.45%