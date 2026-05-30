# Project 14 - CNN on CIFAR-10

## 👨‍💻 Author
**Nikhil Goyal**
- GitHub: [@goyalnik](https://github.com/goyalnik)
- LinkedIn: [Nikhil Goyal](https://www.linkedin.com/in/nikhil-goyal-30aa50124/)

## 📌 Description
CNN trained on CIFAR-10 dataset to classify 10 categories of color images.

## 🎯 Classes
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## 🎯 Architecture
Input(3x32x32) → Conv(32) → Pool → Conv(64) → Pool → FC(256) → Output(10)

## 🛠️ Tech Stack
- Python 3.x
- PyTorch
- torchvision

## 🚀 How to Run
pip install -r requirements.txt
python train.py
python predict.py

## 📊 Expected Output
Epoch 1, Loss: 1.4821
...
Epoch 5, Loss: 0.9823
Test Accuracy: 65.45%