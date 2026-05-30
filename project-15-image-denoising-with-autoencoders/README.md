# Project 15 - Image Denoising with Autoencoders

## 👨‍💻 Author
**Nikhil Goyal**
- GitHub: [@goyalnik](https://github.com/goyalnik)
- LinkedIn: [Nikhil Goyal](https://www.linkedin.com/in/nikhil-goyal-30aa50124/)

## 📌 Description
Autoencoder that removes Gaussian noise from MNIST images.

## 🎯 Architecture
Encoder: 784 → 128 → 64
Decoder: 64 → 128 → 784

## 🛠️ Tech Stack
- Python 3.x
- PyTorch
- torchvision

## 🚀 How to Run
pip install -r requirements.txt
python train.py
python predict.py

## 📊 Expected Output
Epoch 1, Loss: 0.1823
...
Epoch 5, Loss: 0.0634