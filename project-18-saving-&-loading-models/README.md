# Project 18 - Saving & Loading Models

## 👨‍💻 Author
**Nikhil Goyal**
- GitHub: [@goyalnik](https://github.com/goyalnik)
- LinkedIn: [Nikhil Goyal](https://www.linkedin.com/in/nikhil-goyal-30aa50124/)

## 📌 Description
Demonstrates 3 ways to save and load PyTorch models.

## 🎯 Methods Covered
| Method | File | Use Case |
|--------|------|----------|
| state_dict | model_state.pth | Recommended — portable |
| Full model | model_full.pth | Quick save |
| Checkpoint | model_checkpoint.pth | Resume training |

## 🛠️ Tech Stack
- Python 3.x
- PyTorch

## 🚀 How to Run
pip install -r requirements.txt
python train.py
python predict.py

## 📊 Expected Output
Epoch 2, Loss: 1.1234
...
✅ state_dict saved
✅ Full model saved
✅ Checkpoint saved