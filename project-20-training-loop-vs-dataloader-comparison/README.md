# Project 20 - Training Loop vs DataLoader Comparison

## 👨‍💻 Author
**Nikhil Goyal**
- GitHub: [@goyalnik](https://github.com/goyalnik)
- LinkedIn: [Nikhil Goyal](https://www.linkedin.com/in/nikhil-goyal-30aa50124/)

## 📌 Description
Compares manual batching vs PyTorch DataLoader for training efficiency.

## 🎯 Key Difference
| Feature | Manual Batching | DataLoader |
|---------|----------------|------------|
| Shuffling | Manual | Automatic |
| Speed | Slower | Faster |
| Code | Complex | Clean |
| Multiprocessing | No | Yes |

## 🛠️ Tech Stack
- Python 3.x
- PyTorch
- torchvision

## 🚀 How to Run
pip install -r requirements.txt
python train.py
python predict.py

## 📊 Expected Output
Manual Batching Time: 45.23s
DataLoader Time: 38.12s
DataLoader is faster ✅