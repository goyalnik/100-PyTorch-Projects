# EfficientNet-B0 on Custom Dataset

This project demonstrates transfer learning using EfficientNet-B0 for image classification on a custom dataset.

## Features

* Pretrained EfficientNet-B0
* Transfer Learning
* Custom Image Classification
* PyTorch Implementation
* GPU Support

## Dataset Structure

custom_data/

├── train/

│ ├── class1/

│ ├── class2/

│

└── test/

├── class1/

├── class2/

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

## Prediction

```bash
python predict.py
```

## Requirements

* Python 3.9+
* PyTorch
* TorchVision

## Output

After training, the model will be saved as:

```text
efficientnet_custom.pth
```

## Author

Nikhil Goyal
