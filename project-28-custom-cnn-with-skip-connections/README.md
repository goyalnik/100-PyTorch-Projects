# CNN with Skip Connections - CIFAR-10

Custom CNN architecture with skip connections trained on CIFAR-10 dataset.

## Features

- Skip connections (residual connections)
- CIFAR-10 image classification
- PyTorch implementation
- Training and prediction scripts

## Installation

```bash
pip install torch torchvision pillow numpy
```

## Usage

### Training

```bash
python train.py
```

This will:
- Download CIFAR-10 dataset
- Train for 20 epochs
- Save model as `model.pth`

### Prediction

```bash
python predict.py --image path/to/image.jpg
```

Or with custom model:
```bash
python predict.py --image path/to/image.jpg --model path/to/model.pth
```

## Model Architecture

- Conv layer: 3 → 32 channels
- Conv layer: 32 → 32 channels (with skip connection)
- Pool layer: MaxPool2d
- Conv layer: 32 → 64 channels
- FC layers: 64*16*16 → 128 → 10

## Classes

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Results

Expected accuracy: ~70-75% on test set

## Files

- `train.py` - Training script
- `predict.py` - Prediction script
- `model.pth` - Trained model (generated after training)
- `.gitignore` - Git configuration
- `README.md` - This file