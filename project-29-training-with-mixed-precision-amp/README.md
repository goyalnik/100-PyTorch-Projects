# Training with Mixed Precision (AMP)

Automatic Mixed Precision (AMP) training using PyTorch to accelerate CNN training on CIFAR-10.

## What is Mixed Precision?

Mixed Precision uses both FP16 (float16) and FP32 (float32) data types:
- **FP16**: Faster computation, less memory
- **FP32**: Better precision and numerical stability

**Benefits:**
- Faster training speed (~1.5-2x)
- Lower memory usage (50% less)
- Better GPU utilization
- Minimal accuracy loss

## Features

- CNN with BatchNorm layers
- Automatic Mixed Precision (AMP) with GradScaler
- CIFAR-10 image classification
- Training and prediction scripts

## Installation

```bash
pip install torch torchvision pillow numpy
```

## Usage

### Training with AMP

```bash
python train.py
```

This will:
- Download CIFAR-10 dataset
- Train for 10 epochs with Mixed Precision
- Save model as `model_amp.pth`

### Prediction

```bash
python predict.py --image path/to/image.jpg
```

Or with custom model:
```bash
python predict.py --image path/to/image.jpg --model path/to/model_amp.pth
```

## Model Architecture

- Conv2d: 3 → 32
- BatchNorm2d + ReLU + MaxPool
- Conv2d: 32 → 64
- BatchNorm2d + ReLU + MaxPool
- Conv2d: 64 → 128
- BatchNorm2d + ReLU + MaxPool
- FC: 128*4*4 → 256 → 10

## AMP Implementation

```python
# Initialize GradScaler
scaler = torch.cuda.amp.GradScaler()

# Forward pass with autocast
with torch.cuda.amp.autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)

# Backward with scaled loss
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Classes

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Performance

- Speed: ~1.5-2x faster than standard training
- Memory: 50% less VRAM usage
- Expected Accuracy: ~75-78%

## Requirements

- GPU with CUDA support (AMP works best on GPUs)
- PyTorch 1.6+
- CUDA 11.0+

## Files

- `train.py` - Training script with AMP
- `predict.py` - Prediction script
- `model_amp.pth` - Trained model (generated after training)
- `.gitignore` - Git configuration
- `README.md` - This file