# predict Feature Maps in CNN

Extract and predict feature maps from intermediate layers in a CNN to understand what the model learns.

## What are Feature Maps?

Feature maps are the outputs of convolutional layers. They show:
- What patterns the network has learned
- Which parts of the image activate certain features
- How the network processes the input

## Features

- Train a CNN with multiple conv layers
- Extract feature maps from intermediate layers using hooks
- predict feature maps with heatmaps
- Supports multiple layers (conv1, conv2, conv3)

## Installation

```bash
pip install torch torchvision pillow numpy matplotlib
```

## Usage

### Training the Model

```bash
python train.py
```

This will:
- Download CIFAR-10 dataset
- Train CNN for 5 epochs
- Save model as `model_features.pth`

### Visualizing Feature Maps

```bash
# predict conv1 layer (default)
python predict.py

# predict conv2 layer
python predict.py --layer conv2

# predict conv3 layer
python predict.py --layer conv3

# predict specific number of features
python predict.py --layer conv1 --num-features 32

# predict with custom image
python predict.py --layer conv1 --image path/to/image.jpg
```

## Model Architecture

```
Input (3, 32, 32)
  ↓
Conv1 (3 → 16) + BatchNorm + ReLU + MaxPool
  ↓
Conv2 (16 → 32) + BatchNorm + ReLU + MaxPool
  ↓
Conv3 (32 → 64) + BatchNorm + ReLU + MaxPool
  ↓
FC (64*4*4 → 128 → 10)
```

## Feature Map Details

| Layer | Input | Output | Kernel | Activation |
|-------|-------|--------|--------|------------|
| conv1 | 32x32 | 32x32  | 3x3    | 16 filters |
| pool1 | 32x32 | 16x16  | 2x2    | - |
| conv2 | 16x16 | 16x16  | 3x3    | 32 filters |
| pool2 | 16x16 | 8x8    | 2x2    | - |
| conv3 | 8x8   | 8x8    | 3x3    | 64 filters |
| pool3 | 8x8   | 4x4    | 2x2    | - |

## How Hooks Work

Hooks are functions that capture intermediate outputs:

```python
def hook(module, input, output):
    features.append(output.detach())

model.conv1.register_forward_hook(hook)
```

## Visualization Examples

- **Conv1**: Detects edges, textures
- **Conv2**: Combines edges into shapes
- **Conv3**: Detects complex patterns

## Files

- `train.py` - Train the feature CNN
- `predict.py` - predict feature maps
- `model_features.pth` - Trained model
- `.gitignore` - Git configuration
- `README.md` - This file

## Output

Feature maps are saved as PNG images:
- `feature_maps_conv1.png`
- `feature_maps_conv2.png`
- `feature_maps_conv3.png`