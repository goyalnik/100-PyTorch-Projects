# Grad-CAM Visualization

Gradient-weighted Class Activation Mapping (Grad-CAM) to predict CNN decisions and understand which regions influence predictions.

## What is Grad-CAM?

Grad-CAM uses gradients of the class score with respect to feature maps to highlight important regions in an image:

1. Forward pass: Get feature activations
2. Backward pass: Compute gradients
3. Combine: Weight activations by gradients
4. predict: Overlay on original image

## Formula

```
CAM = ReLU(Σ (αc * Ac))

where:
- αc = Gradient weights (average gradient per channel)
- Ac = Feature activations
```

## Features

- Grad-CAM implementation using hooks
- Works with any CNN architecture
- Overlay visualization on original images
- Support for custom images or CIFAR-10 dataset
- Saves visualization as PNG

## Installation

```bash
pip install torch torchvision pillow numpy matplotlib opencv-python
```

## Usage

### Training the Model

```bash
python train.py
```

Trains CNN on CIFAR-10 for 10 epochs.

### Visualizing Grad-CAM

```bash
# Use random image from CIFAR-10
python predict.py

# Use custom image
python predict.py --image path/to/image.jpg

# Use custom model
python predict.py --image path/to/image.jpg --model path/to/model.pth
```

## Model Architecture

```
Input (3, 32, 32)
  ↓
Conv(3→32) + BN + ReLU + MaxPool
  ↓
Conv(32→64) + BN + ReLU + MaxPool
  ↓
Conv(64→128) + BN + ReLU + MaxPool
  ↓
FC (128*4*4 → 256 → 10)
```

## How Hooks Work

```python
class GradCAM:
    def forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
```

Hooks capture:
- **Forward hook**: Feature activations
- **Backward hook**: Gradient flow

## Output Files

- `gradcam_visualization.png` - Side-by-side visualization

Contains:
1. Original image
2. Grad-CAM heatmap
3. Overlay on original

## Classes

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Interpretation

- **Red regions**: High importance for prediction
- **Blue regions**: Low importance for prediction
- **Green regions**: Medium importance

## Files

- `train.py` - Train the CNN model
- `gradcam.py` - Grad-CAM implementation
- `predict.py` - Visualization script
- `model_gradcam.pth` - Trained model
- `.gitignore` - Git configuration
- `README.md` - This file

## Tips

1. Use high-resolution images for better visualization
2. Experiment with different alpha values for overlay
3. Try different target layers to see different features
4. Use Grad-CAM to debug misclassifications

## References

- [Grad-CAM Paper](https://arxiv.org/abs/1610.02055)
- [PyTorch Hooks](https://pytorch.org/docs/stable/generated/torch.nn.Module.register_forward_hook.html)