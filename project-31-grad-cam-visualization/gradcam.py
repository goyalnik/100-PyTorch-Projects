import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_backward_hook(self.backward_hook)
    
    def forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, class_idx=None):
        """Generate Grad-CAM heatmap"""
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get class index if not provided
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        target_score = output[0, class_idx]
        target_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        # Calculate weights (average gradients across spatial dimensions)
        weights = gradients.mean(dim=[2, 3], keepdim=True)
        
        # Generate CAM
        cam = (weights * activations).sum(dim=1).squeeze()
        
        # Normalize CAM
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, class_idx
    
    def overlay_heatmap(self, original_img, cam, alpha=0.5):
        """Overlay heatmap on original image"""
        
        # Resize CAM to match image size
        cam = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
        
        # Convert CAM to heatmap
        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert original image to BGR if needed
        if len(original_img.shape) == 3 and original_img.shape[2] == 3:
            original_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        else:
            original_bgr = original_img
        
        # Overlay
        overlay = cv2.addWeighted(original_bgr, 1 - alpha, heatmap, alpha, 0)
        
        return overlay, heatmap

def load_image(image_path, size=(224, 224)):
    """Load and preprocess image"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size, Image.LANCZOS)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    tensor = transform(img).unsqueeze(0)
    
    return tensor, np.array(img)