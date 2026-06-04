import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_backward_hook(self.backward_hook)
    
    def forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        target_score = output[0, class_idx]
        target_score.backward()
        
        gradients = self.gradients
        activations = self.activations
        
        weights = gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * activations).sum(dim=1).squeeze()
        
        cam = torch.clamp(cam, min=0)
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, class_idx
    
    def overlay_heatmap(self, original_img, cam, alpha=0.4):
        cam = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        if len(original_img.shape) == 3 and original_img.shape[2] == 3:
            original_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        else:
            original_bgr = original_img
        
        overlay = cv2.addWeighted(original_bgr, 1 - alpha, heatmap, alpha, 0)
        
        return overlay, heatmap

def load_model(model_path='model_gradcam.pth'):
    model = CNNModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_with_gradcam(image_path=None, model_path='model_gradcam.pth'):
    """Predict and visualize Grad-CAM"""
    
    model = load_model(model_path)
    
    if image_path:
        print(f"Loading image: {image_path}")
        from PIL import Image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224), Image.LANCZOS)
        original_img = np.array(img)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        input_tensor = transform(img).unsqueeze(0)
    else:
        print("Loading random image from CIFAR-10...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
        input_tensor, label = next(iter(test_loader))
        original_img = input_tensor[0].permute(1, 2, 0).numpy()
        original_img = np.uint8((original_img * 0.5 + 0.5) * 255)
    
    input_tensor = input_tensor.to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_idx = output.argmax(dim=1).item()
        confidence = probabilities[0, predicted_idx].item()
    
    print(f"\nPredicted Class: {CLASSES[predicted_idx]}")
    print(f"Confidence: {confidence*100:.2f}%\n")
    
    # Generate Grad-CAM
    print("Generating Grad-CAM...")
    target_layer = model.features[-2]
    gradcam = GradCAM(model, target_layer)
    
    cam, class_idx = gradcam.generate_cam(input_tensor, predicted_idx)
    overlay, heatmap = gradcam.overlay_heatmap(original_img, cam, alpha=0.4)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB) if len(original_img.shape) == 3 else original_img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(cam, cmap='hot')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.suptitle(f'Grad-CAM - Predicted: {CLASSES[class_idx]} ({confidence*100:.2f}%)')
    plt.tight_layout()
    plt.savefig('gradcam_prediction.png', dpi=150, bbox_inches='tight')
    print("Saved as 'gradcam_prediction.png'")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict with Grad-CAM Visualization')
    parser.add_argument('--image', type=str, default=None, help='Image path (optional)')
    parser.add_argument('--model', type=str, default='model_gradcam.pth', help='Model path')
    
    args = parser.parse_args()
    
    predict_with_gradcam(args.image, args.model)