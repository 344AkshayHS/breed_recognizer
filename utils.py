# utils.py
import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models

# --- DIRECTORY HELPERS ---
def ensure_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/heatmaps", exist_ok=True)

def load_image_pil(path):
    return Image.open(path).convert("RGB")

# --- MODEL HELPER ---
def get_model(num_classes, architecture="resnet50", pretrained=False):
    """
    Returns a model with the final layer replaced for num_classes.
    
    Args:
        num_classes (int): number of output classes
        architecture (str): "resnet18" or "resnet50"
        pretrained (bool): whether to use pretrained weights

    Returns:
        model: PyTorch model ready for training or inference
    """
    if architecture == "resnet18":
        model = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT)
    elif architecture == "resnet50":
        model = models.resnet50(weights=None if not pretrained else models.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    # Replace the final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
