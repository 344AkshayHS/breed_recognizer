# gradcam.py
# Simple Grad-CAM for PyTorch (works for classification models)
import torch
import torch.nn.functional as F
import numpy as np
import cv2 
from torchvision import transforms

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.gradients = None
        self.activations = None
        # register hooks
        def forward_hook(module, inp, out):
            self.activations = out.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, target_class=None):
        """
        input_tensor: torch tensor of shape [1,3,H,W], requires_grad=False
        target_class: int or None -> if None, uses predicted class
        returns: heatmap numpy (H,W) normalized 0..1
        """
        self.model.zero_grad()
        out = self.model(input_tensor)  # [1, nclass]
        if target_class is None:
            target_class = out.argmax(dim=1).item()
        score = out[0, target_class]
        score.backward(retain_graph=True)

        grads = self.gradients[0].cpu().numpy()          # C x H' x W'
        acts = self.activations[0].cpu().numpy()         # C x H' x W'
        weights = np.mean(grads, axis=(1,2))             # C
        gcam = np.zeros(acts.shape[1:], dtype=np.float32) # H' x W'
        for i, w in enumerate(weights):
            gcam += w * acts[i]
        gcam = np.maximum(gcam, 0)
        gcam = cv2.resize(gcam, (input_tensor.size(3), input_tensor.size(2)))
        gcam -= gcam.min()
        if gcam.max() != 0:
            gcam /= gcam.max()
        return gcam
