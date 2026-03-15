import torch
import torch.nn as nn
import torchvision.models as models

class MyBreedModel(nn.Module):
    def __init__(self, num_classes):
        super(MyBreedModel, self).__init__()
        self.model = models.resnet50(weights=None)  # match training
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
