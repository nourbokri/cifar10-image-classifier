import torch.nn as nn
import torchvision.models as models


def get_model(num_classes: int = 10):
    """
    Returns a ResNet18 model adapted for CIFAR-10 (10 classes).
    Uses pretrained weights for a strong baseline.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
