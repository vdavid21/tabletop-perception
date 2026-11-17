from torch import nn
from torchvision.models.segmentation import lraspp_mobilenet_v3_large


def build_model(num_classes: int = 3) -> nn.Module:
    return lraspp_mobilenet_v3_large(weights=None, num_classes=num_classes)
