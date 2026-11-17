import torch
from torch import nn
from torchvision.models.segmentation import lraspp_mobilenet_v3_large


def build_model(num_classes: int = 3) -> nn.Module:
    return lraspp_mobilenet_v3_large(weights=None, num_classes=num_classes)


class LogitsOnly(nn.Module):
    """Return the "out" logits tensor instead of torchvision's dict, for export."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]
