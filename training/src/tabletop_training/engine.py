import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)["out"]
        loss = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device, num_classes: int = 3) -> torch.Tensor:
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for images, masks in loader:
        logits = model(images.to(device))["out"]
        preds = logits.argmax(dim=1).cpu()
        index = masks.flatten() * num_classes + preds.flatten()
        confusion += torch.bincount(
            index, minlength=num_classes ** 2
        ).reshape(num_classes, num_classes)
    return confusion

def iou_per_class(confusion: torch.Tensor) -> torch.Tensor:
    confusion = confusion.float()
    overlap = confusion.diag()
    union = confusion.sum(dim=1) + confusion.sum(dim=0) - overlap
    # A class that is neither present nor predicted gives 0 / 0 = NaN.
    return overlap / union

def dice_per_class(confusion: torch.Tensor) -> torch.Tensor:
    confusion = confusion.float()
    overlap = confusion.diag()
    total = confusion.sum(dim=1) + confusion.sum(dim=0)
    # A class that is neither present nor predicted gives 0 / 0 = NaN.
    return 2 * overlap / total
