from pathlib import Path

import matplotlib.pyplot as plt
import torch

from tabletop_training.data import IMAGENET_MEAN, IMAGENET_STD, make_splits
from tabletop_training.model import build_model

TRAINING_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = TRAINING_DIR / "data"
CHECKPOINT_DIR = TRAINING_DIR / "checkpoints"

NUM_IMAGES = 6


def to_displayable(image: torch.Tensor) -> torch.Tensor:
    # Undo Normalize, then go from channels-first to height, width, channels.
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = image * std + mean
    return image.permute(1, 2, 0).clamp(0, 1)


def main():
    device = torch.device("mps")

    # Same seed as in train.py, so these images were never trained on.
    _, val_set = make_splits(DATA_DIR, size=256)

    model = build_model().to(device)
    best_path = CHECKPOINT_DIR / "best.pt"
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    fig, axes = plt.subplots(NUM_IMAGES, 3, figsize=(9, 3 * NUM_IMAGES))
    for row, index in enumerate(range(NUM_IMAGES)):
        image, mask = val_set[index]
        with torch.no_grad():
            logits = model(image.unsqueeze(0).to(device))["out"]
        pred = logits.argmax(dim=1).squeeze(0).cpu()

        # vmin/vmax fix the colour of each class, so both masks match.
        panels = [
            (to_displayable(image), {}, "image"),
            (mask, {"vmin": 0, "vmax": 2}, "true mask"),
            (pred, {"vmin": 0, "vmax": 2}, "predicted mask"),
        ]
        for ax, (data, kwargs, title) in zip(axes[row], panels):
            ax.imshow(data, **kwargs)
            ax.set_title(title)
            ax.axis("off")

    fig.tight_layout()
    out_path = DATA_DIR / "predictions.png"
    fig.savefig(out_path, dpi=100)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
