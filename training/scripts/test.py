from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tabletop_training.data import PetSegmentation, build_transforms
from tabletop_training.engine import dice_per_class, evaluate, iou_per_class
from tabletop_training.model import build_model

TRAINING_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = TRAINING_DIR / "data"
CHECKPOINT_DIR = TRAINING_DIR / "checkpoints"

CLASS_NAMES = ["pet", "background", "border"]


def main():
    device = torch.device("mps")

    model = build_model().to(device)
    best_path = CHECKPOINT_DIR / "best.pt"
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    # The dataset's own test part: never used for training or for picking
    # the best checkpoint.
    test_set = PetSegmentation(DATA_DIR, "test",
                               build_transforms(256, train=False))
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False,
                             num_workers=4)

    confusion = evaluate(model, test_loader, device)
    iou = iou_per_class(confusion)
    dice = dice_per_class(confusion)

    print(f"test images: {len(test_set)}")
    print(f"{'class':<12}{'IoU':>8}{'Dice':>8}")
    for name, i, d in zip(CLASS_NAMES, iou.tolist(), dice.tolist()):
        print(f"{name:<12}{i:>8.3f}{d:>8.3f}")
    # nanmean skips a class that never appears and is never predicted.
    print(f"{'mean':<12}{iou.nanmean().item():>8.3f}"
          f"{dice.nanmean().item():>8.3f}")


if __name__ == "__main__":
    main()
