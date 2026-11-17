import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tabletop_training.data import make_splits
from tabletop_training.engine import evaluate, iou_per_class, train_one_epoch
from tabletop_training.model import build_model

TRAINING_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = TRAINING_DIR / "data"
CHECKPOINT_DIR = TRAINING_DIR / "checkpoints"

# Oxford-IIIT Pet trimap labels after the "- 1" in PetSegmentation.
CLASS_NAMES = ["pet", "background", "border"]


def main():
    device = torch.device("mps")

    train_set, val_set = make_splits(DATA_DIR, size=256)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True,
                              num_workers=4, drop_last=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False,
                            num_workers=4, persistent_workers=True)

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_path = CHECKPOINT_DIR / "best.pt"
    best_miou = 0.0

    num_epochs = 15
    for epoch in range(1, num_epochs + 1):
        start = time.perf_counter()
        loss = train_one_epoch(model, train_loader, optimizer, device)
        confusion = evaluate(model, val_loader, device)
        iou = iou_per_class(confusion)
        miou = iou.mean().item()
        seconds = time.perf_counter() - start

        per_class = "  ".join(f"{name} {value:.3f}"
                              for name, value in zip(CLASS_NAMES, iou.tolist()))
        line = (f"epoch {epoch:2d}/{num_epochs}  loss {loss:.4f}  "
                f"mIoU {miou:.3f}  [{per_class}]  {seconds:.0f}s")

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), best_path)
            line += "  * saved"
        print(line)

    print(f"best mIoU {best_miou:.3f}, weights in {best_path}")


# Required with num_workers > 0 on macOS: worker processes re-import this file.
if __name__ == "__main__":
    main()
