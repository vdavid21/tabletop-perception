from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from tabletop_training.data import PetSegmentation, build_transforms
from tabletop_training.engine import train_one_epoch
from tabletop_training.model import build_model

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main():
    device = torch.device("mps")
    dataset = PetSegmentation(DATA_DIR, "trainval",
                              build_transforms(256, train=False))
    tiny = Subset(dataset, range(20))
    loader = DataLoader(tiny, batch_size=10, shuffle=True)

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    num_epochs = 100
    for epoch in range(1, num_epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, device)
        print(f"epoch {epoch:3d}/{num_epochs}  loss {loss:.4f}")


if __name__ == "__main__":
    main()
