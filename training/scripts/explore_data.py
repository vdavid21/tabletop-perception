from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import OxfordIIITPet

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main():
    ds = OxfordIIITPet(root=DATA_DIR, split="trainval",
                       target_types="segmentation", download=True)
    print(f"samples: {len(ds)}")

    index = 55
    image, mask = ds[index]
    mask_np = np.array(mask)
    values, counts = np.unique(mask_np, return_counts=True)
    print(f"image size: {image.size}, mode: {image.mode}")
    print(f"mask size: {mask.size}, values: {values}, counts: {counts}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title("image")
    axes[1].imshow(mask_np)
    axes[1].set_title("mask")
    for ax in axes:
        ax.axis("off")
    fig.savefig(DATA_DIR / f"sample_{index}.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
