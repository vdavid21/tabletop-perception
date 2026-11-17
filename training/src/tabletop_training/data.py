from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset
from torchvision import tv_tensors
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# One item: a normalised image (3, H, W) and its class-index mask (H, W).
Sample = tuple[torch.Tensor, torch.Tensor]


def build_transforms(size: int, train: bool) -> v2.Compose:
    steps: list[v2.Transform] = [v2.Resize((size, size))]
    if train:
        steps.append(v2.RandomHorizontalFlip(p=0.5))
    steps += [
        v2.ToDtype(
            {tv_tensors.Image: torch.float32,
             tv_tensors.Mask: torch.int64,
             "others": None},
            scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return v2.Compose(steps)

def make_splits(root: Path, size: int, val_fraction: float = 0.1,
                seed: int = 0) -> tuple[Subset[Sample], Subset[Sample]]:
    # Two views of the same "trainval" images, differing only in transforms.
    # A single Subset can't do this: Subsets share their parent's transforms.
    train_base = PetSegmentation(root, "trainval",
                                 build_transforms(size, train=True))
    val_base = PetSegmentation(root, "trainval",
                               build_transforms(size, train=False))

    n = len(train_base)
    n_val = int(round(n * val_fraction))
    if not 0 < n_val < n:
        raise ValueError(f"val_fraction={val_fraction} gives {n_val} of {n} "
                         "images for validation")

    # A local generator keeps the split deterministic for a given seed
    # without touching (or depending on) the global RNG state.
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()

    # One permutation, sliced into two disjoint parts, so no image is shared.
    val_indices = perm[:n_val]
    train_indices = perm[n_val:]
    return Subset(train_base, train_indices), Subset(val_base, val_indices)

class PetSegmentation(Dataset[Sample]):
    def __init__(self, root: Path, split: str, transforms: v2.Compose):
        self.base = OxfordIIITPet(root=root, split=split,
                                  target_types="segmentation", download=True)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Sample:
        image, mask = self.base[index]
        image = tv_tensors.Image(F.pil_to_tensor(image))
        mask = tv_tensors.Mask(F.pil_to_tensor(mask))
        image, mask = self.transforms(image, mask)
        return image, mask.squeeze(0) - 1
