from pathlib import Path

import numpy as np
import onnxruntime
import torch

from tabletop_training.data import make_splits
from tabletop_training.model import LogitsOnly, build_model

TRAINING_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = TRAINING_DIR / "data"
CHECKPOINT_DIR = TRAINING_DIR / "checkpoints"
EXPORT_DIR = TRAINING_DIR / "exports"

NUM_IMAGES = 20


def main():
    # Same weights and wrapper as export_onnx.py, on the CPU like the session.
    model = build_model()
    model.load_state_dict(
        torch.load(CHECKPOINT_DIR / "best.pt", map_location="cpu"))
    model = LogitsOnly(model).eval()

    session = onnxruntime.InferenceSession(
        EXPORT_DIR / "pet_lraspp_256.onnx",
        providers=["CPUExecutionProvider"])

    # Real validation images: realistic value ranges, unlike random noise.
    _, val_set = make_splits(DATA_DIR, size=256)

    max_diff = 0.0
    same_pixels = 0
    total_pixels = 0
    for index in range(NUM_IMAGES):
        image, _ = val_set[index]
        x = image.unsqueeze(0)

        with torch.no_grad():
            torch_logits = model(x).numpy()
        # np.asarray: run() is typed to return sparse tensors, lists or dicts
        # too; this model only returns a dense array.
        onnx_logits = np.asarray(session.run(None, {"image": x.numpy()})[0])

        diff = float(np.abs(torch_logits - onnx_logits).max())
        same = (torch_logits.argmax(axis=1) == onnx_logits.argmax(axis=1))
        max_diff = max(max_diff, diff)
        same_pixels += int(same.sum())
        total_pixels += same.size
        print(f"image {index:2d}  max |diff| {diff:.2e}  "
              f"same class {same.mean():.4%}")

    print(f"\nover {NUM_IMAGES} images:")
    print(f"  largest absolute logit difference: {max_diff:.2e}")
    print(f"  pixels with the same class: {same_pixels / total_pixels:.4%} "
          f"({total_pixels - same_pixels} of {total_pixels} differ)")


if __name__ == "__main__":
    main()
