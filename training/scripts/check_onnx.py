import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from torchvision import tv_tensors

from tabletop_training.data import (IMAGENET_MEAN, IMAGENET_STD,
                                    PetSegmentation, build_transforms,
                                    make_splits)
from tabletop_training.model import LogitsOnly, build_model

TRAINING_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = TRAINING_DIR / "data"
CHECKPOINT_DIR = TRAINING_DIR / "checkpoints"
EXPORT_DIR = TRAINING_DIR / "exports"

ONNX_PATH = EXPORT_DIR / "pet_lraspp_256.onnx"
PARITY_PATH = EXPORT_DIR / "parity_reference.npz"
PREPROCESS_PATH = EXPORT_DIR / "preprocess_reference.npz"
MANIFEST_PATH = EXPORT_DIR / "parity_reference.json"

INPUT_SIZE = 256
NUM_IMAGES = 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(array: np.ndarray) -> dict:
    # Hash the raw C-ordered bytes, so the hash doesn't depend on how
    # np.savez happens to pack the file.
    data = np.ascontiguousarray(array)
    return {
        "sha256": hashlib.sha256(data.tobytes()).hexdigest(),
        "shape": list(data.shape),
        "dtype": str(data.dtype),
    }


def main():
    # Same weights and wrapper as export_onnx.py, on the CPU like the session.
    model = build_model()
    model.load_state_dict(
        torch.load(CHECKPOINT_DIR / "best.pt", map_location="cpu"))
    model = LogitsOnly(model).eval()

    session = onnxruntime.InferenceSession(
        ONNX_PATH, providers=["CPUExecutionProvider"])

    # Real validation images: realistic value ranges, unlike random noise.
    _, val_set = make_splits(DATA_DIR, size=INPUT_SIZE)
    pet_dataset = val_set.dataset
    assert isinstance(pet_dataset, PetSegmentation)
    eval_transforms = build_transforms(INPUT_SIZE, train=False)

    inputs, logits, raw_bgr, dataset_indices = [], [], [], []
    max_diff = 0.0
    same_pixels = 0
    total_pixels = 0
    for i in range(NUM_IMAGES):
        image, _ = val_set[i]
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
        print(f"image {i:2d}  max |diff| {diff:.2e}  "
              f"same class {same.mean():.4%}")

        # The original image at its original size, in BGR like the camera.
        index = val_set.indices[i]
        rgb = np.array(pet_dataset.base[index][0])
        bgr = np.ascontiguousarray(rgb[..., ::-1])

        # Self-check: BGR -> RGB -> eval transforms must give exactly the
        # stored input, or the two reference files would disagree.
        rebuilt = eval_transforms(tv_tensors.Image(
            torch.from_numpy(bgr[..., ::-1].copy()).permute(2, 0, 1)))
        if not torch.equal(rebuilt, image):
            raise RuntimeError(f"raw image {i} does not reproduce its input")

        inputs.append(x.numpy())
        logits.append(onnx_logits)
        raw_bgr.append(bgr)
        dataset_indices.append(index)

    print(f"\nover {NUM_IMAGES} images:")
    print(f"  largest absolute logit difference: {max_diff:.2e}")
    print(f"  pixels with the same class: {same_pixels / total_pixels:.4%} "
          f"({total_pixels - same_pixels} of {total_pixels} differ)")

    # Engine reference: preprocessed inputs and ONNX Runtime's logits.
    parity = {
        "inputs": np.concatenate(inputs).astype(np.float32),
        "logits": np.concatenate(logits).astype(np.float32),
    }
    np.savez(PARITY_PATH, allow_pickle=False, **parity)

    # Preprocessing reference: raw BGR images. Sizes differ, so one key each;
    # image_XX belongs to inputs[XX] in the parity reference.
    preprocess = {f"image_{i:02d}": img for i, img in enumerate(raw_bgr)}
    np.savez(PREPROCESS_PATH, allow_pickle=False, **preprocess)

    manifest = {
        "model": {
            "onnx": {"file": ONNX_PATH.name,
                     "sha256": sha256_file(ONNX_PATH)},
            "input_name": "image",
            "output_name": "logits",
            "classes": ["pet", "background", "border"],
        },
        "preprocessing": {
            "raw_colour_order": "BGR",
            "raw_layout": "HWC",
            "raw_dtype": "uint8",
            "model_colour_order": "RGB",
            "resize": {"size": [INPUT_SIZE, INPUT_SIZE],
                       "keep_aspect_ratio": False,
                       "interpolation": "bilinear",
                       "antialias": True},
            "scale": "divide by 255",
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "model_layout": "NCHW",
            "model_dtype": "float32",
        },
        "dataset": {"name": "Oxford-IIIT Pet", "split": "trainval",
                    "validation_indices": dataset_indices},
        "arrays": {
            PARITY_PATH.name: {k: describe(v) for k, v in parity.items()},
            PREPROCESS_PATH.name: {k: describe(v)
                                   for k, v in preprocess.items()},
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")

    for path in (PARITY_PATH, PREPROCESS_PATH, MANIFEST_PATH):
        print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
