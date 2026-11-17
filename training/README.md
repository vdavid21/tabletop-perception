# Training

Trains a pet segmentation model (LR-ASPP MobileNetV3-Large) on the
Oxford-IIIT Pet dataset. Each pixel is labelled pet, background or border.

## Setup

```sh
uv sync
```

The dataset downloads into `data/` the first time a script runs.

## Scripts

Run from this folder with `uv run python scripts/<name>.py`.

| Script         | What it does |
|----------------|--------------|
| `explore_data` | Print the dataset size and the mask values of one sample; save it as `data/sample_55.png` |
| `overfit`      | Fit 20 images as a sanity check; loss should approach zero |
| `train`        | Train for 15 epochs; saves the best model to `checkpoints/best.pt` |
| `visualize`    | Save image / true mask / prediction for 6 validation images to `data/predictions.png` |
| `test`         | Report per-class IoU and Dice on the test set |
| `export_onnx`  | Export `checkpoints/best.pt` to `exports/pet_lraspp_256.onnx` (plus its `.onnx.data` weights file; keep both together) |
| `check_onnx`   | Compare PyTorch and ONNX Runtime on 20 validation images: largest logit difference and share of matching pixels |

## Results

Test set, 3,669 images, 256×256:

| Class      | IoU   | Dice  |
|------------|-------|-------|
| pet        | 0.874 | 0.933 |
| background | 0.935 | 0.966 |
| border     | 0.568 | 0.724 |
| mean       | 0.792 | 0.875 |

ONNX export vs PyTorch, CPU, 20 validation images:

| Check                             | Result                           |
|-----------------------------------|----------------------------------|
| Largest absolute logit difference | 3.1e-05                          |
| Pixels with the same class        | 99.9999% (1 of 1,310,720 differ) |

The differences are float32 rounding, so the export matches the trained model.
