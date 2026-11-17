from pathlib import Path

import torch

from tabletop_training.model import LogitsOnly, build_model

TRAINING_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = TRAINING_DIR / "checkpoints"
EXPORT_DIR = TRAINING_DIR / "exports"


def main():
    model = build_model()
    model.load_state_dict(
        torch.load(CHECKPOINT_DIR / "best.pt", map_location="cpu"))
    model = LogitsOnly(model).eval()

    EXPORT_DIR.mkdir(exist_ok=True)
    example = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        model, (example,), EXPORT_DIR / "pet_lraspp_256.onnx",
        input_names=["image"], output_names=["logits"],
        dynamo=True,
    )


if __name__ == "__main__":
    main()
