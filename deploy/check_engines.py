"""Compare TensorRT engines against the ONNX Runtime reference from the Mac."""
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorrt as trt

try:
    from cuda.bindings import runtime as cudart  # cuda-python 12.6 and later
except ImportError:
    from cuda import cudart  # older cuda-python

REPO_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_DIR / "models"
MANIFEST_PATH = MODELS_DIR / "parity_reference.json"
PARITY_PATH = MODELS_DIR / "parity_reference.npz"
PLANS = {
    "fp32": MODELS_DIR / "pet_lraspp_256_fp32.plan",
    "fp16": MODELS_DIR / "pet_lraspp_256_fp16.plan",
}
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def cuda_call(result):
    err, *values = result
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")
    return values[0] if values else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(array: np.ndarray) -> dict:
    data = np.ascontiguousarray(array)
    return {"sha256": hashlib.sha256(data.tobytes()).hexdigest(),
            "shape": list(data.shape), "dtype": str(data.dtype)}


def verify_reference(manifest: dict) -> None:
    # Proves the files arrived intact. It cannot prove the plans were built
    # from this ONNX file; the build logs record that.
    onnx_entry = manifest["model"]["onnx"]
    if sha256_file(MODELS_DIR / onnx_entry["file"]) != onnx_entry["sha256"]:
        raise RuntimeError(f"{onnx_entry['file']}: hash differs from manifest")
    for file_name, expected in manifest["arrays"].items():
        with np.load(MODELS_DIR / file_name, allow_pickle=False) as data:
            if sorted(data.files) != sorted(expected):
                raise RuntimeError(f"{file_name}: keys differ from manifest")
            for key, entry in expected.items():
                if describe(data[key]) != entry:
                    raise RuntimeError(f"{file_name}[{key}] differs from manifest")
    print("reference files match the manifest")


class Engine:
    """Minimal TensorRT runner: one fixed-shape input, one output."""

    def __init__(self, plan_path: Path, input_name: str, output_name: str):
        self.logger = TRT_LOGGER
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(plan_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"could not load {plan_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda_call(cudart.cudaStreamCreate())

        self.input_name, self.output_name = input_name, output_name
        self.shapes, self.dtypes, self.device = {}, {}, {}
        for name in (input_name, output_name):
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            nbytes = int(np.prod(shape)) * dtype.itemsize
            self.device[name] = cuda_call(cudart.cudaMalloc(nbytes))
            self.context.set_tensor_address(name, int(self.device[name]))
            self.shapes[name], self.dtypes[name] = shape, dtype

    def run(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=self.dtypes[self.input_name])
        if x.shape != self.shapes[self.input_name]:
            raise ValueError(f"expected {self.shapes[self.input_name]}, got {x.shape}")
        out = np.empty(self.shapes[self.output_name],
                       dtype=self.dtypes[self.output_name])
        h2d = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        d2h = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        cuda_call(cudart.cudaMemcpyAsync(self.device[self.input_name],
                                         x.ctypes.data, x.nbytes, h2d, self.stream))
        if not self.context.execute_async_v3(int(self.stream)):
            raise RuntimeError("inference failed")
        cuda_call(cudart.cudaMemcpyAsync(out.ctypes.data,
                                         self.device[self.output_name],
                                         out.nbytes, d2h, self.stream))
        cuda_call(cudart.cudaStreamSynchronize(self.stream))
        return out

    def close(self) -> None:
        for ptr in self.device.values():
            cuda_call(cudart.cudaFree(ptr))
        cuda_call(cudart.cudaStreamDestroy(self.stream))


def boundary_mask(classes: np.ndarray) -> np.ndarray:
    """True where any of a pixel's 8 neighbours has a different class."""
    height, width = classes.shape[1:]
    padded = np.pad(classes, ((0, 0), (1, 1), (1, 1)), mode="edge")
    mask = np.zeros(classes.shape, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neighbour = padded[:, 1 + dy:1 + dy + height, 1 + dx:1 + dx + width]
            mask |= neighbour != classes
    return mask


def compare(name, engine, inputs, ref_logits, class_names) -> None:
    outputs = np.stack([engine.run(x[None])[0] for x in inputs])
    diff = np.abs(outputs - ref_logits)
    ref_classes = ref_logits.argmax(axis=1)
    changed = outputs.argmax(axis=1) != ref_classes
    n_changed = int(changed.sum())

    print(f"\n{name}")
    print(f"  largest |logit difference|: {diff.max():.2e} "
          f"(largest |reference logit|: {np.abs(ref_logits).max():.1f})")
    print(f"  same class: {1 - n_changed / changed.size:.4%} "
          f"({n_changed} of {changed.size} pixels differ)")
    if n_changed:
        on_boundary = boundary_mask(ref_classes)
        print(f"  changed pixels on a class boundary: "
              f"{on_boundary[changed].mean():.1%} "
              f"(boundary pixels are {on_boundary.mean():.1%} of all pixels)")
        counts = np.bincount(ref_classes[changed], minlength=len(class_names))
        print("  changed pixels by reference class: "
              + ", ".join(f"{c} {n}" for n, c in zip(class_names, counts)))


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    verify_reference(manifest)
    with np.load(PARITY_PATH, allow_pickle=False) as data:
        inputs, ref_logits = data["inputs"], data["logits"]

    model = manifest["model"]
    for name, plan in PLANS.items():
        engine = Engine(plan, model["input_name"], model["output_name"])
        try:
            compare(name, engine, inputs, ref_logits, model["classes"])
        finally:
            engine.close()


if __name__ == "__main__":
    main()
