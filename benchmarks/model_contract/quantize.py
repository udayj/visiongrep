#!/usr/bin/env python3
"""Dynamically quantize the pinned DataComp CLIP ONNX models."""

# Reproduction environment: onnx==1.20.1, onnxruntime==1.27.0

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import tempfile
from pathlib import Path

from onnxruntime.quantization import QuantType, quant_pre_process, quantize_dynamic

VISUAL_SHA256 = "3f7e6f94e5a34bc7ee8aba84aec0f963f56974ab405fbcd334c8e1c3f832bd2c"
TEXTUAL_SHA256 = "ee267cd64f0f77362670ae0140476ed51ee8c5a761d41636e09997f2fdddcacc"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"checksum mismatch for {path}: expected {expected}, got {actual}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-onnx", required=True, type=Path)
    parser.add_argument("--textual-onnx", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    for package, expected in (("onnx", "1.20.1"), ("onnxruntime", "1.27.0")):
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise RuntimeError(f"{package} version must be {expected}, got {actual}")

    models = (
        (args.visual_onnx, VISUAL_SHA256, "visual-int8.onnx"),
        (args.textual_onnx, TEXTUAL_SHA256, "textual-int8.onnx"),
    )
    for source, expected, _ in models:
        require_sha256(source, expected)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as directory:
        preprocessed = Path(directory) / "preprocessed.onnx"
        for source, _, filename in models:
            destination = args.output_dir / filename
            # Symbolic shape inference fails on this model's dynamic Reshape nodes.
            quant_pre_process(source, preprocessed, skip_symbolic_shape=True)
            quantize_dynamic(
                preprocessed,
                destination,
                weight_type=QuantType.QInt8,
                per_channel=True,
                op_types_to_quantize=["MatMul", "Gemm"],
            )
            print(f"{destination}: {destination.stat().st_size} bytes, sha256 {sha256(destination)}")


if __name__ == "__main__":
    main()
