#!/usr/bin/env python3
"""Dynamically quantize the pinned DataComp CLIP vision model."""

# Reproduction environment: onnx==1.20.1, onnxruntime==1.27.0

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import re
import tempfile
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quant_pre_process, quantize_dynamic

VISUAL_SHA256 = "3f7e6f94e5a34bc7ee8aba84aec0f963f56974ab405fbcd334c8e1c3f832bd2c"
EXCLUSION_PATTERNS = (r"/visual/MatMul", r"/visual/transformer/resblocks\.0/.*")


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
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    for package, expected in (("onnx", "1.20.1"), ("onnxruntime", "1.27.0")):
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise RuntimeError(f"{package} version must be {expected}, got {actual}")

    require_sha256(args.visual_onnx, VISUAL_SHA256)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as directory:
        preprocessed = Path(directory) / "preprocessed.onnx"
        destination = args.output_dir / "visual-int8.onnx"
        # Symbolic shape inference fails on this model's dynamic Reshape nodes.
        quant_pre_process(args.visual_onnx, preprocessed, skip_symbolic_shape=True)
        nodes_to_exclude = set()
        quantized_candidates = set()
        for node in onnx.load(preprocessed).graph.node:
            if any(re.fullmatch(pattern, node.name) for pattern in EXCLUSION_PATTERNS):
                nodes_to_exclude.add(node.name)
                if node.op_type in ("MatMul", "Gemm"):
                    quantized_candidates.add(node.name)
                # ORT replaces Gemm with MatMul before applying exclusions.
                if node.op_type == "Gemm":
                    nodes_to_exclude.add(node.name + "_MatMul")
        expected = {
            "/visual/MatMul",
            "/visual/transformer/resblocks.0/attn/MatMul",
            "/visual/transformer/resblocks.0/attn/MatMul_1",
            "/visual/transformer/resblocks.0/attn/MatMul_2",
            "/visual/transformer/resblocks.0/attn/Gemm",
            "/visual/transformer/resblocks.0/mlp/c_fc/MatMul",
            "/visual/transformer/resblocks.0/mlp/c_proj/MatMul",
        }
        if quantized_candidates != expected:
            raise ValueError("preprocessed vision graph does not match the selected exclusion patterns")
        quantize_dynamic(
            preprocessed,
            destination,
            weight_type=QuantType.QInt8,
            per_channel=True,
            op_types_to_quantize=["MatMul", "Gemm"],
            nodes_to_exclude=sorted(nodes_to_exclude),
            extra_options={"MatMulConstBOnly": True},
        )
        integer_nodes = {
            node.name.removesuffix("_quant")
            for node in onnx.load(destination).graph.node
            if node.op_type == "MatMulInteger"
        }
        if integer_nodes.intersection(nodes_to_exclude):
            raise ValueError("quantized vision graph contains excluded integer nodes")
        print(f"{destination}: {destination.stat().st_size} bytes, sha256 {sha256(destination)}")


if __name__ == "__main__":
    main()
