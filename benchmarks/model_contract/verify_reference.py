#!/usr/bin/env python3
"""Verify pinned DataComp ONNX artifacts against the original OpenCLIP reference.

This development-only harness deliberately accepts local artifact paths. Downloading and caching
are handled by the surrounding workflow so checksums can be verified before any model is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch
from PIL import Image

OPENCLIP_REVISION = "4afec35ffe57a943d569ff7ee888061830164da8"
OPENCLIP_WEIGHTS_SHA256 = "92c26d60d3200ed5ed040dff31a8d19f8140648da8007216c25744c478deef27"
RCLIP_MODEL_REVISION = "17b9d07433aad73f70d338d8a1c7a4cef83887e0"
VISUAL_ONNX_SHA256 = "3f7e6f94e5a34bc7ee8aba84aec0f963f56974ab405fbcd334c8e1c3f832bd2c"
TEXTUAL_ONNX_SHA256 = "ee267cd64f0f77362670ae0140476ed51ee8c5a761d41636e09997f2fdddcacc"
TOKENIZER_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
MODEL_NAME = "ViT-B-32-256"
COSINE_TOLERANCE = 0.99999
MAX_ABSOLUTE_TOLERANCE = 1e-4
PREPROCESS_MAX_ABSOLUTE_TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openclip-weights", required=True, type=Path)
    parser.add_argument("--visual-onnx", required=True, type=Path)
    parser.add_argument("--textual-onnx", required=True, type=Path)
    parser.add_argument("--tokenizer-vocab", required=True, type=Path)
    parser.add_argument(
        "--rclip-source",
        required=True,
        type=Path,
        help="checkout of rclip v3.3.0 at commit 3dcec2de5e23311473f6fb6433e602aa4f4ca812",
    )
    parser.add_argument("--golden-output", type=Path)
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="generate reference fixtures without loading ONNX Runtime",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"checksum mismatch for {path}: expected {expected}, got {actual}")


def normalize_rows(values: np.ndarray[Any, np.dtype[np.float32]]) -> np.ndarray[Any, np.dtype[np.float32]]:
    values = np.asarray(values, dtype=np.float32)
    return values / np.linalg.norm(values, axis=-1, keepdims=True)


def comparison(reference: np.ndarray[Any, Any], candidate: np.ndarray[Any, Any]) -> dict[str, float]:
    reference = normalize_rows(reference)
    candidate = normalize_rows(candidate)
    cosine = np.sum(reference * candidate, axis=1)
    return {
        "minimum_cosine": float(cosine.min()),
        "maximum_absolute_error": float(np.max(np.abs(reference - candidate))),
    }


def require_close(label: str, metrics: dict[str, float]) -> None:
    if metrics["minimum_cosine"] < COSINE_TOLERANCE:
        raise RuntimeError(f"{label} minimum cosine failed: {metrics}")
    if metrics["maximum_absolute_error"] > MAX_ABSOLUTE_TOLERANCE:
        raise RuntimeError(f"{label} maximum absolute error failed: {metrics}")


def generated_images() -> list[Image.Image]:
    images = []
    for seed, (width, height) in enumerate([(320, 240), (240, 320), (256, 256)]):
        y, x = np.indices((height, width), dtype=np.uint32)
        pixels = np.stack(
            [
                (x + 17 * seed) % 256,
                (y * 3 + 29 * seed) % 256,
                (x + y * 2 + 43 * seed) % 256,
            ],
            axis=-1,
        ).astype(np.uint8)
        images.append(Image.fromarray(pixels, mode="RGB"))
    return images


def top_k(scores: np.ndarray[Any, Any], paths: list[str], count: int) -> list[list[str]]:
    return [
        [path for _, path in sorted(zip(row.tolist(), paths), key=lambda item: (-item[0], item[1]))[:count]]
        for row in scores
    ]


def main() -> None:
    args = parse_args()
    require_sha256(args.openclip_weights, OPENCLIP_WEIGHTS_SHA256)
    require_sha256(args.visual_onnx, VISUAL_ONNX_SHA256)
    require_sha256(args.textual_onnx, TEXTUAL_ONNX_SHA256)
    require_sha256(args.tokenizer_vocab, TOKENIZER_SHA256)

    sys.path.insert(0, str(args.rclip_source))
    from rclip.utils.preprocess import preprocess as rclip_preprocess
    from rclip.utils.tokenizer import SimpleTokenizer

    model, _, reference_preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=str(args.openclip_weights)
    )
    model.eval()
    queries = [
        "a red car beside a blue bicycle",
        "three dogs running on grass",
        "a screenshot containing a settings dialog",
        "東京の夜景",
    ]
    reference_tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    reference_tokens = np.asarray(reference_tokenizer(queries).numpy(), dtype=np.int64)
    rclip_tokens = np.asarray(SimpleTokenizer(bpe_path=str(args.tokenizer_vocab))(queries), dtype=np.int64)
    if not np.array_equal(reference_tokens, rclip_tokens):
        raise RuntimeError("rclip and OpenCLIP token IDs differ")

    images = generated_images()
    reference_pixels = np.stack([np.asarray(reference_preprocess(image), dtype=np.float32) for image in images])
    rclip_pixels = np.stack([rclip_preprocess(image) for image in images])
    preprocess_max_absolute_error = float(np.max(np.abs(reference_pixels - rclip_pixels)))
    if preprocess_max_absolute_error > PREPROCESS_MAX_ABSOLUTE_TOLERANCE:
        raise RuntimeError(
            "rclip preprocessing differs from OpenCLIP: "
            f"maximum absolute error {preprocess_max_absolute_error}"
        )

    with torch.no_grad():
        reference_text = np.asarray(model.encode_text(torch.from_numpy(reference_tokens)).numpy(), dtype=np.float32)
        reference_images = np.asarray(model.visual(torch.from_numpy(reference_pixels)).numpy(), dtype=np.float32)

    paths = ["generated-landscape.png", "generated-portrait.png", "generated-square.png"]
    text_metrics = None
    image_metrics = None
    score_max_absolute_error = None
    rankings_exact = None
    if not args.reference_only:
        import onnxruntime as ort

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        text_session = ort.InferenceSession(
            str(args.textual_onnx), sess_options=session_options, providers=["CPUExecutionProvider"]
        )
        visual_session = ort.InferenceSession(
            str(args.visual_onnx), sess_options=session_options, providers=["CPUExecutionProvider"]
        )
        (onnx_text,) = text_session.run(None, {"input": reference_tokens})
        (onnx_images,) = visual_session.run(None, {"input": reference_pixels})

        text_metrics = comparison(reference_text, onnx_text)
        image_metrics = comparison(reference_images, onnx_images)
        require_close("text", text_metrics)
        require_close("image", image_metrics)

        reference_scores = normalize_rows(reference_text) @ normalize_rows(reference_images).T
        onnx_scores = normalize_rows(onnx_text) @ normalize_rows(onnx_images).T
        score_max_absolute_error = float(np.max(np.abs(reference_scores - onnx_scores)))
        reference_rankings = top_k(reference_scores, paths, len(paths))
        onnx_rankings = top_k(onnx_scores, paths, len(paths))
        rankings_exact = reference_rankings == onnx_rankings
        if not rankings_exact:
            raise RuntimeError("ONNX and OpenCLIP rankings differ")

    report = {
        "schema_version": 1,
        "passed": not args.reference_only,
        "reference_only": args.reference_only,
        "contracts": {
            "openclip_revision": OPENCLIP_REVISION,
            "openclip_weights_sha256": OPENCLIP_WEIGHTS_SHA256,
            "rclip_model_revision": RCLIP_MODEL_REVISION,
            "visual_onnx_sha256": VISUAL_ONNX_SHA256,
            "textual_onnx_sha256": TEXTUAL_ONNX_SHA256,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "model_name": MODEL_NAME,
            "image_size": 256,
            "context_length": 77,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "open_clip_torch": importlib.metadata.version("open-clip-torch"),
            "onnxruntime": importlib.metadata.version("onnxruntime"),
            "numpy": np.__version__,
            "execution_provider": "CPUExecutionProvider",
        },
        "tolerances": {
            "minimum_cosine": COSINE_TOLERANCE,
            "maximum_absolute_error": MAX_ABSOLUTE_TOLERANCE,
            "preprocess_maximum_absolute_error": PREPROCESS_MAX_ABSOLUTE_TOLERANCE,
        },
        "results": {
            "token_ids_exact": True,
            "preprocess_maximum_absolute_error": preprocess_max_absolute_error,
            "text": text_metrics,
            "image": image_metrics,
            "score_maximum_absolute_error": score_max_absolute_error,
            "rankings_exact": rankings_exact,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if args.golden_output is not None:
        golden = {
            "schema_version": 1,
            "contract": report["contracts"],
            "reference": "open_clip_torch 3.3.0 CPU, pinned safetensors",
            "queries": [
                {
                    "query": query,
                    "token_ids": tokens.tolist(),
                    "embedding_le_hex": normalize_rows(reference_text)[index].tobytes().hex(),
                }
                for index, (query, tokens) in enumerate(zip(queries, reference_tokens))
            ],
            "images": [
                {
                    "name": paths[index],
                    "width": image.width,
                    "height": image.height,
                    "seed": index,
                    "embedding_le_hex": normalize_rows(reference_images)[index].tobytes().hex(),
                }
                for index, image in enumerate(images)
            ],
        }
        args.golden_output.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
