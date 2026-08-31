#!/usr/bin/env python3
"""Correctness gate for the experimental Apple Silicon Core ML vision encoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import onnxruntime as ort
from PIL import Image

MAXIMUM_ABSOLUTE_ERROR = 1e-3
MINIMUM_COSINE = 0.99999
THRESHOLD = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-onnx", required=True, type=Path)
    parser.add_argument("--textual-onnx", required=True, type=Path)
    parser.add_argument("--coreml-package", required=True, type=Path)
    parser.add_argument("--rclip-source", required=True, type=Path)
    parser.add_argument("--golden", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def normalize(values: np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.float32]]:
    values = np.asarray(values, dtype=np.float32)
    return values / np.linalg.norm(values, axis=-1, keepdims=True)


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


def rankings(scores: np.ndarray[Any, Any], paths: list[str]) -> list[list[str]]:
    return [
        [path for _, path in sorted(zip(row.tolist(), paths), key=lambda item: (-item[0], item[1]))]
        for row in scores
    ]


def decisions(scores: np.ndarray[Any, Any]) -> list[list[bool]]:
    return (scores >= THRESHOLD).tolist()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.rclip_source.resolve()))
    from rclip.utils.preprocess import preprocess

    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    pixels = np.stack([preprocess(image) for image in generated_images()])
    visual_session = ort.InferenceSession(
        str(args.visual_onnx), providers=["CPUExecutionProvider"]
    )
    (onnx_images,) = visual_session.run(None, {"input": pixels})
    onnx_images = normalize(onnx_images)

    coreml_model = ct.models.MLModel(str(args.coreml_package))
    padded = np.concatenate([pixels, np.repeat(pixels[-1:], 8 - len(pixels), axis=0)])
    first_coreml = np.asarray(coreml_model.predict({"input": padded})["output"], dtype=np.float32)[
        : len(pixels)
    ]
    second_coreml = np.asarray(coreml_model.predict({"input": padded})["output"], dtype=np.float32)[
        : len(pixels)
    ]
    first_coreml = normalize(first_coreml)
    second_coreml = normalize(second_coreml)

    tokens = np.asarray([query["token_ids"] for query in golden["queries"]], dtype=np.int64)
    text_session = ort.InferenceSession(
        str(args.textual_onnx), providers=["CPUExecutionProvider"]
    )
    (text_embeddings,) = text_session.run(None, {"input": tokens})
    text_embeddings = normalize(text_embeddings)
    onnx_scores = text_embeddings @ onnx_images.T
    first_scores = text_embeddings @ first_coreml.T
    second_scores = text_embeddings @ second_coreml.T
    paths = [image["name"] for image in golden["images"]]

    cosines = np.sum(onnx_images * first_coreml, axis=1)
    maximum_error = float(np.max(np.abs(onnx_images - first_coreml)))
    minimum_cosine = float(cosines.min())
    first_rankings = rankings(first_scores, paths)
    report = {
        "schema_version": 1,
        "maximum_absolute_error": maximum_error,
        "minimum_cosine": minimum_cosine,
        "tolerances": {
            "maximum_absolute_error": MAXIMUM_ABSOLUTE_ERROR,
            "minimum_cosine": MINIMUM_COSINE,
        },
        "top_k_paths_and_order_exact": first_rankings == rankings(onnx_scores, paths),
        "threshold_decisions_exact": decisions(first_scores) == decisions(onnx_scores),
        "repeat_rankings_exact": first_rankings == rankings(second_scores, paths),
        "repeat_threshold_decisions_exact": decisions(first_scores) == decisions(second_scores),
    }
    report["passed"] = (
        maximum_error <= MAXIMUM_ABSOLUTE_ERROR
        and minimum_cosine >= MINIMUM_COSINE
        and report["top_k_paths_and_order_exact"]
        and report["threshold_decisions_exact"]
        and report["repeat_rankings_exact"]
        and report["repeat_threshold_decisions_exact"]
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
