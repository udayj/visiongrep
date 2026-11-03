#!/usr/bin/env python3
"""Compare the pinned CLIP encoders with local dynamically quantized encoders.

The default mode uses checksum-named image objects from ``bench.py prepare``.
The COCO mode expects the official validation images and caption annotations.
Timings use the ONNX Runtime CPU execution provider and include inference only,
except for the separately reported session creation times.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from tokenizers import Tokenizer

BENCHMARKS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARKS))
import bench  # noqa: E402
import quality  # noqa: E402

RUNS = 20
WARMUP = 3
IMAGE_SIZE = 256
MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)[:, None, None]
STD = np.asarray([0.26862954, 0.2613026, 0.2757771], dtype=np.float32)[:, None, None]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-fp32", type=Path, required=True)
    parser.add_argument("--textual-fp32", type=Path, required=True)
    parser.add_argument("--visual-int8", type=Path, required=True)
    parser.add_argument("--textual-int8", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--image-objects", type=Path)
    parser.add_argument("--coco", type=Path, help="directory containing val2017/ and annotations/")
    args = parser.parse_args()
    if args.coco is None and (args.textual_int8 is None or args.image_objects is None):
        parser.error("--textual-int8 and --image-objects are required without --coco")
    return args


def verify(path, expected):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"checksum mismatch for {path}: expected {expected}, got {actual}")


def load_image(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        if width < height:
            size = (IMAGE_SIZE, height * IMAGE_SIZE // width)
        else:
            size = (width * IMAGE_SIZE // height, IMAGE_SIZE)
        image = image.resize(size, Image.Resampling.BICUBIC)
        left = round((size[0] - IMAGE_SIZE) / 2)
        top = round((size[1] - IMAGE_SIZE) / 2)
        image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
        pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
        return (pixels - MEAN) / STD


def normalize(values):
    values = np.asarray(values, dtype=np.float32)
    lengths = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.all(np.isfinite(lengths)) or np.any(lengths == 0):
        raise ValueError("model produced a zero or non-finite embedding")
    return values / lengths


def create_session(path):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def session_and_creation_time(path):
    samples = []
    for _ in range(RUNS):
        start = time.perf_counter()
        session = create_session(path)
        samples.append(time.perf_counter() - start)
        del session
    return create_session(path), statistics.median(samples)


def infer(session, values):
    return session.run(None, {session.get_inputs()[0].name: values})[0]


def latency(session, batches):
    for i in range(WARMUP):
        infer(session, batches[i % len(batches)])
    samples = []
    for i in range(RUNS):
        batch = batches[i % len(batches)]
        start = time.perf_counter()
        infer(session, batch)
        samples.append((time.perf_counter() - start) / len(batch))
    return statistics.median(samples)


def image_embeddings(session, images):
    embeddings = []
    for start in range(0, len(images), 8):
        batch = np.stack(images[start:start + 8])
        embeddings.append(infer(session, batch))
        print(f"\rimages: {min(start + 8, len(images))}/{len(images)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return normalize(np.concatenate(embeddings))


def text_embeddings(session, tokens):
    return normalize(np.concatenate([infer(session, token) for token in tokens]))


def retrieval_runs(queries, paths, image_vectors, text_vectors):
    scores = text_vectors @ image_vectors.T
    runs = []
    for query, row in zip(queries, scores):
        ranking = sorted(range(len(paths)), key=lambda i: (-float(row[i]), paths[i]))
        runs.append(query | {"results": [{"path": paths[i], "score": float(row[i])} for i in ranking]})
    return runs


def coco_corpus(directory):
    annotation_path = directory / "annotations" / "captions_val2017.json"
    annotations = json.loads(annotation_path.read_text())
    images = sorted(annotations["images"], key=lambda row: row["file_name"])
    if len(images) != 5000:
        raise ValueError(f"expected 5,000 COCO validation images, found {len(images)}")
    indexes = {image["id"]: index for index, image in enumerate(images)}
    if len(indexes) != len(images):
        raise ValueError("COCO image IDs are not unique")
    captions = annotations["annotations"]
    correct = np.asarray([indexes[caption["image_id"]] for caption in captions])
    if len(captions) < 25000 or len(set(correct)) != len(images):
        raise ValueError("COCO caption annotations are incomplete")
    paths = [directory / "val2017" / image["file_name"] for image in images]
    return paths, [caption["caption"] for caption in captions], correct


def coco_image_embeddings(session, paths):
    embeddings = []
    for start in range(0, len(paths), 8):
        batch = np.stack([load_image(path) for path in paths[start:start + 8]])
        embeddings.append(infer(session, batch))
    return normalize(np.concatenate(embeddings))


def coco_text_embeddings(session, tokens):
    embeddings = []
    for start in range(0, len(tokens), 32):
        embeddings.append(infer(session, tokens[start:start + 32]))
    return normalize(np.concatenate(embeddings))


def coco_retrieval(text, fp32_images, int8_images, correct):
    hits = np.zeros((2, 3), dtype=np.int64)
    overlap = 0
    score_differences = []
    for start in range(0, len(text), 256):
        stop = min(start + 256, len(text))
        fp32_scores = text[start:stop] @ fp32_images.T
        int8_scores = text[start:stop] @ int8_images.T
        # Images are ordered by filename, so stable sorting resolves ties consistently.
        fp32_top = np.argsort(-fp32_scores, axis=1, kind="stable")[:, :10]
        int8_top = np.argsort(-int8_scores, axis=1, kind="stable")[:, :10]
        targets = correct[start:stop, None]
        for model, ranking in enumerate((fp32_top, int8_top)):
            for column, k in enumerate((1, 5, 10)):
                hits[model, column] += np.count_nonzero(np.any(ranking[:, :k] == targets, axis=1))
        overlap += np.count_nonzero(fp32_top[:, :, None] == int8_top[:, None, :])
        rows = np.arange(stop - start)
        score_differences.append(fp32_scores[rows, targets[:, 0]] - int8_scores[rows, targets[:, 0]])
    differences = np.concatenate(score_differences).astype(np.float64)
    return hits / len(text), overlap / (len(text) * 10), differences.mean(), differences.std()


def evaluate_coco(args):
    paths, captions, correct = coco_corpus(args.coco)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.enable_truncation(max_length=77)
    tokenizer.enable_padding(length=77, pad_id=0, pad_token="<|endoftext|>")
    tokens = np.asarray([encoding.ids for encoding in tokenizer.encode_batch(captions)], dtype=np.int64)

    # Keep only the timing sample in memory; preprocessing all 5,000 images at once is large.
    sample = [load_image(path) for path in paths[:RUNS * 8]]
    batch1 = [np.stack(sample[i:i + 1]) for i in range(RUNS)]
    batch8 = [np.stack(sample[i:i + 8]) for i in range(0, RUNS * 8, 8)]
    results = {}
    for name, path in (("vision fp32", args.visual_fp32), ("vision int8", args.visual_int8),
                       ("text fp32", args.textual_fp32)):
        print(f"Evaluating {name}...", file=sys.stderr, flush=True)
        session, creation = session_and_creation_time(path)
        if name.startswith("vision"):
            times = (latency(session, batch1), latency(session, batch8))
            vectors = coco_image_embeddings(session, paths)
        else:
            times = (latency(session, [token[None, :] for token in tokens[:RUNS]]),)
            vectors = coco_text_embeddings(session, tokens)
        results[name] = {"creation": creation, "times": times, "vectors": vectors}
        del session

    recalls, overlap, score_mean, score_std = coco_retrieval(
        results["text fp32"]["vectors"], results["vision fp32"]["vectors"],
        results["vision int8"]["vectors"], correct,
    )
    print(f"COCO val2017 text-to-image: {len(paths)} images, {len(captions)} captions")
    print(f"{'Metric':32} {'fp32':>12} {'int8':>12} {'int8 - fp32':>12}")
    for column, k in enumerate((1, 5, 10)):
        fp32, int8 = recalls[:, column]
        print(f"{f'recall@{k}':32} {fp32:12.6f} {int8:12.6f} {int8 - fp32:12.6f}")
    print(f"Top-10 set overlap: {overlap:.6f}")
    print(f"Correct-image score fp32 - int8: mean {score_mean:.6f}, population stddev {score_std:.6f}")
    print("Timing (median of 20 runs, ms)      vision fp32   vision int8     text fp32")
    print(f"{'session creation':32}" + "".join(f" {results[name]['creation'] * 1000:13.3f}" for name in results))
    for label, index in (("vision per image, batch 1", 0), ("vision per image, batch 8", 1)):
        print(f"{label:32}" + "".join(f" {results[name]['times'][index] * 1000:13.3f}"
                                        for name in ("vision fp32", "vision int8")))
    print(f"{'text per query':32}" + " " * 28 + f" {results['text fp32']['times'][0] * 1000:13.3f}")


def main():
    args = parse_args()
    artifacts = {item["name"]: item for item in bench.read(BENCHMARKS / "artifacts.json")}
    for path, name in ((args.visual_fp32, "datacomp_vision.onnx"),
                       (args.textual_fp32, "datacomp_text.onnx"),
                       (args.tokenizer, "datacomp_tokenizer.json")):
        verify(path, artifacts[name]["sha256"])

    if args.coco is not None:
        evaluate_coco(args)
        return

    image_rows = bench.inputs()
    paths = [row["file_name"] for row in image_rows]
    images = []
    for row in image_rows:
        path = args.image_objects / row["sha256"]
        verify(path, row["sha256"])
        images.append(load_image(path))

    specification = json.loads((BENCHMARKS / "corpora/quality-500.json").read_text())
    queries = specification["queries"]
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.enable_truncation(max_length=77)
    tokenizer.enable_padding(length=77, pad_id=0, pad_token="<|endoftext|>")
    tokens = [np.asarray([tokenizer.encode(row["query"], add_special_tokens=True).ids], dtype=np.int64)
              for row in queries]

    results = {}
    for name, path in (("vision fp32", args.visual_fp32), ("vision int8", args.visual_int8),
                       ("text fp32", args.textual_fp32), ("text int8", args.textual_int8)):
        print(f"Evaluating {name}...", file=sys.stderr, flush=True)
        session, creation = session_and_creation_time(path)
        if name.startswith("vision"):
            batch1 = [np.stack(images[i:i + 1]) for i in range(RUNS)]
            batch8 = [np.stack(images[i:i + 8]) for i in range(0, RUNS * 8, 8)]
            times = (latency(session, batch1), latency(session, batch8))
            vectors = image_embeddings(session, images)
        else:
            times = (latency(session, tokens),)
            vectors = text_embeddings(session, tokens)
        results[name] = {"creation": creation, "times": times, "vectors": vectors}
        del session

    print("Embedding cosine                        mean       minimum")
    for label in ("vision", "text"):
        cosine = np.sum(results[f"{label} fp32"]["vectors"] * results[f"{label} int8"]["vectors"], axis=1)
        print(f"{label:36} {np.mean(cosine):10.6f} {np.min(cosine):13.6f}")

    pairings = [("fp32/fp32", "vision fp32", "text fp32"),
                ("int8/fp32", "vision int8", "text fp32"),
                ("fp32/int8", "vision fp32", "text int8"),
                ("int8/int8", "vision int8", "text int8")]
    runs = [retrieval_runs(queries, paths, results[vision]["vectors"], results[text]["vectors"])
            for _, vision, text in pairings]
    metrics = [quality.evaluate(run, 0.25) for run in runs]
    print("\nPairings (vision/text)" + "".join(f" {name:>13}" for name, _, _ in pairings))
    for metric in metrics[0]:
        print(f"{metric:32}" + "".join(f" {row[metric]:13.4f}" for row in metrics))
    print(f"{'top10_agreement':32}" + "".join(f" {quality.top10_agreement(runs[0], run):13.4f}" for run in runs))

    print("\nTiming (median of 20 runs, ms)      vision fp32   vision int8     text fp32     text int8")
    print(f"{'session creation':32}" + "".join(f" {results[name]['creation'] * 1000:13.3f}" for name in results))
    for label, index in (("vision per image, batch 1", 0), ("vision per image, batch 8", 1)):
        print(f"{label:32}" + "".join(f" {results[name]['times'][index] * 1000:13.3f}" for name in ("vision fp32", "vision int8")))
    print(f"{'text per query':32}" + " " * 28 + "".join(f" {results[name]['times'][0] * 1000:13.3f}" for name in ("text fp32", "text int8")))


if __name__ == "__main__":
    main()
