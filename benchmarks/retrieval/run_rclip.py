#!/usr/bin/env python3
"""Run pinned rclip source and artifacts against the shared retrieval corpus."""

from __future__ import annotations

import argparse
import hashlib
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common import (
    corpus_image_count,
    environment_metadata,
    load_queries,
    peak_rss_kib,
    read_json,
    write_json,
)

RCLIP_COMMIT = "3dcec2de5e23311473f6fb6433e602aa4f4ca812"
VISUAL_SHA256 = "3f7e6f94e5a34bc7ee8aba84aec0f963f56974ab405fbcd334c8e1c3f832bd2c"
TEXTUAL_SHA256 = "ee267cd64f0f77362670ae0140476ed51ee8c5a761d41636e09997f2fdddcacc"
TOKENIZER_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rclip-source", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--corpus-manifest", required=True, type=Path)
    parser.add_argument("--query-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--system", default="rclip")
    parser.add_argument("--top", default=10, type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_artifact(source: Path, destination: Path, expected_sha256: str) -> None:
    if sha256(source) != expected_sha256:
        raise RuntimeError(f"checksum mismatch for {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copyfile(source, destination)
    if sha256(destination) != expected_sha256:
        raise RuntimeError(f"installed artifact checksum mismatch for {destination}")


def main() -> None:
    args = parse_args()
    if args.top < 10:
        raise ValueError("--top must be at least 10 for Recall@10")
    actual_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=args.rclip_source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != RCLIP_COMMIT:
        raise RuntimeError(f"rclip source mismatch: expected {RCLIP_COMMIT}, got {actual_commit}")

    model_dir = args.data_dir / "ViT-B-32-256-datacomp_s34b_b86k"
    install_artifact(args.artifact_dir / "visual.onnx", model_dir / "visual.onnx", VISUAL_SHA256)
    install_artifact(args.artifact_dir / "textual.onnx", model_dir / "textual.onnx", TEXTUAL_SHA256)
    install_artifact(
        args.artifact_dir / "bpe_simple_vocab_16e6.txt.gz",
        args.data_dir / "tokenizer" / "bpe_simple_vocab_16e6.txt.gz",
        TOKENIZER_SHA256,
    )
    database_path = args.data_dir / "db.sqlite3"
    if database_path.exists():
        raise RuntimeError(f"refusing to reuse benchmark index {database_path}")

    os.environ["RCLIP_DATADIR"] = str(args.data_dir.resolve())
    sys.path.insert(0, str(args.rclip_source.resolve()))
    from rclip.main import init_rclip

    queries = load_queries(args.corpus_manifest, args.query_manifest)
    corpus_dir = str(args.corpus_dir.resolve())
    corpus_size = corpus_image_count(args.corpus_dir)
    indexing_started = time.perf_counter()
    rclip, model, database = init_rclip(
        corpus_dir,
        indexing_batch_size=8,
        no_indexing=False,
        max_image_pixels=None,
    )
    runs = []
    try:
        for index, query in enumerate(queries):
            started = time.perf_counter()
            results = rclip.search(query["query"], corpus_dir, args.top)
            latency_ms = (time.perf_counter() - started) * 1000.0
            if index == 0:
                indexing_ms = (time.perf_counter() - indexing_started) * 1000.0
            runs.append(
                query
                | {
                    "latency_ms": latency_ms,
                    "results": [
                        {"path": Path(result.filepath).name, "score": float(result.score)}
                        for result in results
                    ],
                }
            )
        started = time.perf_counter()
        cached_results = rclip.search(queries[0]["query"], corpus_dir, args.top)
        cached_latency_ms = (time.perf_counter() - started) * 1000.0
    finally:
        rclip.close()
        model.close()
        database.close()

    report = {
        "schema_version": 1,
        "system": {
            "name": args.system,
            "version": "3.3.0",
            "commit": RCLIP_COMMIT,
            "model_contract": "laion-clip-vit-b-32-256-datacomp-s34b-b86k",
        },
        "environment": environment_metadata(),
        "corpus": read_json(args.corpus_manifest),
        "performance": {
            "first_index_and_query_ms": indexing_ms,
            "end_to_end_first_index_images_per_second": corpus_size / (indexing_ms / 1000.0),
            "cached_query_ms": cached_latency_ms,
            "peak_rss_kib": peak_rss_kib(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "index_size_bytes": database_path.stat().st_size,
            "corpus_size": corpus_size,
        },
        "cached_query": {
            "id": queries[0]["id"],
            "latency_ms": cached_latency_ms,
            "results": [
                {"path": Path(result.filepath).name, "score": float(result.score)}
                for result in cached_results
            ],
        },
        "runs": runs,
    }
    write_json(args.output, report)


if __name__ == "__main__":
    main()
