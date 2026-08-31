#!/usr/bin/env python3
"""Run one VisionGrep build against the shared retrieval corpus."""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

from common import corpus_image_count, environment_metadata, load_queries, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--corpus-manifest", required=True, type=Path)
    parser.add_argument("--query-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-home", required=True, type=Path)
    parser.add_argument("--system", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--model-contract", required=True)
    parser.add_argument("--index-path", type=Path)
    parser.add_argument("--timing-dir", type=Path)
    parser.add_argument("--top", default=10, type=int)
    return parser.parse_args()


def run_query(
    args: argparse.Namespace,
    query: dict[str, Any],
    index: int,
    environment: dict[str, str],
    *,
    timing_suffix: str = "",
) -> tuple[dict[str, Any], float]:
    command = [
        str(args.binary.resolve()),
        query["query"],
        str(args.corpus_dir.resolve()),
        "--top",
        str(args.top),
        "--threshold",
        "-1",
        "--json",
        "--quiet",
    ]
    timing_path = None
    if args.index_path is not None:
        command.extend(["--index-path", str(args.index_path.resolve())])
    if args.timing_dir is not None:
        args.timing_dir.mkdir(parents=True, exist_ok=True)
        timing_path = args.timing_dir / f"{index:04d}{timing_suffix}.json"
        command.extend(["--timing", "--timing-file", str(timing_path.resolve())])

    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, env=environment, check=False)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if completed.returncode not in (0, 1):
        raise RuntimeError(
            f"VisionGrep failed for {query['id']} with {completed.returncode}: "
            f"{completed.stderr.decode(errors='replace')}"
        )
    parsed = json.loads(completed.stdout)
    result = query | {
        "latency_ms": elapsed_ms,
        "results": [
            {"path": Path(item["path"]).name, "score": item["score"]} for item in parsed
        ],
    }
    if timing_path is not None:
        result["timing_file"] = timing_path.name
    return result, elapsed_ms


def main() -> None:
    args = parse_args()
    if args.top < 10:
        raise ValueError("--top must be at least 10 for Recall@10")
    queries = load_queries(args.corpus_manifest, args.query_manifest)
    environment = os.environ.copy()
    args.cache_home.mkdir(parents=True, exist_ok=True)
    environment["HOME"] = str(args.cache_home.resolve())
    environment["XDG_CACHE_HOME"] = str(args.cache_home.resolve())

    if args.index_path is not None:
        if args.index_path.exists():
            raise RuntimeError(f"refusing to reuse benchmark index {args.index_path}")
        args.index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path = args.index_path
    else:
        index_path = args.corpus_dir / ".visiongrep.db"
        if index_path.exists():
            raise RuntimeError(f"refusing to reuse benchmark index {index_path}")

    corpus_size = corpus_image_count(args.corpus_dir)
    runs = []
    indexing_started = time.perf_counter()
    for index, query in enumerate(queries):
        run, _ = run_query(args, query, index, environment)
        if index == 0:
            indexing_ms = (time.perf_counter() - indexing_started) * 1000.0
        runs.append(run)
    cached_run, cached_latency_ms = run_query(
        args, queries[0], 0, environment, timing_suffix="-cached"
    )

    report = {
        "schema_version": 1,
        "system": {
            "name": args.system,
            "commit": args.commit,
            "model_contract": args.model_contract,
            "binary": str(args.binary.resolve()),
        },
        "environment": environment_metadata(),
        "corpus": read_json(args.corpus_manifest),
        "performance": {
            "first_index_and_query_ms": indexing_ms,
            "end_to_end_first_index_images_per_second": corpus_size / (indexing_ms / 1000.0),
            "cached_query_ms": cached_latency_ms,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            "index_size_bytes": index_path.stat().st_size,
            "corpus_size": corpus_size,
        },
        "cached_query": cached_run,
        "runs": runs,
    }
    write_json(args.output, report)


if __name__ == "__main__":
    main()
