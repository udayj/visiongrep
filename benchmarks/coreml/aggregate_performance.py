#!/usr/bin/env python3
"""Aggregate repeated Core ML and CPU ONNX end-to-end indexing measurements."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, nargs="+", type=Path)
    parser.add_argument("--coreml", required=True, nargs="+", type=Path)
    parser.add_argument("--embedding-report", required=True, type=Path)
    parser.add_argument("--behavior-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(paths: list[Path]) -> dict[str, Any]:
    reports = [read(path)["performance"] for path in paths]
    return {
        "samples": len(reports),
        "first_index_and_query_median_ms": statistics.median(
            report["first_index_and_query_ms"] for report in reports
        ),
        "indexing_throughput_median_images_per_second": statistics.median(
            report["end_to_end_first_index_images_per_second"] for report in reports
        ),
        "peak_rss_median_kib": statistics.median(report["peak_rss_kib"] for report in reports),
        "index_size_median_bytes": statistics.median(
            report["index_size_bytes"] for report in reports
        ),
    }


def main() -> None:
    args = parse_args()
    onnx = summarize(args.onnx)
    coreml = summarize(args.coreml)
    embedding = read(args.embedding_report)
    behavior = read(args.behavior_report)
    throughput_gain = (
        coreml["indexing_throughput_median_images_per_second"]
        / onnx["indexing_throughput_median_images_per_second"]
        - 1.0
    )
    wall_reduction = 1.0 - (
        coreml["first_index_and_query_median_ms"] / onnx["first_index_and_query_median_ms"]
    )
    correctness_passed = (
        embedding["passed"]
        and behavior["top_k_paths_and_order_exact"]
        and behavior["threshold_decisions_exact"]
    )
    performance_passed = throughput_gain >= 0.20 or wall_reduction >= 0.15
    report = {
        "schema_version": 1,
        "reference": "CPU ONNX",
        "candidate": "Core ML on real Apple Silicon",
        "correctness_passed": correctness_passed,
        "performance_passed": performance_passed,
        "performance_gate": {
            "minimum_throughput_gain": 0.20,
            "minimum_first_index_wall_reduction": 0.15,
        },
        "measured": {
            "throughput_gain_fraction": throughput_gain,
            "first_index_wall_reduction_fraction": wall_reduction,
        },
        "onnx": onnx,
        "coreml": coreml,
        "production_decision": "defer",
        "decision_reason": (
            "This workflow validates rclip's reference Core ML package, not a maintainable Rust "
            "VisionGrep integration. No production Core ML code or dependency is retained in this branch."
        ),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not correctness_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
