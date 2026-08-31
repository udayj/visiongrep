#!/usr/bin/env python3
"""Run repeatable, process-cold VisionGrep phase-timing scenarios."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Any

SUPPORTED_CORPUS_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--cache-home", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", default=21, type=int)
    parser.add_argument("--include-network", action="store_true")
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def summarize(reports: list[dict[str, Any]]) -> dict[str, Any]:
    totals = [report["total_wall_ms"] for report in reports]
    phases: dict[str, list[float]] = {}
    for report in reports:
        for phase in report["phases"]:
            phases.setdefault(phase["phase"], []).append(phase["elapsed_ms"])
    return {
        "samples": len(reports),
        "total_wall_median_ms": statistics.median(totals),
        "total_wall_p95_ms": percentile(totals, 0.95) if len(totals) >= 20 else None,
        "phases": {
            name: {
                "median_ms": statistics.median(values),
                "p95_ms": percentile(values, 0.95) if len(values) >= 20 else None,
            }
            for name, values in sorted(phases.items())
        },
        "environment": reports[0]["environment"],
    }


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.environment = os.environ.copy()
        self.environment["HOME"] = str(args.cache_home.resolve())
        self.environment["XDG_CACHE_HOME"] = str(args.cache_home.resolve())
        args.cache_home.mkdir(parents=True, exist_ok=True)

    def invoke(
        self,
        scenario: str,
        sample: int,
        query: str,
        corpus: Path,
        *,
        index_path: Path | None = None,
        no_cache: bool = False,
        environment: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        timing_dir = self.args.output / "raw" / scenario
        timing_dir.mkdir(parents=True, exist_ok=True)
        timing_path = timing_dir / f"{sample:03d}.json"
        command = [
            str(self.args.binary.resolve()),
            query,
            str(corpus.resolve()),
            "--top",
            "10",
            "--threshold",
            "-1",
            "--json",
            "--quiet",
            "--timing",
            "--timing-file",
            str(timing_path.resolve()),
        ]
        if index_path is not None:
            command.extend(["--index-path", str(index_path.resolve())])
        if no_cache:
            command.append("--no-cache")
        completed = subprocess.run(
            command,
            capture_output=True,
            env=environment or self.environment,
            check=False,
        )
        if completed.returncode not in (0, 1):
            raise RuntimeError(
                f"scenario {scenario} sample {sample} failed with {completed.returncode}: "
                f"{completed.stderr.decode(errors='replace')}"
            )
        return json.loads(timing_path.read_text(encoding="utf-8"))


def set_read_only(root: Path, read_only: bool) -> None:
    directory_mode = 0o555 if read_only else 0o755
    file_mode = 0o444 if read_only else 0o644
    for path in root.rglob("*"):
        path.chmod(directory_mode if path.is_dir() else file_mode)
    root.chmod(directory_mode)


def main() -> None:
    args = parse_args()
    if args.samples < 3:
        raise ValueError("--samples must be at least 3")
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output directory {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    runner = Runner(args)
    indexes = args.output / "indexes"
    indexes.mkdir()
    queries = [f"benchmark novel query {index}: red bicycle near water" for index in range(args.samples)]
    results: dict[str, Any] = {}

    reports = [
        runner.invoke(
            "models_installed_index_absent",
            sample,
            queries[sample],
            args.corpus,
            index_path=indexes / f"absent-{sample:03d}.db",
        )
        for sample in range(args.samples)
    ]
    results["models_installed_index_absent"] = summarize(reports)

    reports = [
        runner.invoke("no_cache", sample, queries[sample], args.corpus, no_cache=True)
        for sample in range(args.samples)
    ]
    results["no_cache"] = summarize(reports)

    warm_index = indexes / "warm.db"
    runner.invoke("warm_setup", 0, "warm setup query", args.corpus, index_path=warm_index)
    reports = [
        runner.invoke(
            "unchanged_novel_query",
            sample,
            queries[sample],
            args.corpus,
            index_path=warm_index,
        )
        for sample in range(args.samples)
    ]
    results["unchanged_novel_query"] = summarize(reports)

    cached_query = "exact cached benchmark query"
    runner.invoke("cached_setup", 0, cached_query, args.corpus, index_path=warm_index)
    reports = [
        runner.invoke(
            "unchanged_exact_cached_query",
            sample,
            cached_query,
            args.corpus,
            index_path=warm_index,
        )
        for sample in range(args.samples)
    ]
    results["unchanged_exact_cached_query"] = summarize(reports)

    changed_corpus = args.output / "changed-corpus"
    shutil.copytree(args.corpus, changed_corpus)
    changed_index = indexes / "changed.db"
    runner.invoke("changed_setup", 0, "changed setup query", changed_corpus, index_path=changed_index)
    image_paths = sorted(
        path
        for path in changed_corpus.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_CORPUS_SUFFIXES
    )
    change_count = max(1, round(len(image_paths) * 0.01))
    reports = []
    for sample in range(args.samples):
        for offset in range(change_count):
            path = image_paths[(sample * change_count + offset) % len(image_paths)]
            metadata = path.stat()
            os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000_000))
        reports.append(
            runner.invoke(
                "one_percent_changed",
                sample,
                queries[sample],
                changed_corpus,
                index_path=changed_index,
            )
        )
    results["one_percent_changed"] = summarize(reports) | {
        "changed_images_per_sample": change_count
    }

    read_only_corpus = args.output / "read-only-corpus"
    shutil.copytree(args.corpus, read_only_corpus)
    try:
        set_read_only(read_only_corpus, True)
        reports = [
            runner.invoke(
                "read_only_out_of_tree_index",
                sample,
                queries[sample],
                read_only_corpus,
                index_path=indexes / f"read-only-{sample:03d}.db",
            )
            for sample in range(args.samples)
        ]
    finally:
        set_read_only(read_only_corpus, False)
    results["read_only_out_of_tree_index"] = summarize(reports)

    if args.include_network:
        network_home = args.output / "network-cache"
        network_environment = os.environ.copy()
        network_environment["HOME"] = str(network_home.resolve())
        network_environment["XDG_CACHE_HOME"] = str(network_home.resolve())
        report = runner.invoke(
            "models_and_index_absent_network_variable",
            0,
            "network scenario query",
            args.corpus,
            index_path=indexes / "network.db",
            environment=network_environment,
        )
        results["models_and_index_absent_network_variable"] = summarize([report]) | {
            "network_time_is_variable": True
        }

    output = {
        "schema_version": 1,
        "process_state": "process-cold for every recorded invocation",
        "filesystem_cache_state": "warm/uncontrolled; GitHub-hosted runners do not expose safe cache eviction",
        "model_state": "installed except the explicitly separate network scenario",
        "scenarios": results,
    }
    (args.output / "summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
