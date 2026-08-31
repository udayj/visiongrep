#!/usr/bin/env python3
"""Compute corpus-specific retrieval and abstention metrics from benchmark runs."""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Any

from common import read_json, write_json

DEFAULT_THRESHOLD = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percentile_value) - 1)
    return ordered[index]


def first_relevant_rank(run: dict[str, Any], limit: int) -> int | None:
    relevant = set(run["relevant"])
    for rank, result in enumerate(run["results"][:limit], start=1):
        if result["path"] in relevant:
            return rank
    return None


def metrics(report: dict[str, Any]) -> dict[str, Any]:
    positives = [run for run in report["runs"] if run["kind"] == "positive"]
    absent = [run for run in report["runs"] if run["kind"] == "absent"]
    output: dict[str, Any] = {}
    for k in (1, 5, 10):
        hits = []
        precisions = []
        discounted_gains = []
        for run in positives:
            relevant = set(run["relevant"])
            returned = run["results"][:k]
            relevant_count = sum(result["path"] in relevant for result in returned)
            hits.append(relevant_count > 0)
            precisions.append(relevant_count / k)
            ideal_count = min(len(relevant), k)
            ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
            gain = sum(
                1.0 / math.log2(rank + 1)
                for rank, result in enumerate(returned, start=1)
                if result["path"] in relevant
            )
            discounted_gains.append(gain / ideal if ideal else 0.0)
        output[f"recall_at_{k}"] = sum(hits) / len(hits)
        output[f"precision_at_{k}"] = statistics.fmean(precisions)
        output[f"ndcg_at_{k}"] = statistics.fmean(discounted_gains)

    reciprocal_ranks = []
    for run in positives:
        rank = first_relevant_rank(run, 10)
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
    output["mrr_at_10"] = statistics.fmean(reciprocal_ranks)

    def accepted(run: dict[str, Any], threshold: float) -> list[dict[str, Any]]:
        return [result for result in run["results"] if result["score"] >= threshold]

    false_positives = sum(bool(accepted(run, DEFAULT_THRESHOLD)) for run in absent)
    false_negatives = sum(
        not any(result["path"] in set(run["relevant"]) for result in accepted(run, DEFAULT_THRESHOLD))
        for run in positives
    )
    output["threshold"] = DEFAULT_THRESHOLD
    output["no_match_false_positive_rate"] = false_positives / len(absent)
    output["no_match_false_negative_rate"] = false_negatives / len(positives)
    output["coverage_precision"] = []
    for threshold in (0.15, 0.20, 0.25, 0.30, 0.35):
        covered = [run for run in positives if accepted(run, threshold)]
        correct = sum(
            accepted(run, threshold)[0]["path"] in set(run["relevant"]) for run in covered
        )
        output["coverage_precision"].append(
            {
                "threshold": threshold,
                "positive_query_coverage": len(covered) / len(positives),
                "top1_precision_when_covered": correct / len(covered) if covered else None,
                "absent_query_false_positive_rate": sum(
                    bool(accepted(run, threshold)) for run in absent
                )
                / len(absent),
            }
        )

    latencies = [run["latency_ms"] for run in report["runs"]]
    output["novel_query_latency_ms"] = {
        "samples": len(latencies),
        "median": statistics.median(latencies),
        "p95": percentile(latencies, 0.95),
    }
    output.update(report["performance"])
    output["corpus_specific"] = True
    output["judgement_note"] = (
        "Precision treats unjudged images as non-relevant; COCO captions can make this conservative."
    )
    return output


def main() -> None:
    args = parse_args()
    reports = [read_json(path) for path in args.results]
    summaries = [
        {"system": report["system"], "environment": report["environment"], "metrics": metrics(report)}
        for report in reports
    ]
    parity = []
    by_name = {report["system"]["name"]: report for report in reports}
    if "rclip" in by_name and "visiongrep-datacomp" in by_name:
        reference = {run["id"]: run for run in by_name["rclip"]["runs"]}
        candidate = {run["id"]: run for run in by_name["visiongrep-datacomp"]["runs"]}
        for query_id in sorted(reference):
            reference_paths = [result["path"] for result in reference[query_id]["results"]]
            candidate_paths = [result["path"] for result in candidate[query_id]["results"]]
            parity.append(reference_paths == candidate_paths)

    write_json(
        args.output,
        {
            "schema_version": 1,
            "systems": summaries,
            "rclip_datacomp_top10_parity": {
                "queries": len(parity),
                "exact": all(parity) if parity else None,
                "exact_rate": sum(parity) / len(parity) if parity else None,
            },
        },
    )


if __name__ == "__main__":
    main()
