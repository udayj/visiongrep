#!/usr/bin/env python3
"""Require identical top-K paths, ordering, and threshold decisions between two runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", default=0.25, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = read_json(args.reference)
    candidate = read_json(args.candidate)
    reference_runs = {run["id"]: run for run in reference["runs"]}
    candidate_runs = {run["id"]: run for run in candidate["runs"]}
    if reference_runs.keys() != candidate_runs.keys():
        raise RuntimeError("reference and candidate query IDs differ")

    mismatches = []
    for query_id in sorted(reference_runs):
        reference_results = reference_runs[query_id]["results"]
        candidate_results = candidate_runs[query_id]["results"]
        reference_paths = [result["path"] for result in reference_results]
        candidate_paths = [result["path"] for result in candidate_results]
        reference_accepted = [
            result["path"] for result in reference_results if result["score"] >= args.threshold
        ]
        candidate_accepted = [
            result["path"] for result in candidate_results if result["score"] >= args.threshold
        ]
        if reference_paths != candidate_paths or reference_accepted != candidate_accepted:
            mismatches.append(
                {
                    "query_id": query_id,
                    "reference_paths": reference_paths,
                    "candidate_paths": candidate_paths,
                    "reference_accepted": reference_accepted,
                    "candidate_accepted": candidate_accepted,
                }
            )

    report = {
        "schema_version": 1,
        "reference": reference["system"],
        "candidate": candidate["system"],
        "threshold": args.threshold,
        "queries": len(reference_runs),
        "top_k_paths_and_order_exact": not mismatches,
        "threshold_decisions_exact": not mismatches,
        "mismatches": mismatches,
    }
    write_json(args.output, report)
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
