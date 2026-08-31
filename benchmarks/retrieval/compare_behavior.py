#!/usr/bin/env python3
"""Compare top-K paths, ordering, and threshold decisions, strictly unless report-only."""

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
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="record behavioral differences without making them a failed correctness gate",
    )
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
    ranking_matches = 0
    membership_matches = 0
    threshold_matches = 0
    no_match_matches = 0
    top_one_matches = 0
    score_differences = []
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
        rankings_exact = reference_paths == candidate_paths
        membership_exact = set(reference_paths) == set(candidate_paths)
        threshold_paths_exact = set(reference_accepted) == set(candidate_accepted)
        no_match_exact = bool(reference_accepted) == bool(candidate_accepted)
        top_one_exact = reference_paths[:1] == candidate_paths[:1]
        ranking_matches += rankings_exact
        membership_matches += membership_exact
        threshold_matches += threshold_paths_exact
        no_match_matches += no_match_exact
        top_one_matches += top_one_exact

        candidate_scores = {result["path"]: result["score"] for result in candidate_results}
        score_differences.extend(
            abs(result["score"] - candidate_scores[result["path"]])
            for result in reference_results
            if result["path"] in candidate_scores
        )
        if not (rankings_exact and threshold_paths_exact and no_match_exact):
            mismatches.append(
                {
                    "query_id": query_id,
                    "top_k_paths_and_order_exact": rankings_exact,
                    "top_k_membership_exact": membership_exact,
                    "threshold_accepted_paths_exact": threshold_paths_exact,
                    "no_match_decision_exact": no_match_exact,
                    "reference_paths": reference_paths,
                    "candidate_paths": candidate_paths,
                    "reference_accepted": reference_accepted,
                    "candidate_accepted": candidate_accepted,
                }
            )

    query_count = len(reference_runs)
    report = {
        "schema_version": 1,
        "reference": reference["system"],
        "candidate": candidate["system"],
        "threshold": args.threshold,
        "queries": query_count,
        "top_k_paths_and_order_exact": ranking_matches == query_count,
        "top_k_paths_and_order_exact_rate": ranking_matches / query_count,
        "top_k_membership_exact_rate": membership_matches / query_count,
        "top_one_path_exact_rate": top_one_matches / query_count,
        "threshold_decisions_exact": threshold_matches == query_count,
        "threshold_decisions_exact_rate": threshold_matches / query_count,
        "no_match_decisions_exact": no_match_matches == query_count,
        "no_match_decisions_exact_rate": no_match_matches / query_count,
        "maximum_score_difference_on_shared_paths": max(score_differences, default=0.0),
        "mismatches": mismatches,
    }
    write_json(args.output, report)
    if mismatches and not args.report_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
