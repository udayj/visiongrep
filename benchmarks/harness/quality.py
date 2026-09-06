"""Retrieval and annotation-defined leave-category-out abstention evaluation."""

from __future__ import annotations

import math
import random
import sqlite3
import statistics
import struct
from pathlib import Path


def evaluate(runs: list[dict], threshold: float) -> dict:
    metrics = {}
    captions = [row for row in runs if row["kind"] == "caption"]
    for k in (1, 5, 10):
        recalls, gains = [], []
        for row in captions:
            relevant = set(row["relevant"])
            matches = [item["path"] in relevant for item in row["results"][:k]]
            recalls.append(sum(matches) / len(relevant))
            ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
            gains.append(
                sum(hit / math.log2(i + 2) for i, hit in enumerate(matches)) / ideal
            )
        metrics[f"recall_at_{k}"] = statistics.mean(recalls)
        if k == 10:
            metrics["ndcg_at_10"] = statistics.mean(gains)
    metrics["mrr_at_10"] = statistics.mean(
        next(
            (
                1 / i
                for i, item in enumerate(row["results"][:10], 1)
                if item["path"] in row["relevant"]
            ),
            0,
        )
        for row in captions
    )
    groups = {}
    for row in runs:
        if row["kind"] != "category":
            continue
        accepted = [item for item in row["results"] if item["score"] >= threshold]
        absent_accepted = [
            item for item in accepted if item["path"] not in row["relevant"]
        ]
        groups.setdefault(row["intent"], []).append(
            (bool(absent_accepted), not bool(accepted))
        )
    metrics["absent_false_positive_rate"] = statistics.mean(
        statistics.mean(x[0] for x in group) for group in groups.values()
    )
    metrics["present_false_negative_rate"] = statistics.mean(
        statistics.mean(x[1] for x in group) for group in groups.values()
    )
    rng = random.Random(91017)
    intervals = {}
    for index, name in enumerate(
        ("absent_false_positive_rate", "present_false_negative_rate")
    ):
        values = [statistics.mean(x[index] for x in group) for group in groups.values()]
        draws = sorted(
            statistics.mean(rng.choices(values, k=len(values))) for _ in range(2000)
        )
        intervals[name] = [draws[49], draws[1949]]
    metrics["intent_bootstrap_intervals_95"] = intervals
    metrics.update(
        category_intents=len(groups),
        caption_queries=len(captions),
        threshold=threshold,
        absence_scope="annotation-defined leave-category-out galleries; scores filtered after full ranking",
        paraphrases="macro-averaged within intent; not independent samples",
        uncertainty="conditional on this fixed image corpus and annotation-defined judgments",
    )
    return metrics


def run(scenario, specification: dict, progress) -> dict:
    runs = []
    for index, query in enumerate(specification["queries"]):
        progress(
            query=query["id"], completed=index, total=len(specification["queries"])
        )
        result = scenario.execute(
            f"quality-{index:04d}", query["query"], measured=False, top=scenario.count
        )
        # All scores are required: filtering only a top-K list could conceal false positives.
        if len(result["results"]) != scenario.count:
            raise ValueError(
                "quality evaluation requires one score for every corpus image"
            )
        runs.append(
            query | {"results": result["results"], "exit_code": result["exit_code"]}
        )
    return {"metrics": evaluate(runs, specification["threshold"]), "runs": runs}


def embeddings(path: Path) -> dict:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        return {
            bytes(key) if isinstance(key, bytes) else key.encode(): struct.unpack(
                "<512f", blob
            )
            for key, blob in connection.execute("SELECT path, embedding FROM images")
        }


def compare(
    reference: dict, candidate: dict, reference_index: Path, candidate_index: Path
) -> dict:
    a, b = embeddings(reference_index), embeddings(candidate_index)
    if a.keys() != b.keys():
        return {"passed": False, "reason": "indexed image membership changed"}
    max_error = max(abs(x - y) for key in a for x, y in zip(a[key], b[key]))
    normalized = all(
        abs(math.sqrt(sum(x * x for x in vector)) - 1) <= 0.001
        for values in (a, b)
        for vector in values.values()
    )
    ar, br = reference["runs"], candidate["runs"]
    if [q["id"] for q in ar] != [q["id"] for q in br]:
        raise ValueError("quality query identities differ")
    agreement, max_score_error, exits = [], 0, True
    for left, right in zip(ar, br):
        agreement.append(
            [v["path"] for v in left["results"][:10]]
            == [v["path"] for v in right["results"][:10]]
        )
        scores = {item["path"]: item["score"] for item in right["results"]}
        if scores.keys() != {item["path"] for item in left["results"]}:
            return {"passed": False, "reason": "quality result membership changed"}
        max_score_error = max(
            max_score_error,
            max(abs(item["score"] - scores[item["path"]]) for item in left["results"]),
        )
        exits &= left["exit_code"] == right["exit_code"]
    metric_names = [
        key
        for key, value in reference["metrics"].items()
        if isinstance(value, (int, float))
    ]
    equal_metrics = all(
        abs(reference["metrics"][key] - candidate["metrics"][key]) <= 1e-12
        for key in metric_names
    )
    return {
        "passed": normalized
        and max_error <= 1e-4
        and max_score_error <= 1e-4
        and all(agreement)
        and exits
        and equal_metrics,
        "embedding_max_absolute_error": max_error,
        "normalized": normalized,
        "score_max_absolute_error": max_score_error,
        "top10_exact_agreement": statistics.mean(agreement),
        "exit_code_agreement": exits,
        "metrics_equal": equal_metrics,
        "policy": "fixed model/preprocessing; any ranking or quality change needs investigation",
    }
