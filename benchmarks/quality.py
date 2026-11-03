"""Metrics for the fixed retrieval-quality query set."""

import math
import statistics


def evaluate(runs, threshold):
    captions = [row for row in runs if row["kind"] == "caption"]
    metrics = {}
    for k in (1, 5, 10):
        recalls, gains = [], []
        for row in captions:
            relevant = set(row["relevant"])
            matches = [item["path"] in relevant for item in row["results"][:k]]
            recalls.append(sum(matches) / len(relevant))
            ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
            gains.append(sum(hit / math.log2(i + 2) for i, hit in enumerate(matches)) / ideal)
        metrics[f"recall_at_{k}"] = statistics.mean(recalls)
        if k == 10:
            metrics["ndcg_at_10"] = statistics.mean(gains)
    metrics["mrr_at_10"] = statistics.mean(
        next((1 / i for i, item in enumerate(row["results"][:10], 1)
              if item["path"] in row["relevant"]), 0)
        for row in captions
    )
    groups = {}
    for row in runs:
        if row["kind"] != "category":
            continue
        accepted = [item for item in row["results"] if item["score"] >= threshold]
        absent_accepted = any(item["path"] not in row["relevant"] for item in accepted)
        groups.setdefault(row["intent"], []).append((absent_accepted, not bool(accepted)))
    metrics["absent_false_positive_rate"] = statistics.mean(
        statistics.mean(item[0] for item in group) for group in groups.values()
    )
    metrics["present_false_negative_rate"] = statistics.mean(
        statistics.mean(item[1] for item in group) for group in groups.values()
    )
    return metrics


def top10_agreement(reference, candidate):
    if [row["id"] for row in reference] != [row["id"] for row in candidate]:
        raise ValueError("quality query identities differ")
    return statistics.mean(
        [item["path"] for item in left["results"][:10]]
        == [item["path"] for item in right["results"][:10]]
        for left, right in zip(reference, candidate)
    )
