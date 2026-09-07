"""Paired comparisons with fixed sample budgets and explicit inconclusive outcomes."""

from __future__ import annotations

import math
import random
import statistics


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def summary(values: list[float]) -> dict:
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("measurements must be finite, nonnegative, and nonempty")
    median = statistics.median(values)
    return {
        "samples": len(values),
        "median": median,
        "p95": percentile(values, 0.95) if len(values) >= 20 else None,
        "p95_estimate": "rough" if 20 <= len(values) < 100 else "empirical",
        "mad": statistics.median(abs(x - median) for x in values),
        "cv": statistics.stdev(values) / statistics.mean(values)
        if len(values) > 1 and statistics.mean(values)
        else 0,
    }


def paired(reference: list[float], candidate: list[float], alpha: float = 0.05) -> dict:
    if (
        len(reference) != len(candidate)
        or not reference
        or any(x <= 0 for x in reference)
    ):
        raise ValueError(
            "comparison requires equal nonempty pairs and positive reference times"
        )
    improvements = [1 - c / r for r, c in zip(reference, candidate)]
    rng = random.Random(73421)
    resamples = [
        statistics.median(rng.choices(improvements, k=len(improvements)))
        for _ in range(5000)
    ]
    return {
        "improvement": statistics.median(improvements),
        "interval": [
            percentile(resamples, alpha / 2),
            percentile(resamples, 1 - alpha / 2),
        ],
        "pairs": len(improvements),
        "method": "paired median-ratio percentile bootstrap",
        "alpha": alpha,
    }


def verdict(
    comparisons: dict,
    quality_ok: bool,
    environment_ok: bool,
    *,
    required_scenarios: tuple[str, ...],
    improvement: float = 0.05,
    regression: float = 0.05,
) -> tuple[str, list[str]]:
    if not environment_ok:
        return "invalid", [
            "foundation drift, unstable measurements, or incompatible environment"
        ]
    if not quality_ok:
        return "does_not_qualify", [
            "quality/behavior changed beyond the frozen tolerance"
        ]
    if not required_scenarios:
        raise ValueError("qualification requires a nonempty standard suite")
    definite_regressions = [
        name
        for name, result in comparisons.items()
        if result["interval"][1] < -regression
    ]
    if definite_regressions:
        return "does_not_qualify", [
            f"material regression: {name}" for name in definite_regressions
        ]
    missing = sorted(set(required_scenarios) - comparisons.keys())
    if missing:
        return "inconclusive", [
            f"standard scenario not measured: {name}" for name in missing
        ]
    if any(result["pairs"] < 20 for result in comparisons.values()):
        return "inconclusive", [
            "screening sample budget; confirmation requires at least 20 pairs"
        ]
    uncertain = [
        name for name, item in comparisons.items() if item["interval"][0] < -regression
    ]
    if uncertain:
        return "inconclusive", [
            f"cannot exclude material regression: {name}" for name in uncertain
        ]
    improved = [
        name
        for name in required_scenarios
        if comparisons[name]["improvement"] >= improvement
        and comparisons[name]["interval"][0] > 0
    ]
    if improved:
        return "qualifies", [
            f"supported improvement of at least {improvement:.0%}: {name}"
            for name in improved
        ] + ["all standard scenarios passed their regression checks"]
    if all(
        comparisons[name]["interval"][1] < improvement for name in required_scenarios
    ):
        return "does_not_qualify", [
            f"no standard scenario improved by {improvement:.0%}"
        ]
    return "inconclusive", [
        f"no supported improvement of at least {improvement:.0%} established"
    ]
