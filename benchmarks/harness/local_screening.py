"""Coarse local triage: precision of typical latency, not uniform invocations."""

import random
import statistics

from .statistics import percentile, summary
from .calibration import compatible_harness

POLICY = "local-median-v3"
MEDIAN_PRECISION = 0.10
BATCH_DRIFT = 0.15
# v3 changes analysis only. Do not accept arbitrary old measurement harnesses.
REANALYZABLE_V2 = "6d323d1709693c70c61aae7f924703faa1e51b0828808aa86fce76434d52b79c"


def enabled(profile: dict) -> bool:
    return profile["name"] == "local-quick" and not profile.get("cloud", False)


def batches(times: list[float]) -> list[float]:
    if len(times) < 9 or len(times) % 3:
        raise ValueError(
            "local screening requires at least three complete triples; fresh recordings required"
        )
    summary(times)
    if any(value <= 0 for value in times):
        raise ValueError("local timing measurements must be positive")
    return [statistics.median(times[i : i + 3]) for i in range(0, len(times), 3)]


def estimate(sessions: list[list[float]]) -> dict:
    """Retain session boundaries when estimating median uncertainty.

    With just three sessions/batches this is an approximate screening interval,
    not a calibrated confidence guarantee. Batch drift guards against pooling
    observations from visibly different timing regimes.
    """
    if not sessions:
        raise ValueError("local estimate requires sessions")
    grouped = [batches(times) for times in sessions]
    medians = [value for session in grouped for value in session]
    center = statistics.median(medians)
    rng = random.Random(81473)
    draws = []
    for _ in range(5000):
        selected = rng.choices(grouped, k=len(grouped))
        draws.append(
            statistics.median(
                [
                    value
                    for session in selected
                    for value in rng.choices(session, k=len(session))
                ]
            )
        )
    interval = [percentile(draws, 0.025), percentile(draws, 0.975)]
    precision = max(abs(value - center) for value in interval) / center
    drift = max(abs(value - center) for value in medians) / center
    reasons = []
    if precision > MEDIAN_PRECISION:
        reasons.append("median uncertainty exceeds 10%")
    if drift > BATCH_DRIFT:
        reasons.append("batch median drift exceeds 15%")
    return {
        "median_ms": center,
        "median_interval_ms": interval,
        "interval_method": "95% approximate hierarchical session/batch percentile bootstrap; 5000 draws",
        "relative_median_uncertainty": precision,
        "max_relative_batch_deviation": drift,
        "batch_medians_ms": grouped,
        "session_medians_ms": [statistics.median(session) for session in grouped],
        "raw_summaries": [summary(times) for times in sessions],
        "reasons": reasons,
    }


def uncertainty(times: list[float]) -> list[str]:
    summary(times)
    if len(times) < 9 or len(times) % 3:
        return ["fewer than three complete calibration batches (rehearsal only)"]
    return estimate([times])["reasons"]


def reference_bounds(info: dict) -> list[float]:
    lo, hi = info["median_interval_ms"]
    return [lo * (1 - MEDIAN_PRECISION), hi * (1 + MEDIAN_PRECISION)]


def compatible_recording(contract: dict) -> bool:
    profile = contract["profile"]
    return (
        enabled(profile)
        and profile["samples"] == 9
        and (
            (
                profile.get("screening_policy") == POLICY
                and compatible_harness(contract.get("harness_sha256"))
            )
            or (
                profile.get("screening_policy") == "local-batches-v2"
                and contract.get("harness_sha256") == REANALYZABLE_V2
            )
        )
    )


def candidate_decision(
    comparisons: dict, resources: dict, scenarios: list[str]
) -> tuple[str, list[str]]:
    """A local shortlist decision never authorizes or substitutes for cloud runs."""
    required_resources = {
        name + ":" + field
        for name in scenarios
        for field in ("peak_rss_bytes", "index_bytes")
    }
    if set(comparisons) != set(scenarios) or not required_resources <= resources.keys():
        return "inconclusive", ["incomplete local timing or resource comparisons"]
    checks = comparisons | resources
    regressions = [name for name, item in checks.items() if item["interval"][1] < -0.05]
    if regressions:
        return "does_not_qualify", [
            "material local regression: " + name for name in regressions
        ]
    if any(item["interval"][0] < -0.05 for item in checks.values()):
        return "inconclusive", [
            "cannot exclude a material regression in the local suite"
        ]
    improved = [
        name
        for name, item in comparisons.items()
        if item["improvement"] >= 0.05 and item["interval"][0] > 0
    ]
    if improved:
        return "promising", [
            "supported local improvement: " + name for name in improved
        ] + [
            "local suite only; eligible for consideration for cloud validation, not qualified"
        ]
    if all(item["interval"][1] < 0.05 for item in comparisons.values()):
        return "does_not_qualify", [
            "no material improvement supported by the local suite"
        ]
    return "inconclusive", ["local improvement remains uncertain"]
