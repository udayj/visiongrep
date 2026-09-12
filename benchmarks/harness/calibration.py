"""Typical latency, block-bootstrap precision, and sustained timing drift."""

import random
import statistics

from .statistics import blocks, percentile, summary

METHOD = "median-block-bootstrap"
MEDIAN_PRECISION = 0.10
BATCH_DRIFT = 0.15
RESAMPLES = 5000


def without_microcode(contract: dict) -> dict:
    environment = contract["environment"]
    if "hardware" not in environment:
        return contract
    return contract | {"environment": environment | {"hardware": {
        key: value for key, value in environment["hardware"].items() if key != "microcode"
    }}}


def measurement_contract(contract: dict) -> dict:
    """Compare recorded experimental conditions, not analysis implementation labels.

    Keep original harness hashes in source provenance. Explicit reanalysis does not
    infer a different measurement protocol from an old analysis label.
    """
    result = without_microcode(contract).copy()
    result.pop("harness_sha256", None)
    result["profile"] = {key: value for key, value in result["profile"].items()
                         if key != "screening_policy"}
    return result


def batches(times: list[float]) -> list[float]:
    """Three consecutive, equal-duration-in-sample-count windows for drift checks."""
    width = len(times) // 3
    return [statistics.median(times[i:i + width]) for i in range(0, len(times), width)]


def estimate(sessions: list[list[float]]) -> dict:
    if not sessions:
        raise ValueError("median estimate requires sessions")
    raw = [summary(times) for times in sessions]
    if any(any(value <= 0 for value in times) for times in sessions):
        raise ValueError("latency samples must be positive")
    medians = [statistics.median(times) for times in sessions]
    center = statistics.median(medians)
    enough = all(len(times) >= 9 and len(times) % 3 == 0 for times in sessions)
    info = {"median_ms": center, "session_medians_ms": medians, "raw_summaries": raw,
            "median_interval_ms": None, "relative_median_uncertainty": None,
            "batch_medians_ms": [], "max_relative_batch_deviation": None,
            "max_relative_session_deviation": max(abs(m - center) / center for m in medians),
            "reasons": [], "interval_method": "95% approximate hierarchical block percentile bootstrap; 5000 draws; blocks of 3"}
    if not enough:
        info["reasons"].append("at least nine samples in complete triples per session required for median precision and drift")
        return info
    grouped = [blocks(times) for times in sessions]
    rng = random.Random(81473)
    draws = []
    for _ in range(RESAMPLES):
        selected = rng.choices(grouped, k=len(grouped))
        draws.append(statistics.median([
            statistics.median([value for block in rng.choices(session, k=len(session)) for value in block])
            for session in selected
        ]))
    interval = [percentile(draws, 0.025), percentile(draws, 0.975)]
    precision = max(abs(value - center) / center for value in interval)
    windows = [batches(times) for times in sessions]
    drift = max(abs(value - median) / median for window, median in zip(windows, medians) for value in window)
    info.update(median_interval_ms=interval, relative_median_uncertainty=precision,
                batch_medians_ms=windows, max_relative_batch_deviation=drift)
    if precision > MEDIAN_PRECISION:
        info["reasons"].append("median uncertainty exceeds 10%")
    if drift > BATCH_DRIFT:
        info["reasons"].append("batch median drift exceeds 15%")
    if info["max_relative_session_deviation"] > BATCH_DRIFT:
        info["reasons"].append("session median drift exceeds 15%")
    return info


def reference_bounds(info: dict) -> list[float]:
    # Do not compound a broad confidence interval with the environment-drift margin.
    return [info["median_ms"] * (1 - MEDIAN_PRECISION),
            info["median_ms"] * (1 + MEDIAN_PRECISION)]
