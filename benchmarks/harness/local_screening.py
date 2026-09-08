"""Local timing uncertainty is separate from execution and correctness failures.

Keep the original tolerances; obtain more complete batches rather than fitting
limits to retained runs. No samples are trimmed, retried, or replaced.
"""

from .statistics import summary

POLICY = "local-batches-v2"


def enabled(profile: dict) -> bool:
    return profile["name"] == "local-quick" and not profile.get("cloud", False)


def batches(times: list[float]) -> list[float]:
    if len(times) < 9 or len(times) % 3:
        raise ValueError(
            "local screening requires at least three complete triples; fresh recordings required"
        )
    summary(times)
    return [summary(times[i : i + 3])["median"] for i in range(0, len(times), 3)]


def bounds(values: list[float]) -> tuple[list[float], list[str]]:
    info = summary(values)
    radius = max(3 * 1.4826 * info["mad"], 0.03 * info["median"])
    interval = [info["median"] - radius, info["median"] + radius]
    reasons = []
    if radius > 0.15 * info["median"] or info["cv"] > 0.10:
        reasons.append("calibration spread exceeds the frozen noise limits")
    if any(not interval[0] <= value <= interval[1] for value in values):
        reasons.append("outlying calibration batch")
    return interval, reasons


def uncertainty(times: list[float]) -> list[str]:
    info = summary(times)
    reasons = []
    if info["cv"] > 0.10:
        reasons.append("raw timing CV exceeds 10%")
    if len(times) < 9 or len(times) % 3:
        reasons.append("fewer than three complete calibration batches (rehearsal only)")
    elif summary(batches(times))["cv"] > 0.10:
        reasons.append("batch timing CV exceeds 10%")
    return reasons
