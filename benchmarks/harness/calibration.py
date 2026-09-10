"""Coarse environment calibration, separate from paired candidate qualification."""

from .statistics import summary
from .storage import harness_digest

CLOUD_POLICY = "cloud-session-median-v1"
MAX_VARIATION = 0.10
# These changes affect analysis only, not the recorded invocations or their timing.
REANALYZABLE_V3 = "f6750c4af39c4f55de442f197e217394fe24f9ef01c40d87e17bfa38e95feec9"


def without_microcode(contract: dict) -> dict:
    """Keep host firmware in report metadata, not the compatibility contract."""
    environment = contract["environment"]
    if "hardware" not in environment:
        return contract
    return contract | {
        "environment": environment
        | {
            "hardware": {
                key: value
                for key, value in environment["hardware"].items()
                if key != "microcode"
            }
        }
    }


def session_summary(times: list[float]) -> dict:
    info = summary(times)
    if any(value <= 0 for value in times) or info["cv"] > MAX_VARIATION:
        raise ValueError(
            "foundation variation too large; session CV exceeds 10% or timing is nonpositive"
        )
    return info


def reference(sessions: list[list[float]]) -> dict:
    """Give independent sessions equal weight; retain every raw observation."""
    summaries = [session_summary(times) for times in sessions]
    medians = [info["median"] for info in summaries]
    center = summary(medians)["median"]
    bounds = [center * (1 - MAX_VARIATION), center * (1 + MAX_VARIATION)]
    if any(not bounds[0] <= median <= bounds[1] for median in medians):
        raise ValueError(
            "foundation variation too large; session median drift exceeds 10%"
        )
    return {"median_ms": center, "bounds_ms": bounds, "sessions": summaries}


def compatible_harness(fingerprint: str | None) -> bool:
    return fingerprint in (harness_digest(), REANALYZABLE_V3)
