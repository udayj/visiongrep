"""Focused source-graph versus optimized-graph cloud diagnostics."""

from __future__ import annotations

import statistics
import time
from pathlib import Path

from . import behavior, calibration
from .scenarios import Scenario
from .statistics import paired, summary

REFERENCE_COMMIT = "4c8c613c8af16591860252683d29f4f0edaee723"
SCENARIOS = (
    "novel_text",
    "external_image_first",
    "persistent_text",
    "cached_text",
)
ENDPOINTS = (
    ("novel_text", "wall_ms", "primary"),
    ("external_image_first", "wall_ms", "primary"),
    ("persistent_text", "first_response_ms", "primary"),
    ("persistent_text", "wall_ms", "warm_control"),
    ("persistent_text", "cached_response_ms", "warm_control"),
    ("cached_text", "wall_ms", "warm_control"),
)
MATERIAL_THRESHOLD = 0.05


def materiality(interval: list[float]) -> tuple[str, str]:
    lower, upper = interval
    if upper < -MATERIAL_THRESHOLD:
        return "material_regression", "95% interval is entirely worse than -5%"
    if lower < -MATERIAL_THRESHOLD:
        return "inconclusive", "95% interval cannot exclude a regression over 5%"
    if lower > MATERIAL_THRESHOLD:
        return "material_improvement", "95% interval is entirely above 5%"
    if upper < MATERIAL_THRESHOLD:
        return (
            "material_improvement_not_demonstrated",
            "95% interval is entirely below 5%",
        )
    return "inconclusive", "95% interval crosses the 5% material threshold"


def effect_region(improvement: float) -> str:
    if improvement > MATERIAL_THRESHOLD:
        return "material_improvement"
    if improvement < -MATERIAL_THRESHOLD:
        return "material_regression"
    return "below_material_threshold"


def comparison(reference: list[float], candidate: list[float]) -> dict:
    result = paired(reference, candidate)
    reference_median = statistics.median(reference)
    candidate_median = statistics.median(candidate)
    interval_classification, reason = materiality(result["interval"])
    order_subgroups = {}
    for name, indices in (
        ("foundation_first", range(0, len(reference), 2)),
        ("candidate_first", range(1, len(reference), 2)),
    ):
        selected_reference = [reference[index] for index in indices]
        selected_candidate = [candidate[index] for index in indices]
        subgroup_improvement = 1 - statistics.median(
            selected_candidate
        ) / statistics.median(selected_reference)
        order_subgroups[name] = {
            "pairs": len(selected_reference),
            "improvement": subgroup_improvement,
            "candidate_to_reference_ratio": 1 - subgroup_improvement,
            "effect_region": effect_region(subgroup_improvement),
        }
    order_sensitive = len(
        {item["effect_region"] for item in order_subgroups.values()}
    ) > 1
    chronological_subgroups = {}
    width = len(reference) // 3
    for number in range(3):
        start = number * width
        end = start + width
        subgroup_improvement = 1 - statistics.median(
            candidate[start:end]
        ) / statistics.median(reference[start:end])
        chronological_subgroups[f"window_{number + 1}"] = {
            "pairs": width,
            "improvement": subgroup_improvement,
            "candidate_to_reference_ratio": 1 - subgroup_improvement,
            "effect_region": effect_region(subgroup_improvement),
        }
    chronological_drift = len(
        {item["effect_region"] for item in chronological_subgroups.values()}
    ) > 1
    return result | {
        "reference_median": reference_median,
        "candidate_median": candidate_median,
        "candidate_to_reference_ratio": candidate_median / reference_median,
        "ratio_interval": [1 - result["interval"][1], 1 - result["interval"][0]],
        "classification": interval_classification,
        "classification_reason": reason,
        "interval_classification": interval_classification,
        "material_threshold": MATERIAL_THRESHOLD,
        "order_subgroups": order_subgroups,
        "order_sensitive": order_sensitive,
        "chronological_subgroups": chronological_subgroups,
        "chronological_drift_across_threshold": chronological_drift,
    }


def cpu_times(path: Path = Path("/proc/stat")) -> list[int]:
    fields = path.read_text().splitlines()[0].split()
    if not fields or fields[0] != "cpu" or len(fields) < 9:
        raise ValueError("cannot read aggregate CPU counters from /proc/stat")
    # Guest counters are already included in user/nice time; do not count them twice.
    return [int(value) for value in fields[1:9]]


def quiet_machine_check(interval_seconds: float = 2) -> dict:
    before = cpu_times()
    time.sleep(interval_seconds)
    after = cpu_times()
    deltas = [end - start for start, end in zip(before, after)]
    if any(value < 0 for value in deltas) or sum(deltas) <= 0:
        raise ValueError("invalid aggregate CPU counter interval")
    idle_fraction = deltas[3] / sum(deltas)
    steal_fraction = deltas[7] / sum(deltas)
    result = {
        "interval_seconds": interval_seconds,
        "idle_fraction": idle_fraction,
        "steal_fraction": steal_fraction,
        "minimum_idle_fraction": 0.90,
        "maximum_steal_fraction": 0.01,
        "counter_deltas": deltas,
    }
    if idle_fraction < 0.90 or steal_fraction > 0.01:
        raise ValueError(f"machine was not quiet before measurement: {result}")
    return result


def measure(run, binaries: dict, identities: dict, cache: Path, corpus: dict) -> None:
    if identities["foundation"] != REFERENCE_COMMIT:
        raise ValueError(
            "graph-cache diagnostic requires v0.2.0 at " + REFERENCE_COMMIT
        )
    if set(binaries) != {"foundation", "candidate"}:
        raise ValueError("graph-cache diagnostic requires source and candidate binaries")

    run.report["graph_cache_diagnostic"] = {
        "diagnostic_only": True,
        "cloud_qualification": False,
        "reference": {"tag": "v0.2.0", "commit": REFERENCE_COMMIT},
        "sample_policy": "21 fixed paired samples; alternating binary order; no exclusion or repeats",
        "filesystem_policy": "foundation source ONNX and candidate optimized ONNX are each pre-read before their timed samples",
        "material_threshold": MATERIAL_THRESHOLD,
        "setup": {},
    }
    for name in SCENARIOS:
        run.progress(stage="graph-cache-diagnostic", scenario=name)
        instances = {}
        observations = {role: [] for role in binaries}
        run.report["samples"][name] = observations
        try:
            for role, binary in binaries.items():
                instances[role] = Scenario(
                    run.directory / "scenarios" / name / role,
                    binary,
                    cache,
                    corpus,
                    name,
                    run.invoke,
                    graph_cache_role=role,
                )
                run.report["graph_cache_diagnostic"]["setup"].setdefault(name, {})[
                    role
                ] = instances[role].graph_cache_setup
            run.report["graph_cache_diagnostic"].setdefault(
                "quiet_machine_checks", {}
            )[name] = quiet_machine_check()
            run.save()
            for sample_index in range(run.profile["samples"]):
                order = (
                    list(binaries)
                    if sample_index % 2 == 0
                    else list(reversed(binaries))
                )
                for role in order:
                    run.progress(
                        binary=role,
                        sample=sample_index,
                        total_samples=run.profile["samples"],
                    )
                    row = instances[role].sample(sample_index)
                    if row["timing"]["environment"]["commit"] != identities[role]:
                        raise ValueError(
                            "binary timing metadata disagrees with build identity"
                        )
                    observations[role].append(row)
                    run.save()
                observations["candidate"][-1]["behavior_comparison"] = behavior.compare(
                    observations["foundation"][-1],
                    observations["candidate"][-1],
                    instances["foundation"].index,
                    instances["candidate"].index,
                )
                run.save()
            run.report["graph_cache_diagnostic"].setdefault(
                "final_artifact_verification", {}
            )[name] = {
                role: instance.verify_graph_cache(include_content=True)
                for role, instance in instances.items()
            }
            run.save()
        finally:
            for instance in instances.values():
                instance.cleanup()
                run.discard_inputs(instance)
    run.summarize()


def summarize(run) -> None:
    expected_roles = {"foundation", "candidate"}
    if tuple(run.profile["scenarios"]) != SCENARIOS:
        raise ValueError("graph-cache diagnostic scenario protocol changed")
    if run.profile["samples"] != 21:
        raise ValueError("graph-cache diagnostic requires exactly 21 pairs")
    if set(run.report["samples"]) != set(SCENARIOS):
        raise ValueError("incomplete or unexpected graph-cache scenarios")

    behavior_failures = []
    for name in SCENARIOS:
        roles = run.report["samples"][name]
        if set(roles) != expected_roles or any(len(rows) != 21 for rows in roles.values()):
            raise ValueError(f"incomplete graph-cache measurements for {name}")
        for role, rows in roles.items():
            for index, row in enumerate(rows):
                if not row.get("behavior", {}).get("passed", False):
                    behavior_failures.append(f"{name}/{role}/{index}: behavior failed")
                if not row.get("graph_cache_validation"):
                    behavior_failures.append(
                        f"{name}/{role}/{index}: graph-cache validation missing"
                    )
                if role == "candidate" and not row.get(
                    "behavior_comparison", {}
                ).get("passed", False):
                    behavior_failures.extend(
                        f"{name}/candidate/{index}: {reason}"
                        for reason in row.get("behavior_comparison", {}).get(
                            "reasons", ["paired behavior failed"]
                        )
                    )

    run.report["analysis_method"] = calibration.METHOD
    diagnostic = run.report.setdefault("graph_cache_diagnostic", {})
    diagnostic.update(
        diagnostic_only=True,
        cloud_qualification=False,
        reference={"tag": "v0.2.0", "commit": REFERENCE_COMMIT},
        material_threshold=MATERIAL_THRESHOLD,
    )
    run.report["summaries"] = {name: {} for name in SCENARIOS}
    run.report["comparisons"] = {}
    timing_reasons = []
    estimates = {}
    for scenario, field, endpoint_class in ENDPOINTS:
        endpoint = scenario if field == "wall_ms" else f"{scenario}:{field}"
        by_role = run.report["samples"][scenario]
        values = {
            role: [row[field] for row in rows] for role, rows in by_role.items()
        }
        for role, samples in values.items():
            run.report["summaries"][scenario].setdefault(role, {})[field] = summary(
                samples
            )
            estimate = calibration.estimate([samples])
            estimates.setdefault(endpoint, {})[role] = estimate
            timing_reasons.extend(
                f"{endpoint}/{role}: {reason}" for reason in estimate["reasons"]
            )
        run.report["comparisons"][endpoint] = comparison(
            values["foundation"], values["candidate"]
        ) | {"endpoint_class": endpoint_class, "metric": field}

    run.report["timing_screen"] = {
        "method": calibration.METHOD,
        "verdict": "inconclusive" if timing_reasons else "stable",
        "reasons": timing_reasons,
        "estimates": estimates,
    }
    run.report["behavior_comparison"] = {
        "passed": not behavior_failures,
        "reasons": behavior_failures,
    }
    if behavior_failures:
        run.report.update(verdict="invalid", reasons=behavior_failures + timing_reasons)
        return

    findings = [
        f"{name}: {item['classification']}"
        for name, item in run.report["comparisons"].items()
    ]
    run.report.update(
        verdict="diagnostic_complete",
        reasons=[
            "Diagnostic only: this result is not cloud-qualified and cannot qualify a candidate.",
            *findings,
        ],
    )
