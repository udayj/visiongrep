"""Read every retained local report, including failures; never rewrite evidence.

Usage: python3 benchmarks/audit_local.py /path/to/runs > /path/to/audit.json
Historical five-sample sessions cannot test the new nine-sample policy. Report
the raw-noise diagnostic separately from eligibility instead of padding batches.
"""

import json
import sys
from pathlib import Path

from harness import local_screening
from harness.statistics import summary
from harness.storage import digest, read_json


def audit(root: Path) -> dict:
    reports = []
    missing = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        config_path = directory / "config.json"
        report_path = directory / "report.json"
        report = read_json(report_path) if report_path.exists() else None
        config = report["config"] if report else read_json(config_path)
        if not local_screening.enabled(config["profile"]):
            continue
        if report is None:
            missing.append(directory.name)
            continue
        scenarios = {}
        for name, roles in report["samples"].items():
            scenarios[name] = {}
            for role, rows in roles.items():
                times = [row["wall_ms"] for row in rows]
                phases = sorted(
                    {p["phase"] for row in rows for p in row["timing"]["phases"]}
                )
                scenarios[name][role] = {
                    "wall_ms": times,
                    "application_wall_ms": [
                        row["timing"]["total_wall_ms"] for row in rows
                    ],
                    "phases_ms": {
                        phase: [
                            sum(
                                p["elapsed_ms"]
                                for p in row["timing"]["phases"]
                                if p["phase"] == phase
                            )
                            for row in rows
                        ]
                        for phase in phases
                    },
                    "summary": summary(times) if times else None,
                    "timing_uncertainty": (
                        local_screening.uncertainty(times) if times else ["incomplete"]
                    ),
                    "behavior_passed": all(
                        row.get("behavior", {}).get("passed", False) for row in rows
                    ),
                }
        reports.append(
            {
                "run": directory.name,
                "report_sha256": digest(report_path),
                "original_verdict": report["verdict"],
                "original_reasons": report["reasons"],
                "mode": config["mode"],
                "profile": config["profile"],
                "contract": report.get("contract"),
                "eligibility": (
                    "fresh recordings required"
                    if config["profile"].get("screening_policy")
                    != local_screening.POLICY
                    else "inspect full recording contract"
                ),
                "scenarios": scenarios,
            }
        )
    return {
        "policy": local_screening.POLICY,
        "reports": reports,
        "missing_local_reports": missing,
    }


if __name__ == "__main__":
    print(
        json.dumps(audit(Path(sys.argv[1])), indent=2, sort_keys=True, allow_nan=False)
    )
