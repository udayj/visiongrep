"""Measure exactly one CLI child in a fresh helper process (isolated peak RSS)."""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path


def main():
    configuration = json.loads(Path(sys.argv[1]).read_text())
    if configuration.get("storage_quiescence_path") and sys.argv[2:] != ["--prepared"]:
        from quiescence import settle

        # Prepare before creating the fresh resource-accounting helper. The sync
        # subprocess must not contribute to the CLI child's CPU or peak RSS.
        for key in ("stdout", "stderr"):
            Path(configuration[key]).touch()
        storage = settle(Path(configuration["storage_quiescence_path"]))
        output = subprocess.check_output([
            sys.executable, __file__, sys.argv[1], "--prepared",
        ], text=True)
        metrics = json.loads(output)
        metrics["storage_quiescence"] = storage
        print(json.dumps(metrics))
        return
    stdout_path = Path(configuration["stdout"])
    stderr_path = Path(configuration["stderr"])
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        diagnostic = os.environ.get("VISIONGREP_BENCH_DIAGNOSTIC") == "1"
        epoch_ns = time.time_ns() if diagnostic else None
        monotonic_ns = time.monotonic_ns() if diagnostic else None
        started = time.perf_counter()
        if "persistent" in configuration:
            # Import as a package; harness/statistics.py must not shadow stdlib statistics.
            sys.path[0] = str(Path(__file__).resolve().parents[1])
            from harness.persistent import run

            metrics = run(configuration, out, err)
        else:
            child = subprocess.run(
                configuration["command"], stdout=out, stderr=err,
                env=os.environ, check=False,
            )
            metrics = {"wall_ms": (time.perf_counter() - started) * 1000,
                       "exit_code": child.returncode}
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    rss = usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
    extra = {}
    if diagnostic:
        extra["diagnostic"] = {
            "started_epoch_ns": epoch_ns,
            "started_monotonic_ns": monotonic_ns,
            "user_seconds": usage.ru_utime,
            "system_seconds": usage.ru_stime,
            "voluntary_context_switches": usage.ru_nvcsw,
            "involuntary_context_switches": usage.ru_nivcsw,
            "major_faults": usage.ru_majflt,
            "minor_faults": usage.ru_minflt,
            "input_blocks": usage.ru_inblock,
            "output_blocks": usage.ru_oublock,
        }
    print(
        json.dumps(
            {**metrics, "peak_rss_bytes": rss, **extra}
        )
    )


if __name__ == "__main__":
    main()
