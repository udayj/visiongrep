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
    stdout_path = Path(configuration["stdout"])
    stderr_path = Path(configuration["stderr"])
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        diagnostic = os.environ.get("VISIONGREP_BENCH_DIAGNOSTIC") == "1"
        epoch_ns = time.time_ns() if diagnostic else None
        monotonic_ns = time.monotonic_ns() if diagnostic else None
        started = time.perf_counter()
        child = subprocess.run(
            configuration["command"],
            stdout=out,
            stderr=err,
            env=os.environ,
            check=False,
        )
        elapsed = (time.perf_counter() - started) * 1000
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
            {"wall_ms": elapsed, "peak_rss_bytes": rss, "exit_code": child.returncode, **extra}
        )
    )


if __name__ == "__main__":
    main()
