"""Fixed-budget SQLite diagnostics; never eligible as foundation recordings.

Reuse the original scenario restoration, measurement helper, checks and uploads.
Only the second batch uses ptrace. Telemetry and live traces stay on tmpfs;
completed artifacts are copied to the run directory outside each measurement.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

from .scenarios import Scenario
from .statistics import summary
from .storage import BENCHMARKS, FOUNDATION, command

SYSTEM_FILES = (
    "stat", "diskstats", "meminfo", "vmstat", "loadavg", "locks",
    "pressure/cpu", "pressure/io", "pressure/memory",
)
TRACE_CALLS = (
    "fsync,fdatasync,sync,syncfs,pwrite64,pread64,read,write,"
    "fcntl,flock,nanosleep,clock_nanosleep,openat,close,"
    "unlink,unlinkat,rename,renameat,renameat2,ftruncate"
)
MAX_TELEMETRY_BYTES = 128 * 1024 * 1024


def read_proc(path: Path):
    try:
        return path.read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError) as error:
        # Processes can exit between enumeration and reading. Preserve missing coverage.
        return {"unavailable": type(error).__name__}


def snapshot(proc: Path = Path("/proc"), *, all_processes: bool = True) -> dict:
    result = {
        "epoch_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "all_processes": all_processes,
        "system": {name: read_proc(proc / name) for name in SYSTEM_FILES},
        "processes": {},
    }
    for path in proc.iterdir():
        if not path.name.isdigit():
            continue
        try:
            if not all_processes and path.stat().st_uid != os.getuid():
                continue
        except (FileNotFoundError, ProcessLookupError):
            continue
        row = {name: read_proc(path / name) for name in ("stat", "schedstat", "io", "wchan")}
        # Per-thread runqueue wait is necessary: inference uses multiple worker threads.
        try:
            row["threads"] = {
                task.name: read_proc(task / "schedstat")
                for task in (path / "task").iterdir()
            }
        except (FileNotFoundError, ProcessLookupError, PermissionError) as error:
            row["threads"] = {"unavailable": type(error).__name__}
        result["processes"][path.name] = row
    result["finished_monotonic_ns"] = time.monotonic_ns()
    return result


class Telemetry:
    def __init__(self, path: Path):
        self.path = path
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.collect, daemon=True)

    def collect(self):
        try:
            with self.path.open("w") as output:
                count = 0
                while not self.stop.is_set():
                    output.write(json.dumps(snapshot(all_processes=count % 10 == 0), separators=(",", ":")) + "\n")
                    output.flush()
                    count += 1
                    if output.tell() > MAX_TELEMETRY_BYTES:
                        raise RuntimeError("telemetry exceeded its 128 MiB tmpfs budget")
                    self.stop.wait(0.1)
        except Exception as error:
            self.error = error

    def check(self):
        if self.error is not None:
            raise RuntimeError("diagnostic telemetry failed") from self.error


def trace_command(args: list[str], destination: Path) -> list[str]:
    return [
        "prlimit", "--fsize=67108864:67108864", "--",
        "strace", "-f", "-ttt", "-T", "-yy", "-s", "128",
        "-e", "trace=" + TRACE_CALLS,
        "-e", "raw=read,pread64,write,pwrite64",
        "-o", str(destination), "--", *args,
    ]


def measure(run, binary: Path, cache: Path, corpus: dict):
    # Leave time for the bootstrap trap to upload raw evidence before hard termination.
    run.deadline -= 180
    if not all(shutil.which(name) for name in ("strace", "prlimit")) or not Path("/proc/diskstats").exists():
        raise ValueError("diagnostics require Linux /proc, strace and prlimit")
    mounts = Path("/proc/mounts").read_text().splitlines()
    if not any(line.split()[1:3] == ["/dev/shm", "tmpfs"] for line in mounts):
        raise ValueError("diagnostics require /dev/shm on tmpfs")
    if Path("/proc/sys/kernel/sched_schedstats").read_text().strip() != "1":
        raise ValueError("diagnostics require enabled kernel.sched_schedstats")
    if shutil.disk_usage("/dev/shm").free < 256 * 1024 * 1024:
        raise ValueError("diagnostics require at least 256 MiB free on tmpfs")
    run.report.update(
        verdict="diagnostic_incomplete",
        reasons=["Diagnostic only: untraced and traced samples are separate; no foundation acceptance."],
        diagnostics={
            "strace_version": command(["strace", "--version"]),
            "trace_calls": TRACE_CALLS,
            "telemetry_interval_seconds": 0.1,
            "telemetry_note": "Interval excludes collector work. Worker-user processes every snapshot; all processes every tenth. Actual timestamps and missing reads retained.",
            "mounts": "\n".join(mounts),
            "scheduler_stats_enabled": read_proc(Path("/proc/sys/kernel/sched_schedstats")),
        },
    )
    run.report["summaries"] = {}
    for batch in run.profile["batches"]:
        for name in run.profile["scenarios"]:
            run.progress(stage="diagnostic", batch=batch, scenario=name)
            rows = []
            run.report["samples"].setdefault(name, {})[batch] = rows
            root = run.directory / "scenarios" / name / batch
            root.mkdir(parents=True, exist_ok=False)
            scenario = None
            with tempfile.TemporaryDirectory(prefix="visiongrep-diagnostic-", dir="/dev/shm") as temporary:
                scratch = Path(temporary)
                telemetry = Telemetry(scratch / "telemetry.jsonl")
                telemetry.thread.start()

                def invoke(args, *, cwd=None, env=None):
                    telemetry.check()
                    is_measurement = len(args) > 1 and args[1] == str(BENCHMARKS / "harness/measurement.py")
                    active = is_measurement and Path(args[2]).parent.name.startswith("sample-")
                    trace = scratch / "syscalls.log"
                    if active:
                        env = dict(env or os.environ, VISIONGREP_BENCH_DIAGNOSTIC="1")
                    if active and batch == "strace":
                        args = trace_command(args, trace)
                    try:
                        return run.invoke(args, cwd=cwd, env=env)
                    finally:
                        if trace.exists():
                            # The helper has exited; these writes are outside its wall clock.
                            shutil.copyfile(trace, root / "observations" / f"sample-{len(rows):04d}" / "syscalls.log")
                            trace.unlink()

                try:
                    scenario = Scenario(root, binary, cache, corpus, name, invoke)
                    for index in range(run.profile["samples"]):
                        telemetry.check()
                        run.progress(batch=batch, scenario=name, sample=index, total_samples=run.profile["samples"])
                        started = time.time_ns()
                        row = scenario.sample(index)
                        row["diagnostic"]["setup_started_epoch_ns"] = started
                        rows.append(row)
                        if row["timing"]["environment"]["commit"] != FOUNDATION:
                            raise ValueError("diagnostic binary differs from pinned foundation")
                        run.save()
                        if not row["behavior"]["passed"]:
                            raise ValueError(f"diagnostic behavior failed: {name}/{batch}/{index}")
                    telemetry.check()
                    run.report["summaries"].setdefault(name, {})[batch] = {
                        "wall_ms": summary([row["wall_ms"] for row in rows]),
                        "phases_ms": {
                            phase: summary([
                                next((p["elapsed_ms"] for p in row["timing"]["phases"] if p["phase"] == phase), 0)
                                for row in rows
                            ])
                            for phase in ("stale_entry_reconciliation", "database_writes")
                        },
                    }
                finally:
                    telemetry.stop.set()
                    telemetry.thread.join()
                    shutil.copyfile(telemetry.path, root / "telemetry.jsonl")
                    if scenario is not None:
                        scenario.cleanup()
                        run.discard_inputs(scenario)
                telemetry.check()
            run.save()
            if os.environ.get("BENCH_S3_RUN"):
                relative = root.relative_to(run.directory)
                command([
                    "aws", "s3", "sync", str(root),
                    os.environ["BENCH_S3_RUN"] + "/" + str(relative),
                    "--only-show-errors",
                ], timeout=120)
    run.report["verdict"] = "diagnostic_complete"
