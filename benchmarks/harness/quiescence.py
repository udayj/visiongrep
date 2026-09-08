"""Flush setup writes and require an observed quiet Linux filesystem before timing."""

import os
import subprocess
import time
from pathlib import Path

TIMEOUT_SECONDS = 30
QUIET_SECONDS = 0.25
MAX_DIRTY_KIB = 16 * 1024


def storage_state(device: Path) -> dict:
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("Dirty", "Writeback"):
            memory[key] = int(value.split()[0])
    fields = [int(value) for value in device.read_text().split()]
    return {
        "dirty_kib": memory["Dirty"],
        "writeback_kib": memory["Writeback"],
        "reads_completed": fields[0],
        "writes_completed": fields[4],
        "sectors_written": fields[6],
        "in_flight": fields[8],
        "io_ms": fields[9],
    }


def quiet(previous: dict, current: dict) -> bool:
    return (
        current["dirty_kib"] <= MAX_DIRTY_KIB
        and current["writeback_kib"] == 0
        and previous["in_flight"] == current["in_flight"] == 0
        and all(previous[key] == current[key] for key in (
            "reads_completed", "writes_completed", "sectors_written", "io_ms",
        ))
    )


def settle(path: Path) -> dict:
    started = time.monotonic()
    device_id = path.stat().st_dev
    device = Path(f"/sys/dev/block/{os.major(device_id)}:{os.minor(device_id)}/stat")
    before = storage_state(device)
    # GNU sync -f calls syncfs on this filesystem, including directory metadata
    # and staged files that are not themselves part of the measured command.
    subprocess.run(["sync", "-f", str(path)], check=True, timeout=TIMEOUT_SECONDS)
    flushed = time.monotonic()
    previous = storage_state(device)
    quiet_since = flushed
    while time.monotonic() - started < TIMEOUT_SECONDS:
        time.sleep(0.05)
        current = storage_state(device)
        now = time.monotonic()
        if not quiet(previous, current):
            quiet_since = now
        elif now - quiet_since >= QUIET_SECONDS:
            return {
                "policy": "linux-syncfs-quiet-v1",
                "device_stat": str(device),
                "sync_ms": (flushed - started) * 1000,
                "total_ms": (now - started) * 1000,
                "required_quiet_seconds": QUIET_SECONDS,
                "max_dirty_kib": MAX_DIRTY_KIB,
                "before": before,
                "after": current,
            }
        previous = current
    raise TimeoutError(f"storage did not settle within {TIMEOUT_SECONDS}s: {current}")
