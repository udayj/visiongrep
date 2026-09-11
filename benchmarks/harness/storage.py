"""Immutable inputs, atomic reports, and bounded subprocess execution."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks"
FOUNDATION = "v0.2.0"


def read_json(path: Path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def harness_digest() -> str:
    value = hashlib.sha256()
    for path in sorted(BENCHMARKS.rglob("*")):
        if path.suffix not in (".py", ".json") or "__pycache__" in path.parts:
            continue
        value.update(str(path.relative_to(BENCHMARKS)).encode())
        value.update(bytes.fromhex(digest(path)))
    return value.hexdigest()


def outside_repository(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError(f"runtime files must be outside the repository: {path}")
    return path


def command(args: list[str], *, timeout: float = 900, env=None, cwd=None) -> str:
    """Kill the whole child process group on timeout or cancellation."""
    with subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=cwd,
        start_new_session=True,
    ) as child:
        try:
            stdout, stderr = child.communicate(timeout=timeout)
        except BaseException:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise
        if child.returncode:
            raise RuntimeError(
                f"command failed ({child.returncode}): {args[0]}\n{stderr[-6000:]}"
            )
        return stdout


def now() -> float:
    return time.time()
