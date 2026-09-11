"""Sequential JSONL round trips; mutations and behavior checks are outside timing."""

import json
import os
import selectors
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from . import behavior
from .storage import BENCHMARKS, write_json


def sample(scenario, index):
    output = scenario.root / "observations" / f"sample-{index:04d}"
    output.mkdir(parents=True)
    config = {
        "command": [str(scenario.binary), "serve", "--stdio", "--index-path",
                    str(scenario.index), str(scenario.images)],
        "stdout": str(output / "responses.jsonl"),
        "stderr": str(output / "stderr.log"),
        "persistent": {
            "root": str(scenario.root), "cache": str(scenario.cache),
            "name": scenario.name, "rows": scenario.rows,
            "changed": scenario.changed, "original_stats": scenario.original_stats,
            "query": f"persistent bicycle near water {index}",
        },
    }
    if scenario.environment.get("BENCH_AMI"):
        config["storage_quiescence_path"] = str(scenario.root)
    write_json(output / "invocation.json", config)
    metrics = json.loads(scenario.invoke(
        [sys.executable, str(BENCHMARKS / "harness/measurement.py"),
         str(output / "invocation.json")], env=scenario.environment,
    ))
    if metrics["exit_code"] != 0:
        raise RuntimeError(f"serve failed: {(output / 'stderr.log').read_text()[-3000:]}")
    metrics["timing"] = {"phases": []}  # Serve does not expose internal phase timings.
    metrics["index_bytes"] = scenario.index.stat().st_size
    metrics["index_bytes_per_image"] = metrics["index_bytes"] / scenario.count
    return metrics


def exchange(child, request):
    payload = (json.dumps(request) + "\n").encode()
    started = time.perf_counter()
    child.stdin.write(payload)
    child.stdin.flush()
    # Read raw bytes so a partial line cannot make readline block past the timeout.
    line = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(child.stdout, selectors.EVENT_READ)
        while not line.endswith(b"\n"):
            remaining = 120 - (time.perf_counter() - started)
            if remaining <= 0 or not selector.select(remaining):
                raise TimeoutError("serve response timed out")
            chunk = os.read(child.stdout.fileno(), 65536)
            if not chunk:
                raise ValueError("serve closed stdout before responding")
            line.extend(chunk)
            if len(line) > 1024 * 1024:
                raise ValueError("serve response exceeds 1 MiB")
    elapsed = (time.perf_counter() - started) * 1000
    response = json.loads(line)
    if response.get("id") != request["id"] or not isinstance(response.get("results"), list):
        raise ValueError(f"invalid serve response: {response}")
    return elapsed, response


def run(config, out, err):
    settings = config["persistent"]
    root = Path(settings["root"])
    scenario = SimpleNamespace(
        root=root, images=root / "images", index=root / "index.db",
        rows=settings["rows"], changed=settings["changed"], name=settings["name"],
    )
    requests = []
    started = time.perf_counter()
    child = subprocess.Popen(config["command"], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=err, bufsize=0)
    startup_ms = (time.perf_counter() - started) * 1000
    try:
        for number in range(5):
            # Text: first use, three novel queries, then a cached control.
            # Updates: first text, first vision use, two retained-vision updates,
            # then an unchanged cached control. Each replacement changes pixels.
            if settings["name"] == "persistent_updates" and 1 <= number <= 3:
                modified = number % 2 == 1
                for row in scenario.rows[:scenario.changed]:
                    source = scenario.rows[-1] if modified else row
                    path = scenario.images / row["file_name"]
                    shutil.copyfile(Path(settings["cache"]) / "objects" / source["sha256"], path)
                    stamp = settings["original_stats"][row["file_name"]] + number * 1_000_000_000
                    os.utime(path, ns=(stamp, stamp))
                scenario.name = "modified_1pct" if modified else "persistent_updates"
            query = settings["query"]
            if settings["name"] == "persistent_text":
                query += f" novel {min(number, 3)}"
            elapsed, response = exchange(child, {
                "id": number, "query": query, "top": 10, "threshold": -1,
            })
            if number == 0:
                elapsed += startup_ms
            out.write((json.dumps(response) + "\n").encode())
            results = [item | {"path": str(Path(item["path"]).relative_to(scenario.images))}
                       for item in response["results"]]
            check = behavior.check(scenario, {"results": results}, None, False, 10)
            if number == 4 and results != requests[-1]["results"]:
                check["reasons"].append("cached control differs from the preceding query")
                check["passed"] = False
            requests.append({"id": number, "wall_ms": elapsed, "results": results,
                             "behavior": check})
        child.stdin.close()
        exit_code = child.wait(timeout=120)
        if child.stdout.read(1):
            raise ValueError("serve emitted unexpected extra output")
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()
        child.stdout.close()
        child.stdin.close()
    reasons = [f"request {row['id']}: {reason}" for row in requests
               for reason in row["behavior"]["reasons"]]
    steady = requests[1:4] if settings["name"] == "persistent_text" else requests[2:4]
    metrics = {
        "wall_ms": statistics.median(row["wall_ms"] for row in steady),
        "first_response_ms": requests[0]["wall_ms"],
        "cached_response_ms": requests[4]["wall_ms"],
        "requests": requests, "results": requests[-1]["results"],
        "behavior": {"passed": not reasons, "reasons": reasons}, "exit_code": exit_code,
    }
    if settings["name"] == "persistent_updates":
        metrics["first_update_ms"] = requests[1]["wall_ms"]
    return metrics
