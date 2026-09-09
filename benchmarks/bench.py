#!/usr/bin/env python3
"""VisionGrep benchmark commands. Run `python3 benchmarks/bench.py --help`."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

from harness import assets, cloud, report, runner
from harness.storage import (
    BENCHMARKS,
    FOUNDATION,
    ROOT,
    command,
    digest,
    outside_repository,
    read_json,
    write_json,
)

DEFAULT_HOME = Path.home() / ".cache/visiongrep-bench"


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser(
        "prepare", help="download/verify reusable artifacts; no inference"
    )
    prepare.add_argument("--cache", type=Path, default=DEFAULT_HOME / "cache")
    prepare.add_argument("--corpus", choices=("500", "10000"), default="500")
    prepare.add_argument("--import-models", type=Path)
    for action in ("plan", "run"):
        run = sub.add_parser(
            action, help="estimate only" if action == "plan" else "start a benchmark"
        )
        run.add_argument(
            "--profile",
            choices=("local-quick", "cloud-standard", "cloud-scale"),
            default="local-quick",
        )
        run.add_argument(
            "--mode", choices=("compare", "validate", "record", "diagnose"), default="compare"
        )
        run.add_argument(
            "--validation-samples",
            type=int,
            choices=(3, 21),
            help="validation only: paired samples per scenario; 3 shortens cloud calibration",
        )
        run.add_argument("--candidate", default="HEAD")
        run.add_argument("--baseline", type=Path)
        run.add_argument("--cache", type=Path, default=DEFAULT_HOME / "cache")
        run.add_argument("--runs", type=Path, default=DEFAULT_HOME / "runs")
        run.add_argument("--max-hours", type=float, default=4)
        run.add_argument("--budget-usd", type=float, default=1)
        run.add_argument(
            "--cloud", type=Path, help="pinned EC2 settings; launches only with run"
        )
        run.add_argument(
            "--cloud-slot", type=int, choices=(1, 2, 3), default=1,
            help="independent cloud launch slot; budgets apply to each run",
        )
        run.add_argument("--detach", action="store_true")
        run.add_argument("--wait", action="store_true", help="cloud only: wait for termination and collect using this authenticated session")
    worker = sub.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--config", required=True, type=Path)
    aggregate = sub.add_parser(
        "foundation", help="freeze calibrated reference from >=3 record sessions"
    )
    aggregate.add_argument("runs", nargs="+", type=Path)
    aggregate.add_argument("--output", required=True, type=Path)
    reconcile = sub.add_parser(
        "cloud-reconcile", help="recover an expired interrupted cloud launch"
    )
    reconcile.add_argument("--cloud", required=True, type=Path)
    reconcile.add_argument("--cloud-slot", type=int, choices=(1, 2, 3), default=1)
    for action in ("status", "logs", "cancel", "report", "collect"):
        control = sub.add_parser(action)
        control.add_argument("run", type=Path)
    return root


def configuration(args) -> dict:
    if args.cloud_slot != 1 and not args.cloud:
        raise ValueError("--cloud-slot requires --cloud")
    if args.wait and (not args.cloud or args.detach):
        raise ValueError("--wait requires --cloud and cannot be combined with --detach")
    profile = read_json(BENCHMARKS / "profiles" / (args.profile + ".json"))
    if args.mode == "diagnose":
        if args.profile != "cloud-standard" or not args.cloud or args.baseline:
            raise ValueError("diagnose requires cloud-standard and --cloud, without --baseline")
        profile = profile | {
            "name": "sqlite-diagnostic",
            "samples": 100,
            "quality": False,
            "scenarios": ["deleted_1pct", "modified_query_image", "modified_1pct", "renamed_1pct"],
            "batches": ["untraced", "strace"],
        }
    if args.validation_samples is not None:
        if args.mode != "validate":
            raise ValueError("--validation-samples requires --mode validate")
        profile["samples"] = args.validation_samples
        profile["calibration_batches"] = (
            3 if args.profile == "local-quick" else (1 if args.validation_samples == 3 else 3)
        )
    if args.max_hours <= 0 or args.max_hours > 24 or args.budget_usd <= 0:
        raise ValueError("runtime must be in (0,24] hours and budget must be positive")
    cache = outside_repository(args.cache)
    runs = outside_repository(args.runs)
    selection = BENCHMARKS / "corpora" / f"coco-{profile['corpus_size']}.json"
    corpus = cache / "corpora" / (digest(selection) + ".json")
    sha = command(
        ["git", "rev-parse", "--verify", args.candidate + "^{commit}"], cwd=ROOT
    ).strip()
    tag = command(
        ["git", "rev-parse", "--verify", "benchmark-foundation-v1^{commit}"], cwd=ROOT
    ).strip()
    if tag != FOUNDATION:
        raise ValueError(
            "foundation tag moved; refusing to change the reference silently"
        )
    if args.mode in ("record", "validate", "diagnose"):
        sha = FOUNDATION
    hourly = 0.26 if args.cloud else 0
    seconds = int(
        min(
            args.max_hours * 3600,
            args.budget_usd / hourly * 3600 if hourly else float("inf"),
        )
    )
    if args.mode == "diagnose":
        seconds = min(seconds, 3600)
    return {
        "cloud_slot": args.cloud_slot,
        "schema_version": 1,
        "directory": str(
            runs / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
        ),
        "profile": profile,
        "candidate": sha,
        "mode": args.mode,
        "baseline": str(args.baseline.resolve()) if args.baseline else None,
        "cache": str(cache),
        "corpus": str(corpus),
        "corpus_sha256": digest(corpus) if corpus.exists() else None,
        "max_seconds": seconds,
        "command_timeout_seconds": seconds,
        "hourly_budget_usd": hourly,
        "budget_usd": args.budget_usd,
    }


def main():
    args = parser().parse_args()
    if args.action == "prepare":
        assets.prepare(
            args.cache,
            BENCHMARKS / "corpora" / f"coco-{args.corpus}.json",
            args.import_models,
        )
        print(f"Prepared immutable artifact cache: {args.cache}")
    elif args.action in ("plan", "run"):
        config = configuration(args)
        if args.action == "plan":
            print(
                json.dumps(
                    config
                    | {
                        "note": "cost ceiling, not runtime prediction; pilot durations are not recorded yet",
                        "maximum_estimated_cost_usd": config["max_seconds"]
                        / 3600
                        * config["hourly_budget_usd"],
                    },
                    indent=2,
                )
            )
            return
        if config["corpus_sha256"] is None:
            raise ValueError("run prepare before benchmarking")
        if config["mode"] == "compare" and not args.baseline:
            raise ValueError(
                "comparison requires --baseline; start with --mode validate"
            )
        if config["profile"]["cloud"] != bool(args.cloud):
            raise ValueError(
                "cloud profiles require --cloud; use local-quick on the MacBook"
            )
        directory = Path(config["directory"])
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "config.json", config)
        if args.cloud:
            handle = cloud.launch(config, args.cloud)
            print(json.dumps(handle, indent=2))
            print(directory, flush=True)
            if args.wait:
                while True:
                    state = cloud.status(directory)
                    print(json.dumps(state), flush=True)
                    if state["instance_state"] == "terminated":
                        break
                    time.sleep(20)
                cloud.collect(directory)
                result = read_json(directory / "remote/report.json")["verdict"]
                print(f"Collected: {directory / 'remote'} ({result})", flush=True)
                if result in ("invalid", "cancelled", "validation_failed", "diagnostic_incomplete"):
                    raise SystemExit(2)
        elif args.detach:
            with (directory / "worker.log").open("w") as log:
                subprocess.Popen(
                    [
                        sys.executable,
                        str(BENCHMARKS / "bench.py"),
                        "worker",
                        "--config",
                        str(directory / "config.json"),
                    ],
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
        else:
            result = runner.worker(directory / "config.json")
            print(result)
            if result in ("invalid", "cancelled", "validation_failed"):
                print(directory)
                raise SystemExit(2)
        print(directory)
    elif args.action == "worker":
        result = runner.worker(args.config)
        print(result)
        if result in ("invalid", "cancelled", "validation_failed"):
            raise SystemExit(2)
    elif args.action == "foundation":
        result = runner.foundation(args.runs, outside_repository(args.output))
        if result:
            print(result)
    elif args.action == "cloud-reconcile":
        cloud.reconcile(args.cloud, slot=args.cloud_slot)
    else:
        directory = args.run.expanduser().resolve()
        if args.action == "status":
            value = (
                cloud.status(directory)
                if (directory / "cloud.json").exists()
                else read_json(directory / "status.json")
            )
            if (directory / "heartbeat.json").exists():
                value["heartbeat"] = read_json(directory / "heartbeat.json")
            print(json.dumps(value, indent=2))
        elif args.action == "cancel":
            if (directory / "cloud.json").exists():
                cloud.collect(directory, cancel=True)
            else:
                (directory / "cancel").touch()
        elif args.action == "collect":
            cloud.collect(directory)
        elif args.action == "logs":
            path = directory / "worker.log"
            if not path.exists():
                path = directory / "remote/bootstrap.log"
            print(path.read_text()[-16000:])
        elif args.action == "report":
            if (
                not (directory / "report.json").exists()
                and (directory / "remote/report.json").exists()
            ):
                directory /= "remote"
            print(report.render(directory))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        print(f"benchmark error: {error}", file=sys.stderr)
        raise SystemExit(2)
