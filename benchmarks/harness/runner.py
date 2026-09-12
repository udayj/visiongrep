"""Build immutable commits and execute bounded, paired benchmark sessions."""

from __future__ import annotations

import fcntl
import os
import platform
import shutil
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

from . import behavior, calibration, local_screening, quality
from .assets import verify
from .builds import build
from .report import render
from .scenarios import CALIBRATION, SCENARIOS, Scenario
from .statistics import paired, summary, verdict
from .storage import (
    BENCHMARKS,
    command,
    digest,
    harness_digest,
    read_json,
    write_json,
)


def environment(profile: dict) -> dict:
    hardware = {}
    if platform.system() == "Darwin":
        for key in ("hw.model", "hw.memsize", "machdep.cpu.brand_string"):
            hardware[key] = command(["sysctl", "-n", key], timeout=10).strip()
    elif platform.system() == "Linux":
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                if key.strip() in ("model name", "microcode", "flags"):
                    hardware.setdefault(key.strip(), value.strip())
        memory = Path("/proc/meminfo").read_text().splitlines()[0].split()[1]
        hardware["memory_gib_floor"] = int(memory) // (1024 * 1024)
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "hostname": socket.gethostname(),
        "hardware": hardware,
        "python": platform.python_version(),
        "profile": profile["name"],
        "instance_id": os.environ.get("BENCH_INSTANCE_ID"),
        "ami": os.environ.get("BENCH_AMI"),
        "region": os.environ.get("AWS_DEFAULT_REGION"),
        "filesystem_cache": "warm by explicit pre-read; OS residency not guaranteed",
        "rayon": "application-owned pool, at most four workers",
        "ort_threads": "application defaults",
    }


def contract(config: dict, profile: dict, corpus: dict) -> dict:
    env = environment(profile)
    return calibration.without_microcode(
        {
            "foundation_sha": config["foundation_sha"],
            "harness_sha256": harness_digest(),
            "corpus_sha256": config["corpus_sha256"],
            "profile": profile,
            "quality_sha256": digest(BENCHMARKS / "corpora/quality-500.json"),
            "artifacts_sha256": digest(BENCHMARKS / "profiles/artifacts.json"),
            "environment": {
                key: env[key]
                for key in (
                    "system",
                    "release",
                    "machine",
                    "cpu_count",
                    "python",
                    "ami",
                    "hardware",
                )
            },
        }
    )


class Run:
    def __init__(self, config: dict):
        self.config = config
        self.directory = Path(config["directory"])
        self.profile = config["profile"]
        self.started = float(os.environ.get("BENCH_LAUNCH_EPOCH", time.time()))
        self.deadline = self.started + config["max_seconds"]
        self.state = {"stage": "starting", "started": self.started, "pid": os.getpid()}
        self.report = {
            "schema_version": 1,
            "verdict": "incomplete",
            "reasons": [],
            "config": config,
            "environment": {},
            "samples": {},
            "calibration": {},
            "quality": {},
            "comparisons": {},
        }
        self.report["schema_version"] = 4
        self.report["timing_screen"] = {"method": calibration.METHOD, "reasons": []}
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)

    def heartbeat(self):
        while not self.stop.wait(5):
            write_json(
                self.directory / "heartbeat.json",
                {"time": time.time(), "pid": os.getpid()},
            )

    def progress(self, **fields):
        self.state.update(fields)
        self.state.update(
            updated=time.time(),
            elapsed_seconds=time.time() - self.started,
            estimated_cost_usd=(time.time() - self.started)
            / 3600
            * self.config["hourly_budget_usd"],
        )
        write_json(self.directory / "status.json", self.state)
        if os.environ.get("BENCH_S3_RUN"):
            command(
                [
                    "aws",
                    "s3",
                    "cp",
                    str(self.directory / "status.json"),
                    os.environ["BENCH_S3_RUN"] + "/status.json",
                    "--only-show-errors",
                ],
                timeout=30,
            )
        if (self.directory / "cancel").exists():
            raise InterruptedError("run cancelled")
        if time.time() >= self.deadline:
            raise TimeoutError("run deadline reached")

    def invoke(self, args, *, cwd=None, env=None):
        self.progress()
        # Poll only process state and cancellation; stdout/stderr are files to avoid pipe deadlocks.
        logs = self.directory / "command-logs"
        logs.mkdir(exist_ok=True)
        token = str(time.time_ns())
        with (
            (logs / (token + ".out")).open("w+") as out,
            (logs / (token + ".err")).open("w+") as err,
        ):
            with subprocess.Popen(
                args, stdout=out, stderr=err, cwd=cwd, env=env, start_new_session=True
            ) as child:
                try:
                    until = min(
                        self.deadline,
                        time.time() + self.config["command_timeout_seconds"],
                    )
                    while child.poll() is None:
                        if (self.directory / "cancel").exists():
                            raise InterruptedError("run cancelled")
                        if time.time() >= until:
                            raise TimeoutError("command or run deadline reached")
                        time.sleep(0.1)
                except BaseException:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                    raise
                out.seek(0)
                err.seek(0)
                if child.returncode:
                    raise RuntimeError(
                        f"command failed ({child.returncode}): {args[0]}\n{err.read()[-4000:]}"
                    )
                return out.read()

    def save(self):
        write_json(self.directory / "report.json", self.report)
        if os.environ.get("BENCH_S3_RUN"):
            # Checkpoint between samples so forced termination retains completed measurements.
            command(
                [
                    "aws",
                    "s3",
                    "cp",
                    str(self.directory / "report.json"),
                    os.environ["BENCH_S3_RUN"] + "/report.json",
                    "--only-show-errors",
                ],
                timeout=30,
            )

    def execute(self):
        self.thread.start()
        try:
            self.perform()
            self.progress(stage="complete")
        except (Exception, KeyboardInterrupt) as error:
            self.report["verdict"] = (
                "cancelled"
                if isinstance(error, (InterruptedError, KeyboardInterrupt))
                else "invalid"
            )
            self.report["reasons"] = [str(error)]
            self.state.update(
                stage=self.report["verdict"], error=str(error), updated=time.time()
            )
            write_json(self.directory / "status.json", self.state)
        finally:
            self.stop.set()
            self.thread.join()
            self.report["elapsed_seconds"] = time.time() - self.started
            self.report["estimated_cost_usd"] = (
                self.report["elapsed_seconds"] / 3600 * self.config["hourly_budget_usd"]
            )
            try:
                self.save()
            except (RuntimeError, OSError) as error:
                self.report["verdict"] = "invalid"
                self.report["reasons"].append(f"final report upload failed: {error}")
                write_json(self.directory / "report.json", self.report)
            render(self.directory)
        return self.report["verdict"]

    def perform(self):
        self.report["environment"] = environment(self.profile)
        config = self.config
        corpus = read_json(Path(config["corpus"]))
        if digest(Path(config["corpus"])) != config["corpus_sha256"]:
            raise ValueError("prepared corpus lock changed after launch")
        cache = Path(config["cache"])
        if self.profile["cloud"]:
            if (
                platform.system() != "Linux"
                or platform.machine() != "x86_64"
                or os.cpu_count() != 4
            ):
                raise ValueError("cloud profile requires Linux x86_64 with four CPUs")
            if not os.environ.get("BENCH_AMI"):
                raise ValueError("cloud profile must run on the configured EC2 worker")
        elif platform.system() != "Darwin":
            raise ValueError("local-quick is the MacBook profile")
        self.progress(stage="verifying-inputs")
        verify(cache, corpus)
        self.progress(stage="building")
        build_cache = cache / "builds"
        commits = {"foundation": config["foundation_sha"]}
        if config["mode"] not in ("record", "diagnose"):
            commits["candidate"] = config["candidate"]
        binaries, identities, settings = {}, {}, {}
        for role, revision in commits.items():
            source = None
            if config.get("sources"):
                source = Path(config["sources"][role])
                if config["source_commits"][role] != revision:
                    raise ValueError("source bundle commit identity differs")
            binaries[role], identities[role], settings[role] = build(
                revision, build_cache, self.invoke, source=source
            )
        self.report["binaries"] = {
            key: {
                "commit": identities[key],
                "sha256": digest(path),
                "build_environment": settings[key],
            }
            for key, path in binaries.items()
        }
        if any(value != settings["foundation"] for value in settings.values()):
            raise ValueError("candidate build environment differs from foundation")
        self.report["contract"] = contract(config, self.profile, corpus) | {
            "build_environment": settings["foundation"]
        }
        baseline = (
            read_json(Path(config["baseline"])) if config.get("baseline") else None
        )
        if config["mode"] == "compare":
            if baseline is None:
                raise ValueError(
                    "comparison requires a calibrated foundation; run validate or record first"
                )
            if calibration.measurement_contract(baseline["contract"]) != calibration.measurement_contract(self.report["contract"]):
                raise ValueError(
                    "foundation measurement contract differs; recalibration is required"
                )
            if baseline.get("analysis_method") != calibration.METHOD:
                raise ValueError("reanalyze foundation with the current analysis method")
            if baseline.get("verdict") != "calibrated":
                raise ValueError(
                    "foundation is not calibrated; reaggregate recording sessions"
                )
        if identities["foundation"] != config["foundation_sha"]:
            raise ValueError("wrong foundation binary identity")
        if config["mode"] == "diagnose":
            from .diagnostics import measure

            measure(self, binaries["foundation"], cache, corpus)
            return
        if config["mode"] == "validate" and identities["candidate"] != config["foundation_sha"]:
            raise ValueError("validation must compare foundation against itself")
        if config["mode"] != "record":
            self.calibrate(binaries["foundation"], cache, corpus, baseline)
        self.measure(binaries, identities, cache, corpus)

    def calibrate(self, binary: Path, cache: Path, corpus: dict, baseline):
        self.progress(stage="calibrating")
        calibration_samples = 3 * self.profile.get("calibration_batches", 3)
        for name in calibration_scenarios(self.profile):
            scenario = Scenario(
                self.directory / "calibration" / name,
                binary,
                cache,
                corpus,
                name,
                self.invoke,
            )
            values = []
            try:
                for sample in range(calibration_samples):
                    self.progress(
                        scenario=name, sample=sample, total_samples=calibration_samples
                    )
                    values.append(scenario.sample(sample))
                    self.report["calibration"][name] = values
                    self.save()
                    if not values[-1]["behavior"]["passed"]:
                        raise ValueError(
                            f"foundation behavior failed during calibration: {name}: {values[-1]['behavior']['reasons']}"
                        )
            finally:
                scenario.cleanup()
                self.discard_inputs(scenario)
            times = [x["wall_ms"] for x in values]
            info = calibration.estimate([times])
            self.report["timing_screen"].setdefault("calibration", {})[name] = info
            reasons = list(info["reasons"])
            if baseline:
                lo, hi = baseline["calibration_bounds"][name]
                if not lo <= info["median_ms"] <= hi:
                    reasons.append("foundation median drift outside reference 10% margin")
            self.report["timing_screen"]["reasons"].extend(
                f"calibration/{name}: {reason}" for reason in reasons
            )

    def measure(self, binaries: dict, identities: dict, cache: Path, corpus: dict):
        for name in self.profile["scenarios"]:
            self.progress(stage="measuring", scenario=name)
            instances = {}
            samples = {key: [] for key in binaries}
            self.report["samples"][name] = samples
            try:
                for key, binary in binaries.items():
                    instances[key] = Scenario(
                        self.directory / "scenarios" / name / key,
                        binary,
                        cache,
                        corpus,
                        name,
                        self.invoke,
                    )
                for sample in range(self.profile["samples"]):
                    order = (
                        list(binaries) if sample % 2 == 0 else list(reversed(binaries))
                    )
                    for key in order:
                        self.progress(
                            binary=key,
                            sample=sample,
                            total_samples=self.profile["samples"],
                        )
                        result = instances[key].sample(sample)
                        expected_sha = identities[key]
                        if result["timing"]["environment"]["commit"] != expected_sha:
                            raise ValueError(
                                "binary timing metadata disagrees with build identity"
                            )
                        samples[key].append(result)
                        self.save()
                    if "candidate" in instances:
                        samples["candidate"][-1]["behavior_comparison"] = (
                            behavior.compare(
                                samples["foundation"][-1],
                                samples["candidate"][-1],
                                instances["foundation"].index,
                                instances["candidate"].index,
                            )
                        )
                        self.save()
            finally:
                for instance in instances.values():
                    instance.cleanup()
                    self.discard_inputs(instance)
        self.progress(stage="quality")
        quality_scenarios = {}
        if self.profile["quality"]:
            spec = read_json(BENCHMARKS / "corpora/quality-500.json")
            if corpus["name"] != spec["corpus"]:
                raise ValueError("quality judgments do not match the corpus")
            try:
                for key, binary in binaries.items():
                    self.progress(binary=key)
                    scenario = Scenario(
                        self.directory / "quality" / key,
                        binary,
                        cache,
                        corpus,
                        "quality",
                        self.invoke,
                    )
                    quality_scenarios[key] = scenario
                    self.report["quality"][key] = quality.run(
                        scenario, spec, self.progress
                    )
                    self.save()
                if "candidate" in binaries:
                    self.report["quality_comparison"] = quality.compare(
                        self.report["quality"]["foundation"],
                        self.report["quality"]["candidate"],
                        quality_scenarios["foundation"].index,
                        quality_scenarios["candidate"].index,
                    )
            finally:
                for instance in quality_scenarios.values():
                    instance.cleanup()
                    self.discard_inputs(instance)
        self.summarize()

    def discard_inputs(self, scenario):
        for name in ("images", "cache"):
            shutil.rmtree(scenario.root / name)
        scenario.external.unlink(missing_ok=True)

    def summarize(self):
        config = self.config
        roles = (
            {"foundation"}
            if config["mode"] == "record"
            else {"foundation", "candidate"}
        )
        if set(self.report["samples"]) != set(self.profile["scenarios"]):
            raise ValueError("incomplete or unexpected measurement scenarios")
        for name in self.profile["scenarios"]:
            observations = self.report["samples"].get(name, {})
            if set(observations) != roles or any(
                len(rows) != self.profile["samples"]
                for rows in observations.values()
            ):
                raise ValueError(f"incomplete measurements for {name}")
        self.report["analysis_method"] = calibration.METHOD
        self.report["summaries"] = {}
        self.report["resource_comparisons"] = {}
        behavior_failures = []
        for name, by_binary in self.report["samples"].items():
            self.report["summaries"][name] = {}
            for key, observations in by_binary.items():
                for number, row in enumerate(observations):
                    checks = [
                        row.get(
                            "behavior",
                            {
                                "passed": False,
                                "reasons": ["missing scenario behavior check"],
                            },
                        )
                    ]
                    if key == "candidate":
                        checks.append(
                            row.get(
                                "behavior_comparison",
                                {
                                    "passed": False,
                                    "reasons": ["missing paired behavior check"],
                                },
                            )
                        )
                    for check in checks:
                        if not check["passed"]:
                            behavior_failures.extend(
                                f"{name}/{key}/{number}: {reason}"
                                for reason in check.get("reasons")
                                or ["scenario behavior check failed"]
                            )
                fields = (
                    "wall_ms",
                    "first_response_ms",
                    "first_update_ms",
                    "cached_response_ms",
                    "peak_rss_bytes",
                    "index_bytes",
                    "index_bytes_per_image",
                    "images_per_second",
                )
                aggregated = {
                    field: summary([row[field] for row in observations])
                    for field in fields
                    if field in observations[0]
                }
                phases = sorted(
                    {
                        p["phase"]
                        for row in observations
                        for p in row["timing"]["phases"]
                    }
                )
                aggregated["phases_ms"] = {
                    phase: summary(
                        [
                            sum(
                                p["elapsed_ms"]
                                for p in row["timing"]["phases"]
                                if p["phase"] == phase
                            )
                            for row in observations
                        ]
                    )
                    for phase in phases
                }
                self.report["summaries"][name][key] = aggregated
            if "candidate" in by_binary and len(by_binary["foundation"]) >= 9 and len(by_binary["foundation"]) % 3 == 0:
                self.report["comparisons"][name] = paired(
                    [x["wall_ms"] for x in by_binary["foundation"]],
                    [x["wall_ms"] for x in by_binary["candidate"]],
                    alpha=0.05 / len(self.profile["scenarios"]),
                )
                for field in ("peak_rss_bytes", "index_bytes"):
                    if (
                        field in by_binary["foundation"][0]
                        and field in by_binary["candidate"][0]
                    ):
                        self.report["resource_comparisons"][name + ":" + field] = (
                            paired(
                                [x[field] for x in by_binary["foundation"]],
                                [x[field] for x in by_binary["candidate"]],
                                alpha=0.05 / (2 * len(self.profile["scenarios"])),
                            )
                        )
        if config["mode"] == "record":
            self.report.update(
                verdict="foundation_recorded",
                reasons=[
                    "aggregate at least three independent sessions before comparison"
                ],
            )
        elif config["mode"] == "validate":
            okay = self.report.get("quality_comparison", {}).get(
                "passed", not self.profile["quality"]
            )
            drift = any(
                abs(item["improvement"]) > 0.05
                for item in self.report["comparisons"].values()
            )
            self.report.update(
                verdict=(
                    "validation_passed" if okay and not drift else "validation_failed"
                ),
                reasons=["same-commit comparison; not an improvement claim"],
            )
        else:
            result, reasons = verdict(
                self.report["comparisons"],
                self.report.get("quality_comparison", {}).get(
                    "passed", not self.profile["quality"]
                ),
                True,
                required_scenarios=SCENARIOS,
            )
            resource_failures = [
                "memory/index regression: " + name
                for name, item in self.report["resource_comparisons"].items()
                if item["interval"][1] < -0.05
            ]
            if resource_failures:
                reasons = (
                    resource_failures
                    if result == "qualifies"
                    else reasons + resource_failures
                )
                result = "does_not_qualify"
            elif result == "qualifies":
                required_resources = {
                    name + ":" + field
                    for name in SCENARIOS
                    for field in ("peak_rss_bytes", "index_bytes")
                    if field != "index_bytes" or name != "no_cache"
                }
                missing_resources = sorted(
                    required_resources - self.report["resource_comparisons"].keys()
                )
                uncertain = [
                    name
                    for name, item in self.report["resource_comparisons"].items()
                    if item["interval"][0] < -0.05
                ]
                if missing_resources:
                    result, reasons = (
                        "inconclusive",
                        [
                            "missing resource check: " + name
                            for name in missing_resources
                        ],
                    )
                elif uncertain:
                    result, reasons = (
                        "inconclusive",
                        [
                            "cannot exclude memory/index regression: " + name
                            for name in uncertain
                        ],
                    )
                else:
                    reasons.append(
                        "quality, behavior, memory, and index-size checks passed"
                    )
            self.report.update(verdict=result, reasons=reasons)
            if self.profile["name"] != "cloud-standard" and result not in (
                "does_not_qualify",
                "invalid",
            ):
                self.report.update(
                    verdict="inconclusive",
                    reasons=reasons
                    + [
                        "screening profile; qualification requires the complete cloud-standard suite"
                    ],
                )
        self.report["behavior_comparison"] = {
            "passed": not behavior_failures,
            "reasons": behavior_failures,
        }
        if behavior_failures:
            existing_failures = (
                self.report["reasons"]
                if self.report["verdict"]
                in ("does_not_qualify", "invalid", "validation_failed")
                else []
            )
            self.report.update(
                verdict={
                    "record": "invalid",
                    "validate": "validation_failed",
                    "compare": "does_not_qualify",
                }[config["mode"]],
                reasons=existing_failures + behavior_failures,
            )
        self.screen_timing()

    def screen_timing(self):
        screen = self.report["timing_screen"]
        if self.config["mode"] == "validate" and any(
            abs(item["improvement"]) > 0.05
            for item in self.report["comparisons"].values()
        ):
            screen["reasons"].append("same-commit timing drift exceeds 5%")
        for name, roles in self.report["samples"].items():
            for role, rows in roles.items():
                times = [row["wall_ms"] for row in rows]
                info = calibration.estimate([times])
                screen.setdefault("estimates", {}).setdefault(name, {})[role] = info
                reasons = info["reasons"]
                screen["reasons"].extend(
                    f"{name}/{role}: {reason}" for reason in reasons
                )
        screen["verdict"] = "inconclusive" if screen["reasons"] else "stable"
        quality_ok = self.report.get("quality_comparison", {}).get(
            "passed", not self.profile["quality"] or self.config["mode"] == "record"
        )
        if self.profile["quality"] and self.config["mode"] == "record":
            expected = len(
                read_json(BENCHMARKS / "corpora/quality-500.json")["queries"]
            )
            quality_runs = self.report["quality"].get("foundation", {}).get("runs", [])
            quality_ok = len(quality_runs) == expected
            if not quality_ok:
                self.report.update(
                    verdict="invalid",
                    reasons=self.report["reasons"]
                    + ["incomplete foundation quality evaluation"],
                )
        resource_failure = any(
            item["interval"][1] < -0.05
            for item in self.report["resource_comparisons"].values()
        )
        # Uncertain timing cannot erase independent correctness/resource failures.
        if (
            screen["reasons"]
            and quality_ok
            and self.report["behavior_comparison"]["passed"]
            and not resource_failure
            and self.report["verdict"] not in ("does_not_qualify", "invalid", "validation_failed")
        ):
            self.report.update(verdict="inconclusive", reasons=list(screen["reasons"]))
        elif (
            self.config["mode"] == "compare"
            and local_screening.enabled(self.profile)
            and not screen["reasons"]
            and quality_ok
            and self.report["behavior_comparison"]["passed"]
            and not resource_failure
        ):
            result, reasons = local_screening.candidate_decision(
                self.report["comparisons"],
                self.report["resource_comparisons"],
                self.profile["scenarios"],
            )
            self.report.update(verdict=result, reasons=reasons)


def worker(config_path: Path) -> str:
    config = read_json(config_path)
    lock_path = Path(config["directory"]).parent / ".active.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "another benchmark is running in this run directory"
            ) from error
        return Run(config).execute()


def calibration_scenarios(profile: dict) -> tuple[str, ...]:
    names = tuple(name for name in CALIBRATION if name in profile["scenarios"])
    if not names:
        raise ValueError("profile has no scenarios for reference calibration")
    return names


def recalculate(record: dict, baseline: dict | None = None) -> dict:
    """Derive current results from raw samples without modifying the source report."""
    import copy

    config = record["config"]
    profile = record["contract"]["profile"]
    run = Run(config | {"profile": profile, "directory": config.get("directory", "."),
                        "max_seconds": config.get("max_seconds", 0)})
    run.report = copy.deepcopy(record)
    run.report.update(verdict="incomplete", reasons=[], comparisons={}, resource_comparisons={},
                      analysis_method=calibration.METHOD, analysis_harness_sha256=harness_digest(),
                      timing_screen={"method": calibration.METHOD, "reasons": []})
    for name, values in record.get("calibration", {}).items():
        if len(values) != 3 * profile.get("calibration_batches", 3):
            raise ValueError(f"incomplete calibration measurements: {name}")
        if not values or any(not row.get("behavior", {}).get("passed", False) for row in values):
            raise ValueError(f"missing or failed calibration behavior: {name}")
        info = calibration.estimate([[row["wall_ms"] for row in values]])
        run.report["timing_screen"].setdefault("calibration", {})[name] = info
        reasons = list(info["reasons"])
        if baseline:
            lo, hi = baseline["calibration_bounds"][name]
            if not lo <= info["median_ms"] <= hi:
                reasons.append("foundation median drift outside reference 10% margin")
        run.report["timing_screen"]["reasons"].extend(f"calibration/{name}: {reason}" for reason in reasons)
    if config["mode"] == "compare":
        if baseline is None:
            run.report["timing_screen"]["reasons"].append("comparison requires a current calibrated foundation")
        elif (baseline.get("analysis_method") != calibration.METHOD
              or baseline.get("verdict") != "calibrated"
              or calibration.measurement_contract(baseline["contract"]) != calibration.measurement_contract(record["contract"])):
            raise ValueError("comparison foundation conditions differ or need reanalysis")
        if set(record.get("calibration", {})) != set(calibration_scenarios(profile)):
            raise ValueError("incomplete comparison calibration")
    run.summarize()
    return run.report


def reanalyze(directory: Path, destination: Path, baseline: Path | None = None) -> str:
    if destination.exists():
        raise ValueError("analysis outputs are immutable; choose a new destination")
    source = directory / "report.json"
    analyzed = recalculate(read_json(source), read_json(baseline) if baseline else None)
    analyzed["source_report"] = {"path": str(source.resolve()), "sha256": digest(source)}
    destination.mkdir(parents=True)
    write_json(destination / "report.json", analyzed)
    render(destination)
    return analyzed["verdict"]


def foundation(reports: list[Path], destination: Path):
    if destination.exists():
        raise ValueError("foundation files are immutable; choose a new destination")
    if len(reports) < 3 or len({str(path.resolve()) for path in reports}) != len(reports):
        raise ValueError("at least three distinct foundation sessions are required")
    records = [read_json(path / "report.json") for path in reports]
    if any(row.get("config", {}).get("mode") != "record" for row in records):
        raise ValueError("foundation requires recording sessions")
    contracts = [calibration.measurement_contract(row["contract"]) for row in records]
    if any(item != contracts[0] for item in contracts):
        raise ValueError("foundation sessions have different contracts")
    profile = contracts[0]["profile"]
    if profile["cloud"] and (any(not row["environment"].get("instance_id") for row in records)
            or len({row["environment"]["instance_id"] for row in records}) != len(records)):
        raise ValueError("cloud foundation requires distinct instances")
    # Recompute instead of trusting an obsolete verdict, including CV-only failures.
    for row in records:
        analyzed = recalculate(row)
        if analyzed["verdict"] not in ("foundation_recorded", "inconclusive"):
            raise ValueError("foundation correctness failed: " + "; ".join(analyzed["reasons"]))
    estimates = {name: calibration.estimate([
        [sample["wall_ms"] for sample in row["samples"][name]["foundation"]]
        for row in records
    ]) for name in profile["scenarios"]}
    reasons = [f"{name}: {reason}" for name, info in estimates.items() for reason in info["reasons"]]
    result = "inconclusive" if reasons else "calibrated"
    write_json(destination, {
        "schema_version": 4, "verdict": result, "reasons": reasons,
        "analysis_method": calibration.METHOD, "contract": contracts[0],
        "source_contracts": [row["contract"] for row in records],
        "estimates": estimates,
        "calibration_bounds": {} if reasons else {
            name: calibration.reference_bounds(estimates[name]) for name in calibration_scenarios(profile)
        },
        "reports": [{"path": str(path.resolve()), "sha256": digest(path / "report.json")} for path in reports],
    })
    return result
