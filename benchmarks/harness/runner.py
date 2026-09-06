"""Build immutable commits and execute bounded, paired benchmark sessions."""

from __future__ import annotations

import fcntl
import os
import platform
import shutil
import signal
import socket
import subprocess
import tarfile
import threading
import time
from pathlib import Path

from . import quality
from .assets import verify
from .report import render
from .scenarios import CALIBRATION, SCENARIOS, Scenario
from .statistics import paired, summary, verdict
from .storage import (
    BENCHMARKS,
    FOUNDATION,
    ROOT,
    command,
    digest,
    harness_digest,
    read_json,
    write_json,
)


def build(revision: str, destination: Path, invoke) -> tuple[Path, str]:
    sha = command(
        ["git", "rev-parse", "--verify", revision + "^{commit}"], cwd=ROOT
    ).strip()
    cached = destination / "build.json"
    if cached.exists():
        record = read_json(cached)
        binary = destination / "target/release/visiongrep"
        if (
            record["commit"] == sha
            and binary.is_file()
            and digest(binary) == record["sha256"]
        ):
            return binary, sha
        raise ValueError(f"cached build is corrupt: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "source.tar"
    command(["git", "archive", "--format=tar", "--output", str(archive), sha], cwd=ROOT)
    with tarfile.open(archive) as source:
        source.extractall(destination, filter="data")
    archive.unlink()
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("CARGO_PROFILE_", "ORT_")) or key in (
            "RUSTFLAGS",
            "CARGO_ENCODED_RUSTFLAGS",
            "RUSTC_WRAPPER",
            "RUSTC_WORKSPACE_WRAPPER",
        ):
            del env[key]
    env["VISIONGREP_BUILD_COMMIT"] = sha
    env["CARGO_TARGET_DIR"] = str(destination / "target")
    invoke(["cargo", "build", "--release", "--locked"], cwd=destination, env=env)
    binary = destination / "target/release/visiongrep"
    write_json(cached, {"commit": sha, "sha256": digest(binary)})
    return binary, sha


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
    return {
        "foundation_sha": FOUNDATION,
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
        self.report["contract"] = contract(config, self.profile, corpus)
        baseline = (
            read_json(Path(config["baseline"])) if config.get("baseline") else None
        )
        if config["mode"] == "compare":
            if baseline is None:
                raise ValueError(
                    "comparison requires a calibrated foundation; run validate or record first"
                )
            if baseline["contract"] != self.report["contract"]:
                raise ValueError(
                    "foundation measurement contract differs; recalibration is required"
                )
        self.progress(stage="building")
        if config.get("binaries"):
            binaries = {
                key: Path(value["path"]) for key, value in config["binaries"].items()
            }
            identities = {
                key: value["sha"] for key, value in config["binaries"].items()
            }
        else:
            build_key = (
                platform.system() + "-" + platform.release() + "-" + platform.machine()
            )
            build_cache = cache / "builds" / build_key
            foundation, sha = build(FOUNDATION, build_cache / FOUNDATION, self.invoke)
            binaries, identities = {"foundation": foundation}, {"foundation": sha}
            if config["mode"] != "record":
                candidate, sha = build(
                    config["candidate"], build_cache / config["candidate"], self.invoke
                )
                binaries["candidate"], identities["candidate"] = candidate, sha
        self.report["binaries"] = {
            key: {"commit": identities[key], "sha256": digest(path)}
            for key, path in binaries.items()
        }
        if identities["foundation"] != FOUNDATION:
            raise ValueError("wrong foundation binary identity")
        if config["mode"] == "validate" and identities["candidate"] != FOUNDATION:
            raise ValueError("validation must compare foundation against itself")
        self.progress(stage="calibrating")
        calibration_corpus = corpus | {"images": corpus["images"][:500]}
        for name in CALIBRATION:
            scenario = Scenario(
                self.directory / "calibration" / name,
                binaries["foundation"],
                cache,
                calibration_corpus,
                name,
                self.invoke,
            )
            values = []
            try:
                for sample in range(9):
                    self.progress(scenario=name, sample=sample, total_samples=9)
                    values.append(scenario.sample(sample))
                    self.report["calibration"][name] = values
                    self.save()
            finally:
                scenario.cleanup()
                self.discard_inputs(scenario)
            medians = [
                summary([x["wall_ms"] for x in values[i : i + 3]])["median"]
                for i in (0, 3, 6)
            ]
            stable = summary(medians)["cv"] <= 0.10
            if baseline:
                lo, hi = baseline["calibration_bounds"][name]
                stable &= all(lo <= value <= hi for value in medians)
            if not stable:
                raise ValueError(
                    f"foundation calibration failed for {name}; retain run, investigate drift"
                )
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
        if any(
            summary([v["wall_ms"] for v in by_binary["foundation"]])["cv"] > 0.10
            for by_binary in self.report["samples"].values()
        ):
            self.report.update(
                verdict="invalid",
                reasons=["within-run foundation variation exceeds 10%"],
            )

    def discard_inputs(self, scenario):
        for name in ("images", "cache"):
            shutil.rmtree(scenario.root / name)
        scenario.external.unlink(missing_ok=True)

    def summarize(self):
        config = self.config
        self.report["summaries"] = {}
        self.report["resource_comparisons"] = {}
        for name, by_binary in self.report["samples"].items():
            self.report["summaries"][name] = {}
            for key, observations in by_binary.items():
                fields = (
                    "wall_ms",
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
            if "candidate" in by_binary:
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
                verdict="validation_passed"
                if okay and not drift
                else "validation_failed",
                reasons=["same-commit comparison; not an improvement claim"],
            )
        else:
            result, reasons = verdict(
                self.report["comparisons"],
                self.report.get("quality_comparison", {}).get("passed", False),
                True,
                required_scenarios=SCENARIOS,
            )
            if self.profile["name"] != "cloud-standard":
                result, reasons = (
                    "inconclusive",
                    [
                        "screening profile; qualification requires the complete cloud-standard suite"
                    ],
                )
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
                regressions = [
                    name
                    for name, item in self.report["resource_comparisons"].items()
                    if item["interval"][1] < -0.05
                ]
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
                elif regressions:
                    result, reasons = (
                        "does_not_qualify",
                        ["memory/index regression: " + name for name in regressions],
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


def foundation(reports: list[Path], destination: Path):
    if destination.exists():
        raise ValueError("foundation files are immutable; choose a new destination")
    records = [read_json(path / "report.json") for path in reports]
    if len(records) < 3 or len({str(path.resolve()) for path in reports}) != len(
        records
    ):
        raise ValueError("at least three distinct foundation sessions are required")
    if any(row["verdict"] != "foundation_recorded" for row in records):
        raise ValueError("all sessions must finish recording successfully")
    first = records[0]["contract"]
    if any(row["contract"] != first for row in records):
        raise ValueError("foundation sessions have different contracts")
    if (
        first["environment"]["ami"]
        and len({row["environment"]["instance_id"] for row in records}) < 3
    ):
        raise ValueError("cloud foundation requires at least three different instances")
    bounds = {}
    for name in CALIBRATION:
        values = [
            summary([v["wall_ms"] for v in row["calibration"][name][i : i + 3]])[
                "median"
            ]
            for row in records
            for i in (0, 3, 6)
        ]
        info = summary(values)
        radius = max(info["mad"] * 1.4826 * 3, info["median"] * 0.03)
        if radius > info["median"] * 0.15 or info["cv"] > 0.10:
            raise ValueError(f"foundation variation too large for {name}")
        bounds[name] = [info["median"] - radius, info["median"] + radius]
        if any(not bounds[name][0] <= value <= bounds[name][1] for value in values):
            raise ValueError(
                f"foundation contains outlying calibration batch for {name}"
            )
    write_json(
        destination,
        {
            "schema_version": 1,
            "contract": first,
            "calibration_bounds": bounds,
            "reports": [
                {"path": str(path), "sha256": digest(path / "report.json")}
                for path in reports
            ],
        },
    )
