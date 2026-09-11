"""Diagnostic isolation and instrumentation checks without AWS or model inference."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench import configuration, parser
from harness.diagnostics import Telemetry, measure, snapshot, trace_command
from harness.runner import foundation
from harness.storage import BENCHMARKS, FOUNDATION, write_json


class DiagnosticConfiguration(unittest.TestCase):
    @patch("bench.command", return_value=FOUNDATION)
    def test_fixed_protocol_and_deadline(self, git):
        config = configuration(parser().parse_args([
            "plan", "--profile", "cloud-standard", "--mode", "diagnose",
            "--cloud", "/tmp/cloud.json",
        ]))
        self.assertEqual(config["mode"], "diagnose")
        self.assertEqual(config["candidate"], FOUNDATION)
        self.assertEqual(config["max_seconds"], 3600)
        self.assertFalse(config["profile"]["quality"])
        self.assertEqual(config["profile"]["samples"], 100)
        self.assertEqual(config["profile"]["batches"], ["untraced", "strace"])
        self.assertEqual(set(config["profile"]["scenarios"]), {
            "deleted_1pct", "modified_query_image", "modified_1pct", "renamed_1pct",
        })

    def test_local_diagnostics_and_sample_override_rejected(self):
        for extra in ([], ["--profile", "cloud-standard", "--cloud", "/tmp/cloud.json", "--validation-samples", "3"]):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                configuration(parser().parse_args(["plan", "--mode", "diagnose", *extra]))

    def test_completed_diagnostics_cannot_become_foundation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = [root / str(i) for i in range(3)]
            for run in runs:
                write_json(run / "report.json", {"verdict": "diagnostic_complete"})
            with self.assertRaisesRegex(ValueError, "finish recording"):
                foundation(runs, root / "foundation.json")


class DiagnosticInstrumentation(unittest.TestCase):
    def test_both_batches_preserve_samples_and_trace_only_measured_children(self):
        calls = []
        original_temporary = tempfile.TemporaryDirectory
        original_read = Path.read_text
        original_exists = Path.exists

        class FakeTelemetry:
            def __init__(self, path):
                self.path = path
                path.write_text('{"snapshot":true}\n')
                self.thread = SimpleNamespace(start=lambda: None, join=lambda: None)
                self.stop = SimpleNamespace(set=lambda: None)

            def check(self):
                pass

        class FakeScenario:
            def __init__(self, root, binary, cache, corpus, name, invoke):
                self.root, self.invoke = root, invoke
                self.call("warmup")

            def call(self, label):
                output = self.root / "observations" / label
                output.mkdir(parents=True)
                self.invoke([sys.executable, str(BENCHMARKS / "harness/measurement.py"), str(output / "invocation.json")], env={})

            def sample(self, index):
                self.call(f"sample-{index:04d}")
                return {
                    "wall_ms": 500 if index == 1 else 20,
                    "diagnostic": {}, "behavior": {"passed": True},
                    "timing": {"environment": {"commit": FOUNDATION}, "phases": []},
                }

            def cleanup(self):
                pass

        def invoke(args, **kwargs):
            calls.append((args, kwargs))
            if "strace" in args:
                Path(args[args.index("-o") + 1]).write_text("retained trace\n")
            return "{}"

        def proc_read(path, *args, **kwargs):
            if str(path) == "/proc/mounts":
                return "tmpfs /dev/shm tmpfs rw 0 0\n"
            if str(path) == "/proc/sys/kernel/sched_schedstats":
                return "1\n"
            return original_read(path, *args, **kwargs)

        with original_temporary() as temporary:
            root = Path(temporary)
            run = SimpleNamespace(
                directory=root, deadline=9999999999, report={"samples": {}},
                config={"foundation_sha": FOUNDATION},
                profile={"batches": ["untraced", "strace"], "scenarios": ["deleted_1pct"], "samples": 2},
                invoke=invoke, progress=lambda **kwargs: None, save=lambda: None,
                discard_inputs=lambda scenario: None,
            )
            with (
                patch("harness.diagnostics.Scenario", FakeScenario),
                patch("harness.diagnostics.Telemetry", FakeTelemetry),
                patch("harness.diagnostics.shutil.which", return_value="tool"),
                patch("harness.diagnostics.shutil.disk_usage", return_value=SimpleNamespace(free=1024**3)),
                patch("harness.diagnostics.command", return_value="strace test"),
                patch("harness.diagnostics.tempfile.TemporaryDirectory", side_effect=lambda **kwargs: original_temporary()),
                patch.object(Path, "read_text", proc_read),
                patch.object(Path, "exists", lambda path: str(path) == "/proc/diskstats" or original_exists(path)),
                patch.dict(os.environ, {}, clear=True),
            ):
                measure(run, Path("binary"), Path("cache"), {})
            self.assertEqual(run.report["verdict"], "diagnostic_complete")
            self.assertEqual(len(calls), 6)
            self.assertEqual(sum("strace" in args for args, _ in calls), 2)
            for batch in ("untraced", "strace"):
                self.assertEqual([r["wall_ms"] for r in run.report["samples"]["deleted_1pct"][batch]], [20, 500])
            self.assertEqual(len(list(root.rglob("syscalls.log"))), 2)

    def test_helper_records_resource_counters_only_when_requested(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "invocation.json"
            write_json(config, {
                "command": [sys.executable, "-c", "print('result')"],
                "stdout": str(root / "stdout"), "stderr": str(root / "stderr"),
            })
            for diagnostic in (False, True):
                env = os.environ.copy()
                env.pop("VISIONGREP_BENCH_DIAGNOSTIC", None)
                if diagnostic:
                    env["VISIONGREP_BENCH_DIAGNOSTIC"] = "1"
                result = json.loads(subprocess.check_output([
                    sys.executable, str(BENCHMARKS / "harness/measurement.py"), str(config),
                ], env=env))
                self.assertEqual(result["exit_code"], 0)
                self.assertEqual((root / "stdout").read_text(), "result\n")
                self.assertEqual("diagnostic" in result, diagnostic)
                if diagnostic:
                    self.assertGreater(result["diagnostic"]["started_epoch_ns"], 0)
                    self.assertGreaterEqual(result["diagnostic"]["system_seconds"], 0)

    def test_trace_keeps_original_arguments_and_bounds_output(self):
        original = ["python3", "measurement.py", "/path with spaces/invocation.json"]
        traced = trace_command(original, Path("/dev/shm/trace.log"))
        self.assertEqual(traced[-3:], original)
        self.assertIn("--fsize=67108864:67108864", traced)
        self.assertIn("-T", traced)
        self.assertIn("-ttt", traced)
        self.assertIn("/dev/shm/trace.log", traced)

    def test_disappearing_process_coverage_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary)
            (proc / "123/task/123").mkdir(parents=True)
            (proc / "123/task/123/schedstat").write_text("100 200 3\n")
            row = snapshot(proc)
            self.assertEqual(row["processes"]["123"]["threads"]["123"], "100 200 3\n")
            self.assertEqual(row["processes"]["123"]["stat"], {"unavailable": "FileNotFoundError"})

    def test_telemetry_failure_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            monitor = Telemetry(Path(temporary) / "telemetry.jsonl")
            with patch("harness.diagnostics.snapshot", side_effect=OSError("injected")):
                monitor.collect()
            with self.assertRaisesRegex(RuntimeError, "telemetry failed"):
                monitor.check()


if __name__ == "__main__":
    unittest.main()
