"""Offline checks of measurement, decisions, manifests, and cleanup contracts."""

import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import cloud, quality
from harness.statistics import paired, summary, verdict
from harness.storage import BENCHMARKS, read_json
from harness.scenarios import SCENARIOS, Scenario
from harness.runner import Run


class ScenarioState(unittest.TestCase):
    def test_each_sample_restores_inputs_and_seed(self):
        class InspectScenario(Scenario):
            def execute(self, label, query, *, measured, **options):
                if not self.index.exists() and not options.get("no_cache"):
                    with sqlite3.connect(self.index) as connection:
                        connection.execute("CREATE TABLE marker (value INTEGER)")
                        connection.execute("INSERT INTO marker VALUES (1)")
                return {
                    "images": {p.name: p.read_bytes() for p in self.images.iterdir()},
                    "options": options,
                    "mode": self.images.stat().st_mode & 0o777,
                }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            (cache / "objects").mkdir(parents=True)
            rows = []
            for index in range(4):
                (cache / "objects" / str(index)).write_bytes(bytes([index]))
                rows.append({"file_name": f"{index}.jpg", "sha256": str(index)})

            def models(cache, destination):
                (destination / "visiongrep/models").mkdir(parents=True)

            with patch("harness.scenarios.stage_models", side_effect=models):
                for name in SCENARIOS:
                    with self.subTest(name=name):
                        scenario = InspectScenario(
                            root / name,
                            Path("unused"),
                            cache,
                            {"images": rows},
                            name,
                            None,
                        )
                        try:
                            first = scenario.sample(0)
                            second = scenario.sample(1)
                            self.assertEqual(first["images"], second["images"])
                            if name == "modified_1pct":
                                self.assertEqual(first["images"]["0.jpg"], bytes([3]))
                            if name == "added_1pct":
                                self.assertEqual(len(first["images"]), 4)
                            if name == "deleted_1pct":
                                self.assertEqual(len(first["images"]), 3)
                            if name == "read_only":
                                self.assertEqual(first["mode"], 0o555)
                            self.assertEqual(
                                (cache / "objects/0").read_bytes(), bytes([0])
                            )
                        finally:
                            scenario.cleanup()


class Lifecycle(unittest.TestCase):
    def test_deadline_kills_child_and_preserves_invalid_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = {
                "directory": temporary,
                "profile": {"name": "test"},
                "max_seconds": 0.1,
                "command_timeout_seconds": 0.1,
                "hourly_budget_usd": 0,
            }
            run = Run(config)

            def work():
                run.invoke([sys.executable, "-c", "import time; time.sleep(10)"])

            with patch.object(run, "perform", side_effect=work):
                self.assertEqual(run.execute(), "invalid")
            self.assertTrue((Path(temporary) / "report.html").exists())
            self.assertIn(
                "deadline", read_json(Path(temporary) / "report.json")["reasons"][0]
            )

    def test_cancel_is_distinct_from_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = {
                "directory": temporary,
                "profile": {"name": "test"},
                "max_seconds": 60,
                "command_timeout_seconds": 60,
                "hourly_budget_usd": 0,
            }
            run = Run(config)
            (Path(temporary) / "cancel").touch()
            with patch.object(run, "perform", side_effect=lambda: run.progress()):
                self.assertEqual(run.execute(), "cancelled")


class Decisions(unittest.TestCase):
    def test_21_is_second_slowest(self):
        self.assertEqual(summary(list(range(1, 22)))["p95"], 20)
        self.assertIsNone(summary([1, 2, 3])["p95"])

    def test_improvement_and_regression(self):
        baseline = [100 + i % 3 for i in range(21)]
        better = paired(baseline, [v * 0.90 for v in baseline])
        worse = paired(baseline, [v * 1.10 for v in baseline])
        self.assertEqual(
            verdict({"target": better}, "target", True, True)[0], "qualifies"
        )
        self.assertEqual(
            verdict({"target": better, "other": worse}, "target", True, True)[0],
            "does_not_qualify",
        )
        self.assertEqual(
            verdict({"target": better}, "target", True, False)[0], "invalid"
        )
        self.assertEqual(
            verdict({"target": better}, "target", False, True)[0], "does_not_qualify"
        )

    def test_short_run_cannot_qualify(self):
        result = paired([100] * 5, [80] * 5)
        self.assertEqual(
            verdict({"target": result}, "target", True, True)[0], "inconclusive"
        )

    def test_small_gain_does_not_qualify(self):
        result = paired([100] * 21, [98] * 21)
        self.assertEqual(
            verdict({"target": result}, "target", True, True)[0], "does_not_qualify"
        )


class Corpora(unittest.TestCase):
    def test_nested_unique_corpora(self):
        small = read_json(BENCHMARKS / "corpora/coco-500.json")
        large = read_json(BENCHMARKS / "corpora/coco-10000.json")
        self.assertEqual(len(small["images"]), 500)
        self.assertEqual(len(large["images"]), 10000)
        self.assertEqual(len({r["file_name"] for r in large["images"]}), 10000)
        self.assertEqual(small["images"], large["images"][:500])

    def test_absent_galleries_exclude_all_annotated_positives(self):
        corpus = read_json(BENCHMARKS / "corpora/coco-500.json")
        queries = read_json(BENCHMARKS / "corpora/quality-500.json")["queries"]
        intents = set()
        for query in queries:
            if query["kind"] != "category":
                continue
            expected = {
                r["file_name"]
                for r in corpus["images"]
                if query["category_id"] in r["category_ids"]
            }
            self.assertEqual(set(query["relevant"]), expected)
            self.assertGreaterEqual(len(expected), 3)
            self.assertLess(len(expected), 500)
            intents.add(query["intent"])
        self.assertEqual(len(intents), 80)
        self.assertEqual(len([q for q in queries if q["kind"] == "caption"]), 200)

    def test_quality_filters_all_scores(self):
        positive = {
            "id": "p",
            "kind": "caption",
            "relevant": ["good.jpg"],
            "results": [{"path": "good.jpg", "score": 0.9}],
        }
        category = {
            "id": "c",
            "intent": "cat",
            "kind": "category",
            "relevant": ["good.jpg"],
            "results": [
                {"path": "good.jpg", "score": 0.9},
                {"path": "bad.jpg", "score": 0.3},
            ],
        }
        result = quality.evaluate([positive, category], 0.25)
        self.assertEqual(result["absent_false_positive_rate"], 1)
        self.assertEqual(result["present_false_negative_rate"], 0)
        self.assertEqual(result["recall_at_10"], 1)


class ProcessMeasurement(unittest.TestCase):
    def test_isolated_child_rss_and_exit_status(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)
            config = {
                "command": [sys.executable, "-c", "print('[]'); raise SystemExit(1)"],
                "stdout": str(path / "out"),
                "stderr": str(path / "err"),
            }
            (path / "config").write_text(json.dumps(config))
            result = subprocess.run(
                [
                    sys.executable,
                    str(BENCHMARKS / "harness/measurement.py"),
                    str(path / "config"),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            sample = json.loads(result.stdout)
            self.assertEqual(sample["exit_code"], 1)
            self.assertGreater(sample["peak_rss_bytes"], 0)
            self.assertGreater(sample["wall_ms"], 0)


class Infrastructure(unittest.TestCase):
    def test_bootstrap_syntax_and_expiry_before_download(self):
        script = cloud.bootstrap(
            "example-bucket", "runs/example", "us-east-1", "ami-1234", 60
        )
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)
        self.assertLess(script.index("systemd-run"), script.index("aws s3 cp"))
        self.assertIn("shutdown -h now", script)
        self.assertIn("runuser -u bench", script)

    def test_placeholders_cannot_launch(self):
        with self.assertRaises(ValueError):
            cloud.validate_settings(
                read_json(BENCHMARKS / "profiles/cloud.example.json")
            )


if __name__ == "__main__":
    unittest.main()
