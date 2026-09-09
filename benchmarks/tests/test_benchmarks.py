"""Offline checks of measurement, decisions, manifests, and cleanup contracts."""

import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import cloud, quality
from harness.statistics import paired, summary, verdict
from harness.storage import BENCHMARKS, FOUNDATION, read_json
from harness.scenarios import SCENARIOS, Scenario
from harness.runner import Run
from bench import configuration, parser


class ValidationConfiguration(unittest.TestCase):
    # Testing profile options must not depend on tags in the developer/CI checkout.
    @patch("bench.command", return_value=FOUNDATION)
    def test_short_validation_does_not_change_comparison_defaults(self, git_command):
        args = parser().parse_args(
            ["plan", "--mode", "validate", "--validation-samples", "3"]
        )
        profile = configuration(args)["profile"]
        self.assertEqual(profile["samples"], 3)
        self.assertEqual(profile["calibration_batches"], 3)
        self.assertTrue(profile["quality"])
        cloud_short = configuration(parser().parse_args([
            "plan", "--mode", "validate", "--profile", "cloud-standard",
            "--validation-samples", "3",
        ]))["profile"]
        self.assertEqual(cloud_short["calibration_batches"], 1)
        standard = configuration(
            parser().parse_args(["plan", "--profile", "cloud-standard"])
        )["profile"]
        self.assertEqual(standard["samples"], 21)
        self.assertEqual(standard.get("calibration_batches", 3), 3)
        self.assertEqual(set(standard["scenarios"]), set(SCENARIOS))

    def test_moved_foundation_tag_is_rejected(self):
        with (
            patch("bench.command", side_effect=[FOUNDATION, "0" * 40]),
            self.assertRaisesRegex(ValueError, "foundation tag moved"),
        ):
            configuration(parser().parse_args(["plan", "--mode", "validate"]))

    def test_validation_override_cannot_weaken_comparison_or_reference(self):
        for mode in ("compare", "record"):
            with (
                self.subTest(mode=mode),
                self.assertRaisesRegex(ValueError, "requires --mode validate"),
            ):
                configuration(
                    parser().parse_args(
                        ["plan", "--mode", mode, "--validation-samples", "3"]
                    )
                )


class ScenarioState(unittest.TestCase):
    def test_each_sample_restores_inputs_and_seed(self):
        class InspectScenario(Scenario):
            def execute(self, label, query, *, measured, **options):
                if not self.index.exists() and not options.get("no_cache"):
                    with closing(sqlite3.connect(self.index)) as connection, connection:
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
    def decision(self, comparisons, *, quality_ok=True, environment_ok=True):
        return verdict(
            comparisons,
            quality_ok,
            environment_ok,
            required_scenarios=("novel_text", "index_absent"),
        )[0]

    def comparison(self, gain, interval=None, pairs=21):
        return {
            "improvement": gain,
            "interval": interval or [gain, gain],
            "pairs": pairs,
        }

    def test_21_is_second_slowest(self):
        self.assertEqual(summary(list(range(1, 22)))["p95"], 20)
        self.assertIsNone(summary([1, 2, 3])["p95"])

    def test_any_standard_scenario_can_supply_the_improvement(self):
        baseline = [100 + i % 3 for i in range(21)]
        better = paired(baseline, [v * 0.90 for v in baseline])
        unchanged = paired(baseline, baseline)
        for improved in ("novel_text", "index_absent"):
            with self.subTest(improved=improved):
                results = {"novel_text": unchanged, "index_absent": unchanged}
                results[improved] = better
                self.assertEqual(self.decision(results), "qualifies")

    def test_regression_elsewhere_blocks_a_supported_improvement(self):
        results = {
            "novel_text": self.comparison(0.1),
            "index_absent": self.comparison(-0.1),
        }
        self.assertEqual(self.decision(results), "does_not_qualify")

    def test_uncertain_regression_cannot_pass(self):
        results = {
            "novel_text": self.comparison(0.1),
            "index_absent": self.comparison(-0.01, [-0.06, 0.03]),
        }
        self.assertEqual(self.decision(results), "inconclusive")

    def test_regression_boundary_passes_when_excluded(self):
        results = {
            "novel_text": self.comparison(0.1),
            "index_absent": self.comparison(0, [-0.05, 0.02]),
        }
        self.assertEqual(self.decision(results), "qualifies")

    def test_unsupported_large_estimate_cannot_pass(self):
        results = {
            "novel_text": self.comparison(0.08, [-0.02, 0.15]),
            "index_absent": self.comparison(0),
        }
        self.assertEqual(self.decision(results), "inconclusive")

    def test_supported_five_percent_boundary(self):
        results = {
            "novel_text": self.comparison(0.05, [0.01, 0.09]),
            "index_absent": self.comparison(0),
        }
        self.assertEqual(self.decision(results), "qualifies")

    def test_no_meaningful_gain_does_not_qualify(self):
        results = {
            "novel_text": self.comparison(0.02),
            "index_absent": self.comparison(0),
        }
        self.assertEqual(self.decision(results), "does_not_qualify")

    def test_short_run_cannot_qualify(self):
        results = {
            "novel_text": self.comparison(0.1, pairs=5),
            "index_absent": self.comparison(0, pairs=5),
        }
        self.assertEqual(self.decision(results), "inconclusive")

    def test_missing_scenarios_and_empty_runs_cannot_qualify(self):
        for results in ({}, {"index_absent": self.comparison(0.2)}):
            self.assertEqual(self.decision(results), "inconclusive")

    def test_quality_and_environment_remain_required(self):
        results = {
            "novel_text": self.comparison(0.1),
            "index_absent": self.comparison(0),
        }
        self.assertEqual(self.decision(results, quality_ok=False), "does_not_qualify")
        self.assertEqual(self.decision(results, environment_ok=False), "invalid")


class SuiteQualification(unittest.TestCase):
    def compare_suite(
        self,
        *,
        profile_name="cloud-standard",
        missing_index=False,
        resource_regression=None,
        quality_ok=True,
        behavior_ok=True,
        mode="compare",
        return_report=False,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            profile = {
                "name": profile_name,
                "samples": 21,
                "quality": True,
                "scenarios": list(SCENARIOS),
            }
            config = {
                "directory": temporary,
                "profile": profile,
                "mode": mode,
                "max_seconds": 60,
                "hourly_budget_usd": 0,
            }
            run = Run(config)
            run.report["quality_comparison"] = {"passed": quality_ok}
            for name in SCENARIOS:
                run.report["samples"][name] = {}
                for role in ("foundation", "candidate"):
                    sample = {
                        "wall_ms": 90
                        if role == "candidate" and name == "index_absent"
                        else 100,
                        "peak_rss_bytes": 1000,
                        "timing": {"phases": []},
                        "behavior": {"passed": True, "reasons": []},
                        "behavior_comparison": {
                            "passed": behavior_ok,
                            "reasons": []
                            if behavior_ok
                            else ["scenario rankings differ"],
                        },
                    }
                    if name != "no_cache" and not (
                        missing_index and role == "candidate" and name == "novel_text"
                    ):
                        sample["index_bytes"] = 2000
                    if (
                        resource_regression
                        and role == "candidate"
                        and name == "index_absent"
                    ):
                        sample[resource_regression] *= 1.1
                    run.report["samples"][name][role] = [sample] * 21
            run.summarize()
            return run.report if return_report else run.report["verdict"]

    def test_full_suite_qualifies_without_a_target(self):
        self.assertEqual(self.compare_suite(), "qualifies")

    def test_resource_and_quality_checks_are_required(self):
        for field in ("peak_rss_bytes", "index_bytes"):
            with self.subTest(field=field):
                self.assertEqual(
                    self.compare_suite(resource_regression=field), "does_not_qualify"
                )
        self.assertEqual(self.compare_suite(quality_ok=False), "does_not_qualify")
        self.assertEqual(self.compare_suite(missing_index=True), "inconclusive")

    def test_screening_profiles_cannot_qualify_even_with_full_samples(self):
        for name in ("local-quick", "cloud-scale"):
            with self.subTest(profile=name):
                self.assertEqual(self.compare_suite(profile_name=name), "inconclusive")

    def test_screening_preserves_quality_failure_and_reason(self):
        report = self.compare_suite(
            profile_name="local-quick", quality_ok=False, return_report=True
        )
        self.assertEqual(report["verdict"], "does_not_qualify")
        self.assertTrue(
            any("quality/behavior" in reason for reason in report["reasons"])
        )

    def test_screening_preserves_resource_failure(self):
        self.assertEqual(
            self.compare_suite(
                profile_name="local-quick", resource_regression="peak_rss_bytes"
            ),
            "does_not_qualify",
        )

    def test_behavior_failure_blocks_improvement_and_validation(self):
        for mode, expected in (
            ("compare", "does_not_qualify"),
            ("validate", "validation_failed"),
            ("record", "invalid"),
        ):
            self.assertEqual(self.compare_suite(behavior_ok=False, mode=mode), expected)

    def test_standard_profile_measures_every_scenario(self):
        profile = read_json(BENCHMARKS / "profiles/cloud-standard.json")
        self.assertCountEqual(profile["scenarios"], SCENARIOS)


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
