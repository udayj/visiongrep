"""Graph-cache diagnostic protocol and decision tests."""

import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench import configuration, parser
from harness.graph_cache import cpu_times, materiality, quiet_machine_check, summarize
from harness.runner import Run
from harness.scenarios import Scenario


class GraphCacheConfiguration(unittest.TestCase):
    @patch(
        "bench.command",
        side_effect=["b" * 40, "4c8c613c8af16591860252683d29f4f0edaee723"],
    )
    def test_fixed_cloud_protocol_without_calibrated_foundation(self, git):
        config = configuration(
            parser().parse_args(
                [
                    "plan",
                    "--profile",
                    "cloud-standard",
                    "--mode",
                    "graph-cache",
                    "--cloud",
                    "/tmp/cloud.json",
                ]
            )
        )
        self.assertEqual(config["candidate"], "b" * 40)
        self.assertIsNone(config["baseline"])
        self.assertEqual(config["profile"]["samples"], 21)
        self.assertEqual(config["max_seconds"], 3600)
        self.assertEqual(config["budget_usd"], 0.30)
        self.assertFalse(config["profile"]["quality"])
        self.assertEqual(
            config["profile"]["scenarios"],
            [
                "novel_text",
                "external_image_first",
                "persistent_text",
                "cached_text",
            ],
        )

    def test_rejects_local_baseline_and_sample_override(self):
        cases = (
            [],
            ["--profile", "cloud-standard", "--cloud", "/tmp/cloud.json", "--baseline", "/tmp/base.json"],
            ["--profile", "cloud-standard", "--cloud", "/tmp/cloud.json", "--validation-samples", "3"],
        )
        for extra in cases:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                configuration(
                    parser().parse_args(["plan", "--mode", "graph-cache", *extra])
                )


class Materiality(unittest.TestCase):
    def test_threshold_and_regression_classification(self):
        self.assertEqual(materiality([0.051, 0.2])[0], "material_improvement")
        self.assertEqual(
            materiality([0.01, 0.049])[0],
            "material_improvement_not_demonstrated",
        )
        self.assertEqual(materiality([0.04, 0.06])[0], "inconclusive")
        self.assertEqual(materiality([-0.2, -0.051])[0], "material_regression")
        self.assertEqual(materiality([-0.06, 0.04])[0], "inconclusive")

    def test_alternating_order_effect_is_reported_without_replacing_ci_rule(self):
        from harness.graph_cache import comparison

        reference = [100] * 21
        candidate = [90 if index % 2 == 0 else 100 for index in range(21)]
        result = comparison(reference, candidate)
        self.assertTrue(result["order_sensitive"])
        self.assertEqual(result["classification"], result["interval_classification"])


class QuietMachine(unittest.TestCase):
    def test_reads_aggregate_counters_without_double_counting_guests(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stat"
            path.write_text("cpu  1 2 3 4 5 6 7 8 9 10\ncpu0 1 2 3 4\n")
            self.assertEqual(cpu_times(path), list(range(1, 9)))

    @patch("harness.graph_cache.time.sleep")
    @patch(
        "harness.graph_cache.cpu_times",
        side_effect=[
            [0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 2, 95, 0, 0, 0, 1],
        ],
    )
    def test_quiet_interval_is_recorded(self, snapshots, sleep):
        result = quiet_machine_check()
        self.assertEqual(result["idle_fraction"], 0.95)
        self.assertEqual(result["steal_fraction"], 0.01)
        sleep.assert_called_once_with(2)

    @patch("harness.graph_cache.time.sleep")
    @patch(
        "harness.graph_cache.cpu_times",
        side_effect=[
            [0, 0, 0, 0, 0, 0, 0, 0],
            [10, 0, 10, 79, 0, 0, 0, 1],
        ],
    )
    def test_busy_interval_fails_without_retry(self, snapshots, sleep):
        with self.assertRaisesRegex(ValueError, "machine was not quiet"):
            quiet_machine_check()
        self.assertEqual(snapshots.call_count, 2)


class GraphIdentity(unittest.TestCase):
    def test_two_nonempty_graphs_remain_identical(self):
        with tempfile.TemporaryDirectory() as temporary:
            scenario = object.__new__(Scenario)
            scenario.root = Path(temporary)
            scenario.graph_cache_role = "candidate"
            optimized = scenario.root / "cache/visiongrep/models/optimized"
            optimized.mkdir(parents=True)
            (optimized / "text.onnx").write_bytes(b"text")
            (optimized / "vision.onnx").write_bytes(b"vision")
            scenario.graph_cache_identity = scenario.optimized_graph_identity(
                include_content=True
            )
            scenario.graph_cache_metadata = [
                {key: value for key, value in item.items() if key != "sha256"}
                for item in scenario.graph_cache_identity
            ]
            self.assertEqual(
                scenario.verify_graph_cache(),
                {
                    "candidate_graphs_unchanged": True,
                    "content_hashes_checked": False,
                },
            )
            (optimized / "text.onnx").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "identities or mtimes changed"):
                scenario.verify_graph_cache(include_content=True)


class GraphSummary(unittest.TestCase):
    def test_reports_primary_service_start_and_warm_controls_without_qualification(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile = {
                "name": "graph-cache-diagnostic",
                "samples": 21,
                "quality": False,
                "scenarios": [
                    "novel_text",
                    "external_image_first",
                    "persistent_text",
                    "cached_text",
                ],
            }
            run = Run(
                {
                    "directory": temporary,
                    "profile": profile,
                    "mode": "graph-cache",
                    "max_seconds": 60,
                    "hourly_budget_usd": 0,
                }
            )
            for scenario in profile["scenarios"]:
                run.report["samples"][scenario] = {}
                for role in ("foundation", "candidate"):
                    value = 80 if role == "candidate" else 100
                    run.report["samples"][scenario][role] = [
                        {
                            "wall_ms": value,
                            "first_response_ms": value,
                            "cached_response_ms": 10,
                            "behavior": {"passed": True},
                            "behavior_comparison": {"passed": True},
                            "graph_cache_validation": {"passed": True},
                        }
                        for _ in range(21)
                    ]
            summarize(run)
            self.assertEqual(run.report["verdict"], "diagnostic_complete")
            self.assertIn("persistent_text:first_response_ms", run.report["comparisons"])
            self.assertEqual(
                run.report["comparisons"]["persistent_text:first_response_ms"][
                    "endpoint_class"
                ],
                "primary",
            )
            self.assertEqual(
                run.report["comparisons"]["persistent_text:cached_response_ms"][
                    "endpoint_class"
                ],
                "warm_control",
            )
            self.assertEqual(
                run.report["comparisons"]["persistent_text"]["endpoint_class"],
                "warm_control",
            )
            self.assertAlmostEqual(
                run.report["comparisons"]["novel_text"][
                    "candidate_to_reference_ratio"
                ],
                0.8,
            )
            self.assertFalse(
                run.report["graph_cache_diagnostic"]["cloud_qualification"]
            )


if __name__ == "__main__":
    unittest.main()
