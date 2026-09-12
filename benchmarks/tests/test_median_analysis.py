"""Typical-latency evidence, drift, paired uncertainty, and immutable reanalysis."""

import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import calibration
from harness.runner import foundation, reanalyze, recalculate
from harness.statistics import paired
from harness.storage import digest, read_json, write_json
import test_foundation as foundation_tests
import test_local_screening as local_tests


class MedianAnalysis(unittest.TestCase):
    def test_transient_clusters_are_diagnostics_but_sustained_drift_is_not(self):
        times = [100, 100, 100, 125, 125, 100, 100] * 3
        info = calibration.estimate([times])
        self.assertGreater(info["raw_summaries"][0]["cv"], 0.10)
        self.assertEqual(info["raw_summaries"][0]["p95"], 125)
        self.assertEqual(info["batch_medians_ms"], [[100, 100, 100]])
        self.assertEqual(info["reasons"], [])
        drift = calibration.estimate([[100] * 14 + [140] * 7])
        self.assertIn("batch median drift exceeds 15%", drift["reasons"])
        between = calibration.estimate([[100] * 21, [100] * 21, [140] * 21])
        self.assertIn("session median drift exceeds 15%", between["reasons"])

    def test_median_is_of_raw_samples_not_median_of_triplet_medians(self):
        times = [1, 1, 100, 2, 2, 100, 100, 100, 100]
        self.assertEqual(calibration.estimate([times])["median_ms"], 100)

    def test_short_recordings_cannot_certify_precision(self):
        for times in ([100] * 3, [100] * 5, [100] * 10):
            info = calibration.estimate([times])
            self.assertIsNone(info["median_interval_ms"])
            self.assertTrue(info["reasons"])
        with tempfile.TemporaryDirectory() as temporary:
            paths = foundation_tests.FoundationBounds().records(Path(temporary), "cloud-scale", [100] * 3)
            for path in paths:
                report = read_json(path / "report.json")
                report["contract"]["profile"]["samples"] = 3
                write_json(path / "report.json", report)
            output = Path(temporary) / "foundation.json"
            self.assertEqual(foundation(paths, output), "inconclusive")
            self.assertEqual(read_json(output)["calibration_bounds"], {})

    def test_invalid_numbers_fail_closed(self):
        for number in (0, -1, math.nan, math.inf):
            with self.subTest(number=number), self.assertRaises(ValueError):
                calibration.estimate([[100] * 20 + [number]])
            with self.subTest(number=number), self.assertRaises(ValueError):
                paired([100] * 21, [100] * 20 + [number])

    def test_paired_comparison_targets_ratio_of_medians_and_keeps_pairs(self):
        # Median per-pair ratios would misleadingly report -100% here.
        result = paired([1, 10, 100] * 7, [2, 100, 10] * 7)
        self.assertEqual(result["improvement"], 0)
        reference = [90, 100, 110] * 7
        result = paired(reference, [x * 0.8 for x in reference])
        self.assertAlmostEqual(result["improvement"], 0.2)
        for bound in result["interval"]:
            self.assertAlmostEqual(bound, 0.2)
        with self.assertRaises(ValueError):
            paired(reference, reference[:-1])

    def test_cloud_noise_cannot_erase_definite_latency_regression(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = local_tests.LocalScreening().run_with_samples(temporary, "compare", [100] * 21)
            run.profile.update(name="cloud-standard", cloud=True, samples=21)
            run.report["quality_comparison"] = {"passed": True}
            for roles in run.report["samples"].values():
                for row in roles["candidate"]:
                    row["wall_ms"] = 140
                roles["candidate"][-1]["wall_ms"] = 1000
            run.summarize()
            self.assertEqual(run.report["verdict"], "does_not_qualify")
            self.assertTrue(any("material regression" in reason for reason in run.report["reasons"]))

    def test_reanalysis_retains_sequences_once_and_never_rewrites_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = foundation_tests.FoundationBounds().records(root)[0]
            report = read_json(path / "report.json")
            report["verdict"] = "invalid"
            report["reasons"] = ["within-run foundation variation exceeds 10%"]
            for row in report["samples"]["persistent_text"]["foundation"]:
                row["requests"] = [{"wall_ms": 12345}] * 5
            write_json(path / "report.json", report)
            before = digest(path / "report.json")
            output = root / "analysis"
            self.assertEqual(reanalyze(path, output), "foundation_recorded")
            self.assertEqual(before, digest(path / "report.json"))
            result = read_json(output / "report.json")
            self.assertEqual(result["samples"], report["samples"])
            self.assertEqual(result["timing_screen"]["estimates"]["persistent_text"]["foundation"]["raw_summaries"][0]["samples"], 21)
            self.assertEqual(result["source_report"]["sha256"], before)
            with self.assertRaisesRegex(ValueError, "immutable"):
                reanalyze(path, output)
            broken = copy.deepcopy(report)
            broken["samples"]["cached_text"]["foundation"][0]["behavior"] = {"passed": False, "reasons": ["bad output"]}
            self.assertEqual(recalculate(broken)["verdict"], "invalid")

    def test_one_instance_never_makes_a_cloud_foundation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = foundation_tests.FoundationBounds().records(root)
            with self.assertRaisesRegex(ValueError, "three distinct"):
                foundation(paths[:1], root / "foundation.json")


if __name__ == "__main__":
    unittest.main()
