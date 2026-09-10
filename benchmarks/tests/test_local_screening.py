"""Local noise never substitutes for correctness or cloud qualification."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import local_screening
from harness.runner import Run, foundation
from harness.report import render
from harness.storage import BENCHMARKS, read_json, write_json
import test_foundation as fixtures


class LocalScreening(unittest.TestCase):
    def test_html_surfaces_median_uncertainty_and_raw_cv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self.run_with_samples(root)
            run.summarize()
            write_json(root / "report.json", run.report)
            html = render(root).read_text()
            self.assertIn("Local median uncertainty and batch stability", html)
            self.assertIn("Raw CV", html)
            self.assertIn("100.000 to 100.000", html)

    def test_precheck_tests_current_median_not_every_batch_against_old_band(self):
        class Scenario:
            def __init__(self, *args):
                pass

            def sample(self, index):
                return {
                    "wall_ms": 110 if index < 3 else 100,
                    "behavior": {"passed": True},
                }

            def cleanup(self):
                pass

        with tempfile.TemporaryDirectory() as root:
            run = self.run_with_samples(root)
            baseline = {
                "calibration_bounds": {
                    name: [97, 103] for name in ("novel_text", "cached_text")
                }
            }
            with (
                patch("harness.runner.Scenario", Scenario),
                patch.object(run, "progress"),
                patch.object(run, "save"),
                patch.object(run, "discard_inputs"),
            ):
                run.calibrate(Path("binary"), Path("cache"), {}, baseline)
                self.assertEqual(run.report["timing_screen"]["reasons"], [])
                baseline["calibration_bounds"]["novel_text"] = [50, 60]
                run.calibrate(Path("binary"), Path("cache"), {}, baseline)
            self.assertTrue(
                any(
                    "foundation median drift" in reason
                    for reason in run.report["timing_screen"]["reasons"]
                )
            )

    def test_reanalysis_of_all_three_retained_v2_sessions(self):
        evidence = read_json(BENCHMARKS / "tests/fixtures/local-v2-recordings.json")
        self.assertEqual(len(evidence), 3)
        self.assertEqual(
            sum(row["original_verdict"] == "inconclusive" for row in evidence), 2
        )
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            paths = fixtures.FoundationBounds().records(root, "local-quick")
            for path, saved in zip(paths, evidence):
                row = read_json(path / "report.json")
                row["contract"]["profile"]["screening_policy"] = "local-batches-v2"
                row["contract"]["harness_sha256"] = local_screening.REANALYZABLE_V2
                row["verdict"] = saved["original_verdict"]
                for name, times in saved["samples"].items():
                    for sample, time in zip(row["samples"][name]["foundation"], times):
                        sample["wall_ms"] = time
                write_json(path / "report.json", row)
            output = root / "reanalysis.json"
            self.assertEqual(foundation(paths, output), "calibrated")
            analyzed = read_json(output)
            self.assertEqual(
                analyzed["source_contract"]["harness_sha256"],
                local_screening.REANALYZABLE_V2,
            )
            self.assertEqual(
                analyzed["contract"]["profile"]["screening_policy"],
                local_screening.POLICY,
            )
            self.assertGreater(
                analyzed["estimates"]["novel_text"]["raw_summaries"][0]["cv"], 0.10
            )
            self.assertLess(
                analyzed["estimates"]["novel_text"]["relative_median_uncertainty"], 0.10
            )
            for name, info in analyzed["estimates"].items():
                self.assertEqual(len(info["batch_medians_ms"]), 3)
                self.assertTrue(
                    all(len(batch) == 3 for batch in info["batch_medians_ms"])
                )
            self.assertEqual(
                read_json(paths[0] / "report.json")["verdict"], "inconclusive"
            )

    def test_uncertainty_and_drift_have_separate_gates(self):
        stable = local_screening.estimate([[100] * 8 + [400]] * 3)
        self.assertEqual(stable["median_interval_ms"], [100, 100])
        self.assertEqual(stable["reasons"], [])
        drifted = local_screening.estimate([[100] * 9, [100] * 9, [140] * 9])
        self.assertIn("batch median drift exceeds 15%", drifted["reasons"])
        imprecise = local_screening.estimate([[89] * 3 + [100] * 3 + [111] * 3])
        self.assertIn("median uncertainty exceeds 10%", imprecise["reasons"])
        self.assertNotIn("batch median drift exceeds 15%", imprecise["reasons"])
        self.assertEqual(stable, local_screening.estimate([[100] * 8 + [400]] * 3))

    def test_candidate_triage_uses_uncertainty_and_never_qualifies(self):
        with tempfile.TemporaryDirectory() as root:
            run = self.run_with_samples(root, "compare")
            for roles in run.report["samples"].values():
                for role, rows in roles.items():
                    for row in rows:
                        if role == "candidate":
                            row["wall_ms"] *= 0.8
                        row["peak_rss_bytes"] = 1000
                        row["index_bytes"] = 2000
            run.summarize()
            self.assertEqual(run.report["verdict"], "promising")
            self.assertIn("not qualified", run.report["reasons"][-1])

        resource = {
            "novel_text:" + field: {"interval": [0, 0]}
            for field in ("peak_rss_bytes", "index_bytes")
        }
        for interval, improvement, expected in (
            ([0.1, 0.2], 0.15, "promising"),
            ([-0.1, 0.2], 0.1, "inconclusive"),
            ([-0.2, -0.1], -0.15, "does_not_qualify"),
            ([-0.01, 0.01], 0, "does_not_qualify"),
        ):
            comparison = {
                "novel_text": {"interval": interval, "improvement": improvement}
            }
            result, _ = local_screening.candidate_decision(
                comparison, resource, ["novel_text"]
            )
            self.assertEqual(result, expected)
        result, _ = local_screening.candidate_decision(comparison, {}, ["novel_text"])
        self.assertEqual(result, "inconclusive")

    def test_only_known_measurement_contract_can_be_reanalyzed(self):
        profile = read_json(BENCHMARKS / "profiles/local-quick.json")
        source = {
            "profile": profile | {"screening_policy": "local-batches-v2"},
            "harness_sha256": local_screening.REANALYZABLE_V2,
        }
        self.assertTrue(local_screening.compatible_recording(source))
        self.assertFalse(
            local_screening.compatible_recording(source | {"harness_sha256": "unknown"})
        )
        self.assertFalse(
            local_screening.compatible_recording(
                source | {"profile": source["profile"] | {"samples": 5}}
            )
        )

    def run_with_samples(self, root, mode="record", times=None):
        profile = read_json(BENCHMARKS / "profiles/local-quick.json") | {
            "quality": False
        }
        run = Run(
            {
                "directory": str(root),
                "profile": profile,
                "mode": mode,
                "max_seconds": 60,
                "hourly_budget_usd": 0,
            }
        )
        times = times or [100] * 8 + [200]
        for name in profile["scenarios"]:
            run.report["samples"][name] = {
                role: [
                    {
                        "wall_ms": t,
                        "timing": {"phases": []},
                        "behavior": {"passed": True},
                        "behavior_comparison": {"passed": True},
                    }
                    for t in times
                ]
                for role in (
                    ["foundation"] if mode == "record" else ["foundation", "candidate"]
                )
            }
        return run

    def test_isolated_spikes_remain_in_diagnostics_without_invalidating_median(self):
        with tempfile.TemporaryDirectory() as root:
            for mode in ("record", "validate", "compare"):
                run = self.run_with_samples(root, mode)
                run.summarize()
                self.assertEqual(
                    run.report["verdict"],
                    {
                        "record": "foundation_recorded",
                        "validate": "validation_passed",
                        "compare": "inconclusive",
                    }[mode],
                )
                self.assertGreater(
                    run.report["timing_screen"]["estimates"]["novel_text"][
                        "foundation"
                    ]["raw_summaries"][0]["cv"],
                    0.10,
                )
                self.assertEqual(
                    run.report["summaries"]["novel_text"]["foundation"]["wall_ms"][
                        "samples"
                    ],
                    9,
                )
                self.assertEqual(
                    run.report["samples"]["novel_text"]["foundation"][-1]["wall_ms"],
                    200,
                )

    def test_noise_does_not_hide_behavior_or_quality_failure(self):
        with tempfile.TemporaryDirectory() as root:
            for mode, expected in (
                ("record", "invalid"),
                ("compare", "does_not_qualify"),
                ("validate", "validation_failed"),
            ):
                run = self.run_with_samples(root, mode)
                run.report["samples"]["cached_text"]["foundation"][0]["behavior"] = {
                    "passed": False,
                    "reasons": ["bad output"],
                }
                run.summarize()
                self.assertEqual(run.report["verdict"], expected)
            run = self.run_with_samples(root, "compare")
            run.report["quality_comparison"] = {"passed": False}
            run.summarize()
            self.assertEqual(run.report["verdict"], "does_not_qualify")
            run = self.run_with_samples(root)
            run.profile["quality"] = True
            run.summarize()
            self.assertEqual(run.report["verdict"], "invalid")

    def test_missing_samples_block_even_when_other_samples_are_noisy(self):
        with tempfile.TemporaryDirectory() as root:
            run = self.run_with_samples(root)
            run.report["samples"]["indexed_image"]["foundation"].pop()
            with self.assertRaisesRegex(ValueError, "incomplete local"):
                run.summarize()

    def test_noise_preserves_definite_resource_failure(self):
        with tempfile.TemporaryDirectory() as root:
            run = self.run_with_samples(root, "compare")
            for roles in run.report["samples"].values():
                for role, rows in roles.items():
                    for row in rows:
                        row["peak_rss_bytes"] = 100 if role == "foundation" else 200
            run.summarize()
            self.assertEqual(run.report["verdict"], "does_not_qualify")
            self.assertTrue(
                any("memory/index" in reason for reason in run.report["reasons"])
            )

    def test_calibration_noise_finishes_all_batches_and_scenarios(self):
        class Scenario:
            calls = 0

            def __init__(self, *args):
                pass

            def sample(self, index):
                Scenario.calls += 1
                return {
                    "wall_ms": 100 if index < 6 else 200,
                    "behavior": {"passed": True},
                }

            def cleanup(self):
                pass

        with tempfile.TemporaryDirectory() as root:
            run = self.run_with_samples(root)
            with (
                patch("harness.runner.Scenario", Scenario),
                patch.object(run, "progress"),
                patch.object(run, "save"),
                patch.object(run, "discard_inputs"),
            ):
                run.calibrate(Path("binary"), Path("cache"), {}, None)
            self.assertEqual(Scenario.calls, 18)
            self.assertTrue(run.report["timing_screen"]["reasons"])
            run.summarize()
            self.assertEqual(run.report["verdict"], "inconclusive")

    def test_all_three_batches_contribute_and_noisy_foundation_has_no_bounds(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            paths = fixtures.FoundationBounds().records(
                root, "local-quick", [98] * 3 + [100] * 3 + [102] * 3
            )
            destination = root / "good.json"
            self.assertEqual(foundation(paths, destination), "calibrated")
            self.assertEqual(
                read_json(destination)["calibration_batch_medians"]["novel_text"],
                [98, 100, 102] * 3,
            )
            row = read_json(paths[-1] / "report.json")
            row["verdict"] = "inconclusive"
            for sample in row["samples"]["modified_1pct"]["foundation"][-3:]:
                sample["wall_ms"] = 300
            write_json(paths[-1] / "report.json", row)
            destination = root / "noisy.json"
            self.assertEqual(foundation(paths, destination), "inconclusive")
            self.assertEqual(read_json(destination)["calibration_bounds"], {})
            self.assertEqual(len(read_json(destination)["reports"]), 3)
            with self.assertRaisesRegex(ValueError, "immutable"):
                foundation(paths, destination)

    def test_legacy_recordings_require_fresh_samples(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            paths = fixtures.FoundationBounds().records(root, "local-quick")
            for path in paths:
                row = read_json(path / "report.json")
                row["contract"]["profile"].pop("screening_policy")
                write_json(path / "report.json", row)
            with self.assertRaisesRegex(ValueError, "fresh recordings"):
                foundation(paths, root / "foundation.json")

    def test_cloud_profiles_do_not_enable_local_policy(self):
        for name in ("cloud-standard", "cloud-scale"):
            self.assertFalse(
                local_screening.enabled(
                    read_json(BENCHMARKS / "profiles" / (name + ".json"))
                )
            )
        # Cloud still rejects excessive within-session variation.
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            paths = fixtures.FoundationBounds().records(root, values=[100] * 20 + [200])
            with self.assertRaisesRegex(ValueError, "variation too large"):
                foundation(paths, root / "cloud.json")

    def test_retained_evidence_includes_failures_without_reclassifying(self):
        evidence = read_json(BENCHMARKS / "results/local-noise-audit-20260908.json")
        self.assertEqual(len(evidence["reports"]), 12)
        self.assertEqual(
            sum(r["original_verdict"] == "invalid" for r in evidence["reports"]), 4
        )
        self.assertTrue(
            all(
                r["eligibility"] == "fresh recordings required"
                for r in evidence["reports"]
            )
        )
        for report in evidence["reports"]:
            for roles in report["scenarios"].values():
                for row in roles.values():
                    self.assertIn(
                        "rehearsal only", local_screening.uncertainty(row["wall_ms"])[0]
                    )
                    with self.assertRaisesRegex(ValueError, "complete triples"):
                        local_screening.estimate([row["wall_ms"]])


if __name__ == "__main__":
    unittest.main()
