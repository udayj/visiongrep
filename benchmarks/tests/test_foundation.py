"""Reference bounds come from recorded scenarios, without duplicate measurements."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.runner import Run, calibration_scenarios, foundation
from harness.storage import BENCHMARKS, FOUNDATION, digest, read_json, write_json


class FoundationRecording(unittest.TestCase):
    def test_only_recording_skips_precheck(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus.json"
            write_json(corpus, {"images": []})
            binary = root / "binary"
            binary.write_bytes(b"binary")
            baseline = root / "baseline.json"
            measured_contract = {"build_environment": {"rustc": "pinned"}}
            write_json(
                baseline, {"contract": measured_contract, "verdict": "calibrated"}
            )
            for mode in ("record", "validate", "compare"):
                config = {
                    "directory": str(root / mode),
                    "profile": {"name": "local-quick", "cloud": False},
                    "max_seconds": 60,
                    "hourly_budget_usd": 0,
                    "cache": str(root / "cache"),
                    "corpus": str(corpus),
                    "corpus_sha256": digest(corpus),
                    "mode": mode,
                    "candidate": FOUNDATION,
                    "baseline": str(baseline) if mode == "compare" else None,
                }
                run = Run(config)
                with (
                    self.subTest(mode=mode),
                    patch("harness.runner.environment", return_value={}),
                    patch("harness.runner.platform.system", return_value="Darwin"),
                    patch("harness.runner.verify"),
                    patch("harness.runner.contract", return_value={}),
                    patch(
                        "harness.runner.build",
                        return_value=(binary, FOUNDATION, {"rustc": "pinned"}),
                    ),
                    patch.object(run, "progress"),
                    patch.object(run, "calibrate") as calibrate,
                    patch.object(run, "measure") as measure,
                ):
                    run.perform()
                    self.assertEqual(calibrate.call_count, mode != "record")
                    measure.assert_called_once()
                    self.assertEqual(run.report["calibration"], {})
                    if mode == "compare":
                        write_json(
                            baseline,
                            {
                                "contract": measured_contract,
                                "verdict": "inconclusive",
                            },
                        )
                        with self.assertRaisesRegex(ValueError, "not calibrated"):
                            run.perform()

    def test_scale_precheck_matches_recorded_corpus_and_retains_drift_guard(self):
        corpora_seen = []

        class Scenario:
            def __init__(self, root, binary, cache, corpus, name, invoke):
                corpora_seen.append(len(corpus["images"]))

            def sample(self, index):
                return {"wall_ms": 100, "behavior": {"passed": True}}

            def cleanup(self):
                pass

        profile = read_json(BENCHMARKS / "profiles/cloud-scale.json")
        with tempfile.TemporaryDirectory() as temporary:
            run = Run(
                {
                    "directory": temporary,
                    "profile": profile,
                    "max_seconds": 60,
                    "hourly_budget_usd": 0,
                }
            )
            bounds = {name: [97, 103] for name in calibration_scenarios(profile)}
            corpus = {"images": [None] * 10000}
            with (
                patch("harness.runner.Scenario", Scenario),
                patch.object(run, "progress"),
                patch.object(run, "save"),
                patch.object(run, "discard_inputs"),
            ):
                run.calibrate(
                    Path("binary"),
                    Path("cache"),
                    corpus,
                    {"calibration_bounds": bounds},
                )
                self.assertEqual(corpora_seen, [10000, 10000])
                self.assertEqual(
                    set(run.report["calibration"]), {"novel_text", "cached_text"}
                )
                self.assertTrue(
                    all(len(rows) == 9 for rows in run.report["calibration"].values())
                )
                with self.assertRaisesRegex(ValueError, "calibration failed"):
                    run.calibrate(
                        Path("binary"),
                        Path("cache"),
                        corpus,
                        {"calibration_bounds": {name: [50, 60] for name in bounds}},
                    )


class FoundationBounds(unittest.TestCase):
    def records(self, root, profile_name="cloud-standard", values=None):
        profile = read_json(BENCHMARKS / "profiles" / (profile_name + ".json"))
        paths = []
        for number in range(3):
            path = root / str(number)
            times = values if values is not None else [100] * profile["samples"]
            write_json(
                path / "report.json",
                {
                    "verdict": "foundation_recorded",
                    "config": {"mode": "record"},
                    "quality": {
                        "foundation": {
                            "runs": [{}]
                            * len(
                                read_json(BENCHMARKS / "corpora/quality-500.json")[
                                    "queries"
                                ]
                            )
                        }
                    },
                    "contract": {
                        "profile": profile,
                        "environment": {
                            "ami": "ami-test" if profile["cloud"] else None
                        },
                    },
                    "environment": {"instance_id": str(number)},
                    "samples": {
                        name: {
                            "foundation": [
                                {"wall_ms": x, "behavior": {"passed": True}}
                                for x in times
                            ]
                        }
                        for name in profile["scenarios"]
                    },
                },
            )
            paths.append(path)
        return paths

    def test_all_profiles_record_without_separate_calibration(self):
        for name in ("local-quick", "cloud-standard", "cloud-scale"):
            with self.subTest(profile=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                paths = self.records(root, name)
                destination = root / "foundation.json"
                foundation(paths, destination)
                expected = {"novel_text", "cached_text"}
                if name == "cloud-standard":
                    expected.add("no_cache")
                self.assertEqual(
                    read_json(destination)["calibration_bounds"],
                    {key: [97, 103] for key in expected},
                )

    def test_all_seven_triples_contribute_to_cloud_bounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            times = [
                value for value in (98, 99, 100, 101, 102, 103, 104) for _ in range(3)
            ]
            paths = self.records(root, values=times)
            destination = root / "foundation.json"
            foundation(paths, destination)
            lo, hi = read_json(destination)["calibration_bounds"]["novel_text"]
            self.assertAlmostEqual((lo + hi) / 2, 101)
            self.assertAlmostEqual(hi - 101, 2 * 1.4826 * 3)

    def test_missing_samples_are_not_replaced_by_old_calibration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.records(root)
            row = read_json(paths[0] / "report.json")
            row["calibration"] = {"novel_text": [{"wall_ms": 100}] * 9}
            row["samples"]["novel_text"]["foundation"].pop()
            write_json(paths[0] / "report.json", row)
            with self.assertRaisesRegex(ValueError, "incomplete foundation"):
                foundation(paths, root / "foundation.json")

    def test_five_samples_cannot_supply_the_new_local_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.records(root, "local-quick", [100, 100, 100, 100, 200])
            with self.assertRaisesRegex(ValueError, "incomplete foundation"):
                foundation(paths, root / "foundation.json")

    def test_drift_between_instances_rejects_foundation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.records(root)
            row = read_json(paths[2] / "report.json")
            row["samples"]["novel_text"]["foundation"] = [{"wall_ms": 130}] * 21
            write_json(paths[2] / "report.json", row)
            with self.assertRaisesRegex(ValueError, "variation too large|outlying"):
                foundation(paths, root / "foundation.json")


if __name__ == "__main__":
    unittest.main()
