"""Reference bounds come from recorded scenarios, without duplicate measurements."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.runner import Run, calibration_scenarios, foundation
from harness import calibration
from harness.storage import (
    BENCHMARKS,
    FOUNDATION,
    digest,
    read_json,
    write_json,
    harness_digest,
)


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
                        "harness_sha256": harness_digest(),
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
                    {
                        key: [90, 110.00000000000001]
                        for key in expected
                    },
                )

    def test_whole_session_median_sets_cloud_bounds(self):
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
            self.assertAlmostEqual(hi - 101, 10.1)

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
            row["samples"]["novel_text"]["foundation"] = [{"wall_ms": 130, "behavior": {"passed": True}}] * 21
            write_json(paths[2] / "report.json", row)
            with self.assertRaisesRegex(ValueError, "variation too large|outlying"):
                foundation(paths, root / "foundation.json")


class SessionCalibration(unittest.TestCase):
    def test_isolated_batch_does_not_override_typical_latency(self):
        # A 12% slower triple is compatible with a stable session overall.
        info = calibration.reference([[100] * 21, [100] * 21, [105] * 18 + [112] * 3])
        self.assertAlmostEqual(info["median_ms"], 100)
        self.assertEqual([s["samples"] for s in info["sessions"]], [21, 21, 21])
        self.assertAlmostEqual(info["bounds_ms"][1], 110)

    def test_material_drift_and_noisy_sessions_are_rejected(self):
        for sessions in (
            [[100] * 21, [100] * 21, [111] * 21],
            [[100] * 20 + [200], [100] * 21, [100] * 21],
            [[0] * 21, [100] * 21, [100] * 21],
        ):
            with self.subTest(sessions=sessions), self.assertRaises(ValueError):
                calibration.reference(sessions)

    def test_known_migration_preserves_source_contracts_and_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = FoundationBounds().records(root)
            originals = []
            for number, path in enumerate(paths):
                row = read_json(path / "report.json")
                row["contract"]["harness_sha256"] = calibration.REANALYZABLE_V3
                row["contract"]["environment"]["hardware"] = {
                    "model name": "same CPU",
                    "microcode": str(number),
                    "flags": "same flags",
                }
                write_json(path / "report.json", row)
                originals.append(row["contract"])
            hashes = [digest(path / "report.json") for path in paths]
            output = root / "foundation.json"
            self.assertEqual(foundation(paths, output), "calibrated")
            result = read_json(output)
            self.assertEqual(result["source_contracts"], originals)
            self.assertEqual(result["contract"]["harness_sha256"], harness_digest())
            self.assertNotIn("microcode", result["contract"]["environment"]["hardware"])
            self.assertEqual([item["sha256"] for item in result["reports"]], hashes)
            self.assertEqual([digest(path / "report.json") for path in paths], hashes)

    def test_other_contract_changes_and_unknown_harness_are_rejected(self):
        for field in ("harness_sha256", "ami", "flags", "build_environment"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                paths = FoundationBounds().records(root)
                row = read_json(paths[2] / "report.json")
                if field == "ami":
                    row["contract"]["environment"][field] = "different"
                elif field == "flags":
                    row["contract"]["environment"]["hardware"] = {"flags": "different"}
                else:
                    row["contract"][field] = "different"
                write_json(paths[2] / "report.json", row)
                with self.assertRaisesRegex(
                    ValueError, "different contracts|unknown measurement"
                ):
                    foundation(paths, root / "foundation.json")

    def test_incomplete_behavior_quality_and_duplicate_instances_are_rejected(self):
        for failure in ("behavior", "quality", "instance"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                paths = FoundationBounds().records(root)
                row = read_json(paths[2] / "report.json")
                if failure == "behavior":
                    row["samples"]["indexed_image"]["foundation"][0].pop("behavior")
                elif failure == "quality":
                    row["quality"]["foundation"]["runs"].pop()
                else:
                    row["environment"]["instance_id"] = "0"
                write_json(paths[2] / "report.json", row)
                with self.assertRaises(ValueError):
                    foundation(paths, root / "foundation.json")

    def test_local_v3_recordings_remain_reanalyzable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = FoundationBounds().records(root, "local-quick")
            for path in paths:
                row = read_json(path / "report.json")
                row["contract"]["harness_sha256"] = calibration.REANALYZABLE_V3
                write_json(path / "report.json", row)
            self.assertEqual(foundation(paths, root / "local.json"), "calibrated")

    def test_cloud_precheck_uses_session_median_and_retains_drift_guard(self):
        times = [100] * 6 + [112] * 3

        class Scenario:
            def __init__(self, *args):
                pass

            def sample(self, index):
                return {"wall_ms": times[index], "behavior": {"passed": True}}

            def cleanup(self):
                pass

        profile = read_json(BENCHMARKS / "profiles/cloud-standard.json")
        with tempfile.TemporaryDirectory() as temporary:
            run = Run(
                {
                    "directory": temporary,
                    "profile": profile,
                    "max_seconds": 60,
                    "hourly_budget_usd": 0,
                }
            )
            baseline = {
                "calibration_bounds": {
                    name: [90, 110] for name in calibration_scenarios(profile)
                }
            }
            with (
                patch("harness.runner.Scenario", Scenario),
                patch.object(run, "progress"),
                patch.object(run, "save"),
                patch.object(run, "discard_inputs"),
            ):
                run.calibrate(Path("binary"), Path("cache"), {}, baseline)
                times[:] = [111] * 9
                with self.assertRaisesRegex(ValueError, "calibration failed"):
                    run.calibrate(Path("binary"), Path("cache"), {}, baseline)


if __name__ == "__main__":
    unittest.main()
