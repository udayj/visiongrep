"""Bounded settling must reject activity rather than accept a fixed sleep."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.quiescence import quiet, settle


IDLE = {
    "dirty_kib": 100, "writeback_kib": 0, "reads_completed": 10,
    "writes_completed": 20, "sectors_written": 30, "in_flight": 0, "io_ms": 40,
}


class StorageQuiescence(unittest.TestCase):
    def test_activity_or_backlog_rejects_quiet_window(self):
        self.assertTrue(quiet(IDLE, IDLE))
        for key, value in (
            ("dirty_kib", 20000), ("writeback_kib", 1), ("in_flight", 1),
            ("reads_completed", 11), ("writes_completed", 21),
            ("sectors_written", 31), ("io_ms", 41),
        ):
            with self.subTest(key=key):
                self.assertFalse(quiet(IDLE, IDLE | {key: value}))

    def test_flush_precedes_observed_quiet_window(self):
        clock = [0.0]
        events = []

        def read(device):
            events.append("observe")
            return IDLE.copy()

        with (
            patch("harness.quiescence.storage_state", side_effect=read),
            patch("harness.quiescence.subprocess.run", side_effect=lambda *a, **kw: events.append("flush")),
            patch("harness.quiescence.time.monotonic", side_effect=lambda: clock[0]),
            patch("harness.quiescence.time.sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)),
        ):
            result = settle(Path(__file__).parent)
        self.assertEqual(events[:3], ["observe", "flush", "observe"])
        self.assertGreaterEqual(result["total_ms"], 250)
        self.assertEqual(result["after"], IDLE)

    def test_persistent_writeback_times_out(self):
        clock = [0.0]
        with (
            patch("harness.quiescence.storage_state", return_value=IDLE | {"writeback_kib": 100}),
            patch("harness.quiescence.subprocess.run"),
            patch("harness.quiescence.time.monotonic", side_effect=lambda: clock[0]),
            patch("harness.quiescence.time.sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)),
            self.assertRaisesRegex(TimeoutError, "did not settle"),
        ):
            settle(Path(__file__).parent)

    def test_flush_failure_is_propagated(self):
        with (
            patch("harness.quiescence.storage_state", return_value=IDLE),
            patch("harness.quiescence.subprocess.run", side_effect=TimeoutError("flush timeout")),
            self.assertRaisesRegex(TimeoutError, "flush timeout"),
        ):
            settle(Path(__file__).parent)

    def test_flush_using_entire_budget_reports_timeout(self):
        with (
            patch("harness.quiescence.storage_state", return_value=IDLE),
            patch("harness.quiescence.subprocess.run"),
            patch("harness.quiescence.time.monotonic", side_effect=[0, 31, 31]),
            self.assertRaisesRegex(TimeoutError, "did not settle"),
        ):
            settle(Path(__file__).parent)
