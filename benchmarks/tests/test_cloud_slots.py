"""Cloud launch slots isolate leases and retain conditional ownership checks."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench import configuration, parser
from harness import cloud
from harness.storage import FOUNDATION, write_json


class CloudSlots(unittest.TestCase):
    def test_legacy_key_and_bounded_slot_names(self):
        self.assertEqual(cloud.lock_key(), "control/active.json")
        self.assertEqual(len({cloud.lock_key(slot) for slot in (1, 2, 3)}), 3)
        for slot in (0, 4, -1):
            with self.assertRaises(ValueError):
                cloud.lock_key(slot)

    @patch("bench.command", return_value=FOUNDATION)
    def test_slot_does_not_change_profile_or_per_run_budget(self, git):
        configs = [configuration(parser().parse_args([
            "plan", "--profile", "cloud-standard", "--mode", "record",
            "--cloud", "/tmp/settings.json", "--cloud-slot", str(slot),
        ])) for slot in (1, 2, 3)]
        self.assertEqual([c["cloud_slot"] for c in configs], [1, 2, 3])
        for field in ("profile", "budget_usd", "max_seconds"):
            self.assertEqual(configs[0][field], configs[1][field])
            self.assertEqual(configs[1][field], configs[2][field])

    def test_parallel_slot_is_rejected_for_local_run(self):
        with self.assertRaisesRegex(ValueError, "requires --cloud"):
            configuration(parser().parse_args(["plan", "--cloud-slot", "2"]))

    def test_acquisition_is_atomic_per_slot(self):
        acquired = set()

        def aws(region, service, action, *args):
            if action == "describe-images":
                return {"Images": [{"Architecture": "x86_64", "RootDeviceType": "ebs"}]}
            self.assertEqual(action, "put-object")
            self.assertEqual(args[args.index("--if-none-match") + 1], "*")
            key = args[args.index("--key") + 1]
            if key in acquired:
                raise RuntimeError("slot occupied")
            acquired.add(key)
            return {"ETag": "lease"}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = root / "settings.json"
            write_json(settings, {"region": "us-east-1", "bucket": "bucket", "ami": "ami-test"})
            with (
                patch("harness.cloud.validate_settings"),
                patch("harness.cloud.aws", side_effect=aws),
                patch("harness.cloud.tempfile.TemporaryDirectory", side_effect=RuntimeError("stop after lease")),
            ):
                for number, slot in enumerate((1, 2, 3, 2)):
                    directory = root / str(number)
                    directory.mkdir()
                    config = {
                        "directory": str(directory), "cloud_slot": slot,
                        "profile": {"cloud": True}, "hourly_budget_usd": 0.26,
                        "max_seconds": 60,
                    }
                    expected = "slot occupied" if number == 3 else "stop after lease"
                    with self.assertRaisesRegex(RuntimeError, expected):
                        cloud.launch(config, settings)
            self.assertEqual(acquired, {cloud.lock_key(s) for s in (1, 2, 3)})

    def test_collection_deletes_only_its_unchanged_lease(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for slot in (None, 2, 3):
                for owner in ("mine", "another-run"):
                    with self.subTest(slot=slot, owner=owner):
                        handle = {"region": "us-east-1", "bucket": "bucket", "run_id": "mine", "prefix": "runs/mine"}
                        if slot is not None:
                            handle["cloud_slot"] = slot
                        write_json(root / "cloud.json", handle)
                        deleted = []

                        def aws(region, service, action, *args):
                            if action == "list-objects-v2":
                                return {"Contents": [{"Key": cloud.lock_key(slot or 1)}]}
                            self.assertEqual(args[args.index("--key") + 1], cloud.lock_key(slot or 1))
                            if action == "get-object":
                                write_json(Path(args[-1]), {"run_id": owner})
                                return {"ETag": "observed-lease"}
                            self.assertEqual(action, "delete-object")
                            self.assertEqual(args[args.index("--if-match") + 1], "observed-lease")
                            deleted.append(action)
                            return {}

                        with (
                            patch("harness.cloud.aws", side_effect=aws),
                            patch("harness.cloud.command"),
                            patch("harness.cloud.status", return_value={"instance_state": "terminated"}),
                        ):
                            if owner == "mine":
                                cloud.collect(root)
                                self.assertEqual(deleted, ["delete-object"])
                            else:
                                cloud.collect(root)
                                self.assertEqual(deleted, [])

    def test_reconciliation_respects_slot_expiry_and_run_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = root / "settings.json"
            write_json(settings, {"region": "us-east-1", "bucket": "bucket"})
            for expired in (False, True):
                deleted = []

                def aws(region, service, action, *args):
                    if action == "describe-instances":
                        filters = json.loads(args[args.index("--filters") + 1])
                        self.assertIn({"Name": "tag:RunId", "Values": ["slot-two-run"]}, filters)
                        return {"Reservations": []}
                    self.assertEqual(args[args.index("--key") + 1], cloud.lock_key(2))
                    if action == "get-object":
                        write_json(Path(args[-1]), {"run_id": "slot-two-run", "expires": 0 if expired else 200})
                        return {"ETag": "lease-two"}
                    self.assertEqual(args[args.index("--if-match") + 1], "lease-two")
                    deleted.append(action)
                    return {}

                with (
                    patch("harness.cloud.validate_settings"),
                    patch("harness.cloud.aws", side_effect=aws),
                    patch("harness.cloud.time.time", return_value=100),
                ):
                    if expired:
                        cloud.reconcile(settings, slot=2)
                        self.assertEqual(deleted, ["delete-object"])
                    else:
                        with self.assertRaisesRegex(ValueError, "has not expired"):
                            cloud.reconcile(settings, slot=2)
                        self.assertEqual(deleted, [])
