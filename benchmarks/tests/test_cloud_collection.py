"""Archived cloud recordings remain collectible after EC2 forgets the instance."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import cloud
from harness.storage import write_json


class CloudCollection(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        write_json(self.root / "cloud.json", {
            "region": "us-east-1", "bucket": "bucket", "instance_id": "i-test",
            "run_id": "mine", "prefix": "runs/mine",
        })

    def test_missing_instance_retains_archived_progress(self):
        for reservations in ([], [{"Instances": []}]):
            def aws(region, service, action, *args):
                if action == "describe-instances":
                    self.assertIn("--filters", args)
                    return {"Reservations": reservations}
                write_json(Path(args[-1]), {"stage": "complete"})
                return {}

            with patch("harness.cloud.aws", side_effect=aws):
                result = cloud.status(self.root)
            self.assertEqual(result["instance_state"], "not_found")
            self.assertEqual(result["progress"]["stage"], "complete")

    def test_ec2_errors_are_not_treated_as_missing(self):
        with patch("harness.cloud.aws", side_effect=RuntimeError("access denied")):
            with self.assertRaisesRegex(RuntimeError, "access denied"):
                cloud.status(self.root)

    def test_collection_can_resume_without_instance_or_lease(self):
        with (
            patch("harness.cloud.status", return_value={"instance_state": "not_found"}),
            patch("harness.cloud.command") as sync,
            patch("harness.cloud.aws", return_value={}) as aws,
        ):
            cloud.collect(self.root)
            cloud.collect(self.root)
        self.assertEqual(sync.call_count, 2)
        self.assertEqual([call.args[2] for call in aws.call_args_list],
                         ["list-objects-v2", "list-objects-v2"])
        self.assertNotIn("--delete", sync.call_args.args[0])

    def test_running_instance_blocks_collection(self):
        with (
            patch("harness.cloud.status", return_value={"instance_state": "running"}),
            patch("harness.cloud.command") as sync,
        ):
            with self.assertRaisesRegex(ValueError, "not yet terminated"):
                cloud.collect(self.root)
        sync.assert_not_called()

    def test_interrupted_download_preserves_lease(self):
        with (
            patch("harness.cloud.status", return_value={"instance_state": "terminated"}),
            patch("harness.cloud.command", side_effect=RuntimeError("interrupted")),
            patch("harness.cloud.aws") as aws,
        ):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                cloud.collect(self.root)
        aws.assert_not_called()
