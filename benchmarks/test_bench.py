import contextlib
import datetime as dt
import io
import json
import math
import sys
import unittest
from unittest.mock import patch

from benchmarks import aws, bench, quality


class ComparisonTests(unittest.TestCase):
    def test_change_and_paired_bootstrap(self):
        change, low, high = bench.change_interval([10, 20, 30], [5, 10, 15])
        self.assertAlmostEqual(change, 0.5)
        self.assertAlmostEqual(low, 0.5)
        self.assertAlmostEqual(high, 0.5)
        self.assertEqual((change, low, high), bench.change_interval([10, 20, 30], [5, 10, 15]))
        change, low, high = bench.change_interval([10, 20], [5, 20])
        self.assertAlmostEqual(change, 1 / 6)
        self.assertEqual((low, high), (0, 0.5))

    def test_same_samples_have_zero_change(self):
        self.assertEqual(bench.change_interval([2, 5, 3, 7], [2, 5, 3, 7]), (0, 0, 0))

    def test_output_comparison(self):
        rows = [{"path": "a.jpg", "score": 0.8}, {"path": "b.jpg", "score": 0.5}]
        self.assertIsNone(bench.output_difference(rows, rows))
        self.assertIsNotNone(bench.output_difference(rows, rows[::-1]))
        self.assertIsNone(bench.output_difference(rows, [rows[0] | {"score": 0.80009}, rows[1]]))
        self.assertIsNotNone(bench.output_difference(rows, [rows[0] | {"score": 0.80011}, rows[1]]))
        self.assertIsNotNone(bench.output_difference(rows, rows[:1]))
        self.assertIsNotNone(bench.output_difference(rows, [rows[0] | {"score": float("nan")}, rows[1]]))

    def test_abba_order(self):
        self.assertEqual([bench.sample_order(i) for i in range(4)], [(0, 1), (1, 0), (0, 1), (1, 0)])

    def test_rss_units(self):
        self.assertEqual(bench.rss_bytes(123, "Darwin"), 123)
        self.assertEqual(bench.rss_bytes(123, "Linux"), 123 * 1024)


class QualityTests(unittest.TestCase):
    def test_quality_metrics(self):
        runs = [
            {"kind": "caption", "relevant": ["a", "b"], "results": [{"path": "a"}, {"path": "x"}, {"path": "b"}]},
            {"kind": "caption", "relevant": ["c"], "results": [{"path": "x"}, {"path": "c"}]},
            {"kind": "category", "intent": "one", "relevant": ["a"], "results": [{"path": "a", "score": 0.8}, {"path": "x", "score": 0.7}]},
            {"kind": "category", "intent": "one", "relevant": ["a"], "results": [{"path": "a", "score": 0.1}]},
            {"kind": "category", "intent": "two", "relevant": ["b"], "results": [{"path": "b", "score": 0.8}]},
        ]
        metrics = quality.evaluate(runs, 0.5)
        self.assertAlmostEqual(metrics["recall_at_1"], 0.25)
        self.assertAlmostEqual(metrics["recall_at_5"], 1)
        self.assertAlmostEqual(metrics["recall_at_10"], 1)
        self.assertAlmostEqual(metrics["mrr_at_10"], 0.75)
        ndcg_first = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(metrics["ndcg_at_10"], (ndcg_first + 1 / math.log2(3)) / 2)
        self.assertAlmostEqual(metrics["absent_false_positive_rate"], 0.25)
        self.assertAlmostEqual(metrics["present_false_negative_rate"], 0.25)

    def test_top_ten_agreement(self):
        reference = [{"id": "one", "results": [{"path": "a"}, {"path": "b"}]},
                     {"id": "two", "results": [{"path": "c"}]}]
        candidate = [{"id": "one", "results": [{"path": "b"}, {"path": "a"}]},
                     {"id": "two", "results": [{"path": "c"}]}]
        self.assertEqual(quality.top10_agreement(reference, candidate), 0.5)
        with self.assertRaises(ValueError):
            quality.top10_agreement(reference, candidate[::-1])


class AwsTests(unittest.TestCase):
    def setUp(self):
        self.settings = {
            "region": "us-east-1", "subnet": "subnet-1", "security_group": "sg-1",
            "instance_profile": "worker", "bucket": "bench-bucket", "ami": "ami-1",
            "scheduler_role": "arn:aws:iam::123:role/scheduler",
        }
        self.ref = "a" * 40
        self.cand = "b" * 40

    def test_user_data_and_instance_command(self):
        flags = ["--samples", "5", "--scenarios", "all", "--allow-output-changes"]
        command_line = ["python3", "benchmarks/bench.py", "compare", self.ref, self.cand, *flags]
        script = aws.user_data(self.settings, "run-1", self.cand, command_line, 60)
        self.assertEqual(script.splitlines()[1], "shutdown -h +60")
        self.assertEqual(script.splitlines()[2], "export HOME=/root")
        before_home = script.split("export HOME=/root", 1)[0]
        self.assertNotIn("$HOME", before_home)
        self.assertNotIn("${HOME", before_home)
        self.assertIn("$HOME", script)
        self.assertIn("trap finish EXIT", script)
        self.assertIn("s3://bench-bucket/runs/run-1/", script)
        self.assertIn(f"git checkout --detach {self.cand}", script)
        self.assertIn(f"compare {self.ref} {self.cand} --samples 5 --scenarios all --allow-output-changes", script)
        args = aws.instance_command(self.settings, "run-1", "c7a.xlarge", script)
        self.assertEqual(args[args.index("--instance-initiated-shutdown-behavior") + 1], "terminate")
        self.assertEqual(args[args.index("--metadata-options") + 1], "HttpTokens=required")
        tags = json.loads(args[args.index("--tag-specifications") + 1])[0]["Tags"]
        self.assertIn({"Key": "Project", "Value": "visiongrep-benchmark"}, tags)
        self.assertIn({"Key": "RunId", "Value": "run-1"}, tags)
        volume = json.loads(args[args.index("--block-device-mappings") + 1])[0]["Ebs"]
        self.assertEqual(volume, {"VolumeSize": 40, "VolumeType": "gp3", "Encrypted": True, "DeleteOnTermination": True})

    def test_script_command_reuses_bootstrap_and_quotes_arguments(self):
        command_line = ["env", "BUCKET=bench-bucket", "bash", "--", "scripts/a file.sh"]
        script = aws.user_data(self.settings, "run-1", self.ref, command_line, 60)
        self.assertIn(f"git checkout --detach {self.ref}", script)
        self.assertIn("python3 benchmarks/bench.py prepare\nenv BUCKET=bench-bucket bash -- 'scripts/a file.sh'", script)
        self.assertIn("trap finish EXIT", script)
        self.assertIn("s3://bench-bucket/runs/run-1/bootstrap.log", script)

    def test_script_dry_run_resolves_ref_and_passes_bucket(self):
        argv = ['bench.py', 'aws-script', 'feature/int8', 'benchmarks/model_contract/aws_coco_int8.sh',
                '--settings', 'settings.json', '--dry-run']
        with patch.object(sys, "argv", argv), \
             patch.object(aws, "settings_from", return_value=self.settings), \
             patch.object(aws, "resolve_remote", return_value=self.ref) as resolve, \
             patch.object(aws, "command") as command, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(bench.main(), 0)
        resolve.assert_called_once_with("feature/int8")
        command.assert_not_called()
        self.assertIn(f"git checkout --detach {self.ref}", output.getvalue())
        self.assertIn("env BUCKET=bench-bucket bash -- benchmarks/model_contract/aws_coco_int8.sh", output.getvalue())
        self.assertIn("scheduler create-schedule", output.getvalue())

    def test_compare_dry_run_resolves_both_refs_and_passes_flags(self):
        argv = ['bench.py', 'aws-run', 'v0.3.0', 'feature/int8', '--samples', '5',
                '--scenarios', 'all', '--allow-output-changes', '--settings', 'settings.json', '--dry-run']
        with patch.object(sys, "argv", argv), \
             patch.object(aws, "settings_from", return_value=self.settings), \
             patch.object(aws, "resolve_remote", side_effect=[self.ref, self.cand]) as resolve, \
             patch.object(aws, "command") as command, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(bench.main(), 0)
        self.assertEqual([call.args for call in resolve.call_args_list], [('v0.3.0',), ('feature/int8',)])
        command.assert_not_called()
        self.assertIn(f"git checkout --detach {self.cand}", output.getvalue())
        self.assertIn(f"compare {self.ref} {self.cand} --samples 5 --scenarios all --allow-output-changes", output.getvalue())
        self.assertIn("scheduler create-schedule", output.getvalue())

    def test_schedule_command(self):
        launch = dt.datetime(2026, 1, 2, 3, 4, tzinfo=dt.timezone.utc)
        args, deadline = aws.schedule_command(self.settings, "run-1", "i-1", launch, 60)
        self.assertEqual(deadline, launch + dt.timedelta(minutes=70))
        self.assertEqual(args[args.index("--schedule-expression") + 1], "at(2026-01-02T04:14:00)")
        self.assertEqual(args[args.index("--action-after-completion") + 1], "DELETE")
        target = json.loads(args[args.index("--target") + 1])
        self.assertEqual(target["Arn"], "arn:aws:scheduler:::aws-sdk:ec2:terminateInstances")
        self.assertEqual(json.loads(target["Input"]), {"InstanceIds": ["i-1"]})
        self.assertEqual(target["RetryPolicy"], {"MaximumRetryAttempts": 2, "MaximumEventAgeInSeconds": 300})

    def test_option_bounds_and_burstable_instances(self):
        for minutes in (15, 240):
            aws.validate_options("c7a.xlarge", minutes)
        for minutes in (14, 241):
            with self.assertRaises(ValueError):
                aws.validate_options("c7a.xlarge", minutes)
        for instance_type in ("t3.large", "t4g.xlarge"):
            with self.assertRaises(ValueError):
                aws.validate_options(instance_type, 60)

    def test_compare_options_rejected_before_launch(self):
        for scenarios in ("all", "index", "novel_text,cached_text"):
            aws.validate_compare_options(5, scenarios)
        for samples, scenarios in ((0, "all"), (-1, "all"), (5, ""), (5, "unknown"), (5, "index,index")):
            with self.assertRaises(ValueError):
                aws.validate_compare_options(samples, scenarios)

    def test_remote_reachability(self):
        with patch.object(aws, "command", side_effect=[self.ref, "origin/main", ""]):
            self.assertEqual(aws.resolve_remote("HEAD"), self.ref)
        with patch.object(aws, "command", side_effect=[self.ref, "", "v0.2.0"]):
            self.assertEqual(aws.resolve_remote("HEAD"), self.ref)
        with patch.object(aws, "command", side_effect=[self.ref, "", ""]):
            with self.assertRaisesRegex(ValueError, "not reachable"):
                aws.resolve_remote("HEAD")


if __name__ == "__main__":
    unittest.main()
