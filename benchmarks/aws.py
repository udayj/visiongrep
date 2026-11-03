"""Launch unattended benchmark instances and retrieve their results using the AWS CLI."""

import datetime as dt
import json
import shlex
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path.home() / ".cache" / "visiongrep-bench"
REPOSITORY = "https://github.com/udayj/visiongrep.git"


def settings_from(path):
    settings = json.loads(Path(path).read_text())
    for key in ("region", "subnet", "security_group", "instance_profile", "bucket", "ami", "scheduler_role"):
        if not settings.get(key) or "REPLACE" in settings[key]:
            raise ValueError(f"AWS settings require {key}")
    return settings


def command(args, *, capture=True):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=capture)
    if result.returncode:
        detail = result.stderr.strip() if capture else "see output above"
        raise RuntimeError(f"{args[0]} failed: {detail}")
    return result.stdout if capture else ""


def aws(settings, *args):
    output = command(["aws", "--region", settings["region"], "--no-cli-pager", *args, "--output", "json"])
    return json.loads(output) if output.strip() else {}


def resolve_remote(ref):
    sha = command(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"]).strip()
    branches = command(["git", "branch", "-r", "--contains", sha]).strip()
    tags = command(["git", "tag", "--contains", sha]).strip()
    if not branches and not tags:
        raise ValueError(f"{ref} is not reachable from a remote-tracking branch or tag")
    return sha


def validate_options(instance_type, max_minutes):
    if instance_type.lower().startswith("t"):
        raise ValueError("burstable t* instances are not suitable for benchmarks")
    if not 15 <= max_minutes <= 240:
        raise ValueError("--max-minutes must be between 15 and 240")


def validate_compare_options(samples, scenarios):
    allowed = {"index", "novel_text", "cached_text", "external_image", "changed_1pct", "server_novel_text"}
    if samples < 1:
        raise ValueError("samples must be positive")
    if scenarios != "all":
        names = scenarios.split(",")
        if any(name not in allowed for name in names) or len(set(names)) != len(names):
            raise ValueError("invalid or duplicate scenario")


def user_data(settings, run_id, ref, cand, flags, max_minutes):
    prefix = f"s3://{settings['bucket']}/runs/{run_id}/"
    compare = shlex.join(["python3", "benchmarks/bench.py", "compare", ref, cand, *flags])
    return f'''#!/bin/bash
shutdown -h +{max_minutes}
export HOME=/root
set -euo pipefail
export AWS_DEFAULT_REGION={shlex.quote(settings['region'])}
finish() {{
  status=$?
  trap - EXIT
  aws s3 sync "$HOME/.cache/visiongrep-bench/runs/" {shlex.quote(prefix)} --exclude '*' --include '*.json' --only-show-errors || true
  aws s3 cp /var/log/cloud-init-output.log {shlex.quote(prefix + 'bootstrap.log')} --only-show-errors || true
  shutdown -h now
  exit "$status"
}}
trap finish EXIT
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential pkg-config libssl-dev clang cmake git curl unzip python3
if ! command -v aws >/dev/null; then
  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-$(uname -m).zip" -o /tmp/awscli.zip
  unzip -q /tmp/awscli.zip -d /tmp
  /tmp/aws/install
fi
if ! command -v cargo >/dev/null; then
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
export PATH="$HOME/.cargo/bin:$PATH"
git clone {shlex.quote(REPOSITORY)} /opt/visiongrep
cd /opt/visiongrep
git checkout --detach {shlex.quote(cand)}
python3 benchmarks/bench.py prepare
{compare}
'''


def instance_command(settings, run_id, instance_type, script):
    profile = settings["instance_profile"]
    profile_spec = {"Arn" if profile.startswith("arn:") else "Name": profile}
    volumes = [{
        "DeviceName": "/dev/sda1",
        "Ebs": {
            "VolumeSize": 40,
            "VolumeType": "gp3",
            "Encrypted": True,
            "DeleteOnTermination": True,
        },
    }]
    tags = [{
        "ResourceType": "instance",
        "Tags": [
            {"Key": "Project", "Value": "visiongrep-benchmark"},
            {"Key": "RunId", "Value": run_id},
        ],
    }]
    return [
        "aws", "--region", settings["region"], "--no-cli-pager",
        "ec2", "run-instances",
        "--image-id", settings["ami"],
        "--instance-type", instance_type,
        "--subnet-id", settings["subnet"],
        "--security-group-ids", settings["security_group"],
        "--iam-instance-profile", json.dumps(profile_spec),
        "--instance-initiated-shutdown-behavior", "terminate",
        "--metadata-options", "HttpTokens=required",
        "--block-device-mappings", json.dumps(volumes),
        "--tag-specifications", json.dumps(tags),
        "--user-data", script,
        "--output", "json",
    ]


def schedule_command(settings, run_id, instance_id, launch_time, max_minutes):
    deadline = launch_time + dt.timedelta(minutes=max_minutes + 10)
    target = {
        "Arn": "arn:aws:scheduler:::aws-sdk:ec2:terminateInstances",
        "RoleArn": settings["scheduler_role"],
        "Input": json.dumps({"InstanceIds": [instance_id]}),
        "RetryPolicy": {"MaximumRetryAttempts": 2, "MaximumEventAgeInSeconds": 300},
    }
    command = [
        "aws", "--region", settings["region"], "--no-cli-pager",
        "scheduler", "create-schedule",
        "--name", f"visiongrep-bench-{run_id}",
        "--schedule-expression", f"at({deadline.strftime('%Y-%m-%dT%H:%M:%S')})",
        "--schedule-expression-timezone", "UTC",
        "--flexible-time-window", "Mode=OFF",
        "--action-after-completion", "DELETE",
        "--target", json.dumps(target),
        "--output", "json",
    ]
    return command, deadline


def run(args):
    validate_options(args.instance_type, args.max_minutes)
    validate_compare_options(args.samples, args.scenarios)
    settings = settings_from(args.settings)
    ref, cand = resolve_remote(args.ref), resolve_remote(args.cand)
    flags = ["--samples", str(args.samples), "--scenarios", args.scenarios]
    if args.allow_output_changes:
        flags.append("--allow-output-changes")
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    script = user_data(settings, run_id, ref, cand, flags, args.max_minutes)
    instance_args = instance_command(settings, run_id, args.instance_type, script)
    schedule_args, deadline = schedule_command(settings, run_id, "INSTANCE_ID", now, args.max_minutes)
    if args.dry_run:
        print(script)
        print(shlex.join(instance_args))
        print(shlex.join(schedule_args))
        return
    instance = json.loads(command(instance_args))["Instances"][0]
    instance_id = instance["InstanceId"]
    try:
        launched = dt.datetime.fromisoformat(instance["LaunchTime"].replace("Z", "+00:00"))
        schedule_args, deadline = schedule_command(settings, run_id, instance_id, launched, args.max_minutes)
        command(schedule_args)
    except (RuntimeError, OSError, ValueError, KeyError):
        try:
            aws(settings, "ec2", "terminate-instances", "--instance-ids", instance_id)
        except (RuntimeError, OSError) as error:
            print(f"failed to terminate {instance_id}: {error}", file=sys.stderr)
        raise
    print(f"Run: {run_id}\nInstance: {instance_id}\nAWS deadline: {deadline.isoformat()}")
    print(shlex.join(["python3", "benchmarks/bench.py", "aws-fetch", run_id, "--settings", str(args.settings)]))


def fetch(args, print_report):
    settings = settings_from(args.settings)
    destination = CACHE / "runs" / args.run_id
    destination.mkdir(parents=True, exist_ok=True)
    command(
        ["aws", "--region", settings["region"], "s3", "sync",
         f"s3://{settings['bucket']}/runs/{args.run_id}/", str(destination),
         "--only-show-errors"],
        capture=False,
    )
    reports = sorted(destination.glob("*.json"))
    if reports:
        print_report(json.loads(reports[-1].read_text()))
        return
    result = aws(
        settings, "ec2", "describe-instances", "--filters",
        f"Name=tag:RunId,Values={args.run_id}",
        "Name=tag:Project,Values=visiongrep-benchmark",
    )
    for reservation in result.get("Reservations", []):
        for instance in reservation["Instances"]:
            print(f"{instance['InstanceId']}: {instance['State']['Name']}")
    log = destination / "bootstrap.log"
    if log.exists():
        print("\n".join(log.read_text(errors="replace").splitlines()[-30:]))
    else:
        print("No log uploaded yet.")


def list_instances(args):
    settings = settings_from(args.settings)
    result = aws(
        settings, "ec2", "describe-instances", "--filters",
        "Name=tag:Project,Values=visiongrep-benchmark",
        "Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down",
    )
    for reservation in result.get("Reservations", []):
        for instance in reservation["Instances"]:
            tags = {tag["Key"]: tag["Value"] for tag in instance.get("Tags", [])}
            print(tags.get("RunId", "?"), instance["InstanceId"], instance["State"]["Name"], instance["LaunchTime"])
    schedules = aws(settings, "scheduler", "list-schedules", "--name-prefix", "visiongrep-bench-")
    for schedule in schedules.get("Schedules", []):
        print("schedule", schedule["Name"], schedule["State"])
