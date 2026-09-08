"""Ephemeral EC2 jobs using AWS CLI, S3 reports, and an independent expiry schedule."""

from __future__ import annotations

import datetime as dt
import json
import re
import shlex
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

from .storage import (
    BENCHMARKS,
    FOUNDATION,
    ROOT,
    command,
    digest,
    read_json,
    write_json,
)


def aws(region: str, *args, timeout=120):
    output = command(
        ["aws", "--region", region, "--no-cli-pager", *args, "--output", "json"],
        timeout=timeout,
    )
    return json.loads(output) if output.strip() else {}


def validate_settings(settings):
    for key in (
        "region",
        "ami",
        "subnet",
        "security_group",
        "instance_profile",
        "bucket",
        "scheduler_role",
    ):
        if not settings.get(key) or "REPLACE" in settings[key]:
            raise ValueError(f"cloud configuration requires {key}")
    if settings["region"] not in ("us-east-1", "us-east-2", "us-west-2"):
        raise ValueError("choose one of the supported US regions")
    if not re.fullmatch(r"ami-[0-9a-f]+", settings["ami"]):
        raise ValueError("pin an AMI ID; aliases/latest images are not accepted")


def lock_key(slot: int = 1) -> str:
    if slot not in (1, 2, 3):
        raise ValueError("cloud slot must be 1, 2, or 3")
    # Slot one retains compatibility with existing handles and launch leases.
    return "control/active.json" if slot == 1 else f"control/active-{slot}.json"


def bootstrap(bucket: str, prefix: str, region: str, ami: str, seconds: int) -> str:
    # Python/Rust/AWS CLI/native build dependencies must already exist in the pinned AMI.
    remote = f"s3://{bucket}/{prefix}"
    script = f"""#!/bin/bash
set -euo pipefail
export AWS_DEFAULT_REGION={shlex.quote(region)}
export BENCH_AMI={shlex.quote(ami)}
export BENCH_S3_RUN={shlex.quote(remote)}
export BENCH_LAUNCH_EPOCH=${{BENCH_LAUNCH_EPOCH:-$(date +%s)}}
mkdir -p /opt/visiongrep-bench
cd /opt/visiongrep-bench
# Install the local deadline before any network or build work.
systemd-run --unit=visiongrep-expiry --on-active={seconds}s /sbin/shutdown -h now
finish() {{
  code=$?
  trap - EXIT
  aws s3 sync /opt/visiongrep-bench/runs/current "$BENCH_S3_RUN" --exclude '*/images/*' --exclude '*/cache/*' --exclude '*.onnx' --only-show-errors || true
  aws s3 cp /var/log/cloud-init-output.log "$BENCH_S3_RUN/bootstrap.log" --only-show-errors || true
  shutdown -h now
  exit "$code"
}}
trap finish EXIT
aws s3 cp "$BENCH_S3_RUN/input.tar.gz" input.tar.gz --only-show-errors
aws s3 cp "$BENCH_S3_RUN/input.sha256" input.sha256 --only-show-errors
sha256sum -c input.sha256
python3 -c "import tarfile; tarfile.open('input.tar.gz').extractall('.', filter='data')"
export BENCH_INSTANCE_ID=$(cat instance-id)
aws s3 sync {shlex.quote("s3://" + bucket + "/artifacts/objects/")} cache/objects/ --only-show-errors
python3 - <<'PYBOOT'
import json, pathlib
root = pathlib.Path('/opt/visiongrep-bench')
config = json.loads((root/'config.json').read_text())
config['directory'] = str(root/'runs/current')
config['cache'] = str(root/'cache')
config['corpus'] = str(root/'corpus.json')
if config.get('baseline'):
    config['baseline'] = str(root/'baseline.json')
config['sources'] = {{role: str(root/role) for role in config['source_commits']}}
(root/'runs/current').mkdir(parents=True, exist_ok=True)
(root/'config.json').write_text(json.dumps(config))
PYBOOT
chown -R bench:bench /opt/visiongrep-bench
runuser -u bench -- python3 benchmarks/bench.py worker --config config.json
"""
    return script


def launch(config: dict, settings_path: Path) -> dict:
    settings = read_json(settings_path)
    validate_settings(settings)
    region, bucket = settings["region"], settings["bucket"]
    image = aws(region, "ec2", "describe-images", "--image-ids", settings["ami"])[
        "Images"
    ][0]
    if image["Architecture"] != "x86_64" or image["RootDeviceType"] != "ebs":
        raise ValueError("the pinned AMI must use x86_64 and an EBS root volume")
    directory = Path(config["directory"])
    run_id = directory.name
    prefix = "runs/" + run_id
    if config["profile"]["cloud"] is not True:
        raise ValueError("cloud launch requires a cloud profile")
    if config["hourly_budget_usd"] <= 0:
        raise ValueError("cloud run must include a positive hourly planning rate")
    # Each slot owns an atomic lease and a separate instance. Never steal a busy slot.
    slot = config.get("cloud_slot", 1)
    key = lock_key(slot)
    lock_path = directory / "cloud-lock.json"
    write_json(
        lock_path,
        {
            "run_id": run_id,
            "expires": dt.datetime.now(dt.timezone.utc).timestamp()
            + config["max_seconds"],
        },
    )
    lease = aws(
        region,
        "s3api",
        "put-object",
        "--bucket",
        bucket,
        "--key",
        key,
        "--body",
        str(lock_path),
        "--if-none-match",
        "*",
    )
    instance_id = None
    schedule_name = "visiongrep-" + run_id
    try:
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            staging = Path(temporary)
            shutil.copytree(
                BENCHMARKS,
                staging / "benchmarks",
                ignore=shutil.ignore_patterns("__pycache__", "work"),
            )
            shutil.copyfile(config["corpus"], staging / "corpus.json")
            if config.get("baseline"):
                shutil.copyfile(config["baseline"], staging / "baseline.json")
            commits = {"foundation": FOUNDATION}
            if config["mode"] not in ("record", "diagnose"):
                commits["candidate"] = config["candidate"]
            for role, sha in commits.items():
                archive = staging / (role + ".tar")
                command(
                    ["git", "archive", "--format=tar", "--output", str(archive), sha],
                    cwd=ROOT,
                )
                with tarfile.open(archive) as source:
                    source.extractall(staging / role, filter="data")
                archive.unlink()
            config = config | {"source_commits": commits}
            write_json(staging / "config.json", config)
            # Instance identity is read via IMDSv2 after boot, avoiding a second bundle upload.
            (staging / "instance-id").write_text("pending")
            bundle = directory / "input.tar.gz"
            with tarfile.open(bundle, "w:gz") as output:
                for path in staging.iterdir():
                    output.add(path, arcname=path.name)
            checksum = directory / "input.sha256"
            checksum.write_text(digest(bundle) + "  input.tar.gz\n")
            command(
                [
                    "aws",
                    "--region",
                    region,
                    "s3",
                    "sync",
                    str(Path(config["cache"]) / "objects"),
                    f"s3://{bucket}/artifacts/objects/",
                    "--only-show-errors",
                ],
                timeout=3600,
            )
            for path in (bundle, checksum):
                command(
                    [
                        "aws",
                        "--region",
                        region,
                        "s3",
                        "cp",
                        str(path),
                        f"s3://{bucket}/{prefix}/{path.name}",
                        "--only-show-errors",
                    ],
                    timeout=600,
                )
        boot = bootstrap(bucket, prefix, region, settings["ami"], config["max_seconds"])
        if config["mode"] == "diagnose":
            # Diagnostic-only tools are installed before building or measuring, with a bound.
            boot = boot.replace(
                "chown -R bench:bench /opt/visiongrep-bench",
                "timeout --kill-after=10s 180s bash -c 'apt-get update && "
                "DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends strace'\n"
                "sysctl -w kernel.sched_schedstats=1\n"
                "chown -R bench:bench /opt/visiongrep-bench",
            )
        boot = boot.replace(
            "export BENCH_INSTANCE_ID=$(cat instance-id)",
            """TOKEN=$(curl --fail -sS -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' http://169.254.169.254/latest/api/token)
export BENCH_INSTANCE_ID=$(curl --fail -sS -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)""",
        )
        boot = boot.replace(
            "set -euo pipefail",
            "set -euo pipefail\nexport BENCH_LAUNCH_EPOCH=" + str(time.time()),
        )
        userdata = directory / "user-data.sh"
        userdata.write_text(boot)
        # Refresh the lease after input upload; retain the original conditional lock ownership.
        write_json(
            lock_path,
            {"run_id": run_id, "expires": time.time() + config["max_seconds"] + 600},
        )
        aws(
            region,
            "s3api",
            "put-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--body",
            str(lock_path),
            "--if-match",
            lease["ETag"],
        )
        response = aws(
            region,
            "ec2",
            "run-instances",
            "--image-id",
            settings["ami"],
            "--instance-type",
            "c7a.xlarge",
            "--count",
            "1",
            "--client-token",
            run_id,
            "--iam-instance-profile",
            json.dumps({"Name": settings["instance_profile"]}),
            "--network-interfaces",
            json.dumps(
                [
                    {
                        "DeviceIndex": 0,
                        "SubnetId": settings["subnet"],
                        "Groups": [settings["security_group"]],
                        "AssociatePublicIpAddress": True,
                    }
                ]
            ),
            "--block-device-mappings",
            json.dumps(
                [
                    {
                        "DeviceName": image["RootDeviceName"],
                        "Ebs": {
                            "VolumeSize": 40,
                            "VolumeType": "gp3",
                            "Iops": 3000,
                            "Throughput": 125,
                            "Encrypted": True,
                            "DeleteOnTermination": True,
                        },
                    }
                ]
            ),
            "--metadata-options",
            "HttpTokens=required,HttpEndpoint=enabled",
            "--instance-initiated-shutdown-behavior",
            "terminate",
            "--user-data",
            "file://" + str(userdata),
            "--tag-specifications",
            json.dumps(
                [
                    {
                        "ResourceType": "instance",
                        "Tags": [
                            {"Key": "Project", "Value": "visiongrep-benchmark"},
                            {"Key": "RunId", "Value": run_id},
                        ],
                    }
                ]
            ),
        )
        instance_id = response["Instances"][0]["InstanceId"]
        expiry = dt.datetime.now(dt.timezone.utc) + dt.timedelta(
            seconds=config["max_seconds"] + 120
        )
        aws(
            region,
            "scheduler",
            "create-schedule",
            "--name",
            schedule_name,
            "--schedule-expression",
            f"at({expiry.strftime('%Y-%m-%dT%H:%M:%S')})",
            "--schedule-expression-timezone",
            "UTC",
            "--flexible-time-window",
            '{"Mode":"OFF"}',
            "--action-after-completion",
            "DELETE",
            "--target",
            json.dumps(
                {
                    "Arn": "arn:aws:scheduler:::aws-sdk:ec2:terminateInstances",
                    "RoleArn": settings["scheduler_role"],
                    "Input": json.dumps({"InstanceIds": [instance_id]}),
                    "RetryPolicy": {
                        "MaximumRetryAttempts": 2,
                        "MaximumEventAgeInSeconds": 300,
                    },
                }
            ),
        )
        handle = {
            "cloud_slot": slot,
            "region": region,
            "bucket": bucket,
            "prefix": prefix,
            "instance_id": instance_id,
            "schedule": schedule_name,
            "run_id": run_id,
            "launched": dt.datetime.now(dt.timezone.utc).timestamp(),
            "hourly_budget_usd": config["hourly_budget_usd"],
        }
        write_json(directory / "cloud.json", handle)
        return handle
    except BaseException:
        if instance_id:
            aws(region, "ec2", "terminate-instances", "--instance-ids", instance_id)
        # Preserve lock on uncertain launch responses; reconcile by the idempotent RunId.
        write_json(
            directory / "launch-failed.json",
            {
                "run_id": run_id,
                "cloud_slot": slot,
                "region": region,
                "bucket": bucket,
                "instance_id": instance_id,
                "action": "cloud-reconcile after expiry",
            },
        )
        raise


def status(directory: Path) -> dict:
    handle = read_json(directory / "cloud.json")
    result = aws(
        handle["region"],
        "ec2",
        "describe-instances",
        "--instance-ids",
        handle["instance_id"],
    )
    state = result["Reservations"][0]["Instances"][0]["State"]["Name"]
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "status.json"
        try:
            aws(
                handle["region"],
                "s3api",
                "get-object",
                "--bucket",
                handle["bucket"],
                "--key",
                handle["prefix"] + "/status.json",
                str(path),
            )
            progress = read_json(path)
        except RuntimeError:
            progress = {
                "stage": "status unavailable; inspect bootstrap log or AWS permissions"
            }
    return {
        "instance_state": state,
        "progress": progress,
        "instance_id": handle["instance_id"],
    }


def collect(directory: Path, cancel=False):
    handle = read_json(directory / "cloud.json")
    region = handle["region"]
    if cancel:
        aws(
            region,
            "ec2",
            "terminate-instances",
            "--instance-ids",
            handle["instance_id"],
        )
    state = status(directory)["instance_state"]
    if state != "terminated":
        raise ValueError(
            "instance is not yet terminated; retry collect once termination finishes"
        )
    command(
        [
            "aws",
            "--region",
            region,
            "s3",
            "sync",
            f"s3://{handle['bucket']}/{handle['prefix']}/",
            str(directory / "remote"),
            "--exclude",
            "input.*",
            "--only-show-errors",
        ],
        timeout=600,
    )
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "lock.json"
        lease = aws(
            region,
            "s3api",
            "get-object",
            "--bucket",
            handle["bucket"],
            "--key",
            lock_key(handle.get("cloud_slot", 1)),
            str(path),
        )
        if read_json(path)["run_id"] != handle["run_id"]:
            raise ValueError("cloud lock belongs to another run; refusing to clear it")
        aws(
            region,
            "s3api",
            "delete-object",
            "--bucket",
            handle["bucket"],
            "--key",
            lock_key(handle.get("cloud_slot", 1)),
            "--if-match",
            lease["ETag"],
        )


def reconcile(settings_path: Path, *, slot: int = 1):
    """Recover a launch interrupted before its local instance handle was saved."""
    settings = read_json(settings_path)
    validate_settings(settings)
    region, bucket = settings["region"], settings["bucket"]
    key = lock_key(slot)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "lock.json"
        lease = aws(
            region,
            "s3api",
            "get-object",
            "--bucket",
            bucket,
            "--key",
            key,
            str(path),
        )
        lock = read_json(path)
    result = aws(
        region,
        "ec2",
        "describe-instances",
        "--filters",
        json.dumps(
            [
                {"Name": "tag:RunId", "Values": [lock["run_id"]]},
                {"Name": "tag:Project", "Values": ["visiongrep-benchmark"]},
            ]
        ),
    )
    active = [
        instance["InstanceId"]
        for reservation in result["Reservations"]
        for instance in reservation["Instances"]
        if instance["State"]["Name"] != "terminated"
    ]
    if active:
        if time.time() < lock["expires"]:
            raise ValueError("run has not expired; use cancel with its handle")
        aws(region, "ec2", "terminate-instances", "--instance-ids", *active)
        raise ValueError(
            "termination requested; retry reconciliation after termination finishes"
        )
    if time.time() < lock["expires"]:
        raise ValueError("launch lease has not expired; refusing to race a launcher")
    aws(
        region,
        "s3api",
        "delete-object",
        "--bucket",
        bucket,
        "--key",
        key,
        "--if-match",
        lease["ETag"],
    )
