# EC2 prerequisites

No resources are created by installation, `plan`, or local runs. `run --cloud` is an
explicit paid operation. Start in us-east-1 with c7a.xlarge, Linux x86_64, four cores,
40 GB encrypted gp3 (3000 IOPS / 125 MiB/s), and one public IPv4 address.

Deploy `stack.json` with CloudFormation and CAPABILITY_IAM, supplying an existing VPC ID.
Choose a public subnet in that VPC with an internet gateway route. The security group
has no inbound rules. No NAT gateway, load balancer, or persistent worker is needed.
Copy outputs, subnet and AMI ID into an out-of-tree copy of `profiles/cloud.example.json`.
Use normal AWS CLI credentials; never put credentials in that file.

The instance role reads artifacts/input bundles and writes reports. It cannot launch
instances or delete the orchestration lock. The separate scheduler role can terminate
tagged benchmark instances. The operator needs EC2 describe/run/terminate, PassRole for
the worker and scheduler roles, Scheduler create/get/delete, and bucket read/write/delete.
Scope permissions to this deployment.

## Pinned machine image

`build-image.sh` is the EC2 user-data script for the initial image. It pins and verifies
the AWS CLI and rustup installers, installs the repository's Rust toolchain, checks tools
as `bench`, and records installed package versions. Run it only on a disposable Ubuntu
24.04 x86_64 builder. Supply `BENCH_AMI_REPORT_URI` under the reports bucket's `runs/`
prefix. It uploads `ready.json` only after successful preparation, then shuts down.

The builder must use instance-initiated shutdown behavior `stop` and a separate AWS
Scheduler termination deadline of 60 minutes. The script limits installation to 35
minutes and schedules local shutdown at 45 minutes. Require both `ready.json` and a
stopped instance before creating the image. Once the image is available, terminate the
builder and remove its schedule. Keep the AMI and snapshot IDs in the setup record;
snapshot storage continues to incur charges until explicitly deleted.

Bake an Ubuntu 24.04 x86_64 AMI once with Python 3.12.x, AWS CLI v2 (including conditional
S3 writes), Rust 1.98.1, a native compiler/linker, pkg-config, OpenSSL development files,
CA certificates, curl, systemd, runuser, and an unprivileged user named `bench`.
Use verified installation procedures and record exact versions in the AMI description.
Disable background package updates for this dedicated benchmark image. Make the toolchain
available to the `bench` user, including `rustup` and the pinned toolchain (not only to
root). The worker builds and measures as `bench`, using the same sanitized build routine
as local runs. Do not use a minimal image without these prerequisites.

The runner never selects `latest` or installs OS packages at boot. AMI changes require
calibration. Builds use each commit's locked dependencies before measurement; first
builds may need network access to registries and ONNX Runtime build assets.

Launch uses IMDSv2, an idempotency token, and project/run tags. The detached worker builds
both revisions, runs measurement as `bench`, uploads reports, and shuts down. Shutdown
means termination and deletion of the temporary root volume.

Two deadlines protect against orphan spend: a local systemd shutdown timer and an
EventBridge Scheduler termination action in the AWS control plane. Scheduling failure
terminates the instance and preserves the lock for reconciliation. Uncertain launches
leave their lease in place. `cloud-reconcile --cloud SETTINGS` can terminate expired
tagged jobs and clear their lease once termination is confirmed. There are no automatic
retries onto successively favorable hosts.

`status RUN` reads EC2 state and the latest status upload. During an expensive invocation,
an old timestamp is not proof of a hung worker. `cancel RUN` requests termination; retry
`collect RUN` after EC2 reaches terminated. Collection downloads evidence and releases
its slot's lock. Collect a completed run before reusing that slot.
Use `--cloud-slot 1`, `2`, or `3` for up to three concurrent runs on distinct
instances and volumes. Slot 1 preserves the legacy lock key. Supply the same slot
to `cloud-reconcile` for interrupted launches. Budgets and expiry are per run;
the combined budget is the sum of the active runs' budgets.

Ordinary S3 runs expire after 90 days. Copy important evidence to `foundations/` or `kept/`
before expiry. Those prefixes and shared `artifacts/` do not expire automatically. Keep
the prepared corpus lock with foundation reports: reacquiring changed source bytes creates
a different measurement contract. Public bucket access is blocked.

The USD 0.26/hour rate is a planning input, not live billing or an AWS quote. The deadline
covers preparation/build/measurement from approximately launch time. Expiry has a short
grace period. Transfers, requests, persistent storage and tax are additional. Check exact
regional prices before paid validation; the pilot establishes actual scenario durations.

Sources: [EC2 billing](https://aws.amazon.com/ec2/pricing/on-demand/),
[S3 conditional writes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html),
[Scheduler API](https://docs.aws.amazon.com/scheduler/latest/APIReference/API_CreateSchedule.html).
