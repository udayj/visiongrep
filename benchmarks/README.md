# Benchmarks

The comparison tool builds two commits and measures them alternately on the same
machine. It reports timing and peak memory, checks that search results agree, and
can compare retrieval quality. Each comparison is fresh; results from separate
runs are not intended to be combined.

## Local comparisons

Python 3.12, Git, Rust and Cargo are required. Download and verify the pinned
models and images once, then compare committed revisions:

```sh
python3 benchmarks/bench.py prepare
python3 benchmarks/bench.py compare v0.2.0 HEAD --samples 15
python3 benchmarks/bench.py quality v0.2.0 HEAD
```

The default
comparison runs every scenario except the slower index build; use
`--scenarios all` to include it or pass comma-separated scenario names. Use
`--allow-output-changes` when a model or preprocessing change intentionally
changes search results. Raw samples and machine information are saved under
`~/.cache/visiongrep-bench/runs/`.

The table reports medians and the percentage change in elapsed time; positive
values mean the candidate is faster. The paired bootstrap interval is an
approximate 95% interval. Run `compare HEAD HEAD` to see the noise on your
machine, and close other heavy applications while measuring. Quality reports
retrieval metrics and, for two revisions, their differences and the fraction of
queries with identical top-ten results.

## AWS

Copy `aws-settings.example.json` outside the repository and fill in the existing
account resources. Install and configure the AWS CLI, then run:

```sh
python3 benchmarks/bench.py aws-run v0.2.0 HEAD --settings /path/to/settings.json
python3 benchmarks/bench.py aws-fetch RUN_ID --settings /path/to/settings.json
python3 benchmarks/bench.py aws-list --settings /path/to/settings.json
```

`aws-run --dry-run` prints the user data and AWS commands without launching an
instance. Both commits must be reachable from a remote-tracking branch or tag so
the worker can clone them. Each run uses its own instance and S3 prefix. The
default `c7a.xlarge` has four dedicated vCPUs and 8 GiB RAM, the minimum for
reliable measurements: builds use about 2–4 GiB, a search process about
400–650 MiB, and two servers about 1–1.3 GiB. Models and optimized graphs
benefit from page cache. Burstable instances are refused because exhausted CPU
credits skew timings. The encrypted 40 GiB gp3 volume leaves room for builds,
caches and corpus copies.

The instance shuts down after the run or the `--max-minutes` deadline (60 by
default). An independent AWS Scheduler deadline terminates it ten minutes
later even if the instance hangs; the schedule deletes itself after firing.
`aws-list` shows active instances and pending schedules, and should show
nothing after runs and deadlines finish. To terminate a stray instance:

```sh
aws ec2 terminate-instances --instance-ids INSTANCE_ID --region REGION
```

The operator needs EC2 run, describe and terminate permissions; `iam:PassRole`
for the worker profile and Scheduler role; Scheduler create and list; and S3
read access to the bucket's `runs/` prefix.
