# SQLite latency diagnostic

## Follow-up: flush setup writes before measuring

The first diagnostic (`20260908-152528-fbac348f`) completed all 800 samples.
The traced deletion outlier spent 407 ms of its 410 ms reconciliation in four
`fsync` calls. Telemetry showed journal/writeback waits and a draining dirty-memory
backlog of about 600 MiB. No SQLite lock retries explained the traced outliers.

The follow-up keeps the same fixed batches, application commit and instance settings.
Cloud invocations now call GNU `sync -f` on the scenario filesystem after setup,
status/report uploads and invocation-file creation, outside the process timer.
They then require at least 250 ms with no observed device I/O, no writeback and
at most 16 MiB globally dirty memory. The 30-second total settling deadline includes
the flush. Failure invalidates the run; samples are never selectively removed.
`storage_quiescence` records the policy, flush/settling duration and before/after
counters in each observation. A fresh helper after settling isolates CLI resource
accounting from the sync subprocess. This policy changes the cloud measurement
contract and does not alter local measurements or SQLite durability settings.

Success criteria: all 800 samples and behavior checks complete, no database phase
above 100 ms (the preselected diagnostic threshold for the original stalls), and
the traced sync durations and telemetry no longer show the original writeback
stalls. Report every maximum and all samples even if these criteria fail. A clean
follow-up on one new instance supports the fix but cannot prove that cloud storage
will never stall. New foundations still require the normal three-instance protocol.

This branch investigates the isolated database-phase stalls in cloud recording
`20260908-124933-332db744`. It runs the unchanged application at
`8b518ed86ff4f29e9c071931c91efa3e5ec31b44`; only the harness is instrumented.
The diagnostic harness has a different fingerprint. Neither batch is foundation
evidence, and no existing foundation or recording is rewritten.

## Fixed protocol

One fresh instance uses the existing pinned AMI, c7a.xlarge, and encrypted 40 GiB
gp3 root volume with 3000 IOPS and 125 MiB/s. The existing bucket lock serializes
launches. Local shutdown and the independent AWS termination schedule remain active.
The runtime ceiling is one hour including setup/builds, using the harness's
historical $0.26/hour planning rate, not current pricing or an exact billing cap.

First run 100 samples each of deletion, modified query image, modification, and
rename, in that order. Repeat that sequence with strace: 800 timed invocations
total. Retain every sample; do not extend the run based on observing a spike.
The original scenario class owns input restoration, warmup, cache warming and
behavior checks. The original runner owns reports, synchronous uploads, deadlines
and process-group cleanup. Quality inference is omitted from this diagnostic.

Strace is installed from the distribution's package repository before building
or measuring. Scheduler statistics are enabled on this disposable instance.
Both batches have periodic system telemetry and extra measurement timestamps/
resource counters. The second batch additionally has ptrace overhead and writes
completed traces outside timing; it is not a performance comparison with the first.

## Evidence

- Each sample retains invocation, results and application phase timings.
- `diagnostic` in each report sample contains epoch/monotonic start timestamps,
  child CPU time, context switches, faults and block-I/O counters.
- `scenarios/SCENARIO/BATCH/telemetry.jsonl` records `/proc` disk counters, CPU/steal,
  dirty/writeback memory, pressure, locks, process states, I/O and per-thread
  scheduler counters. Worker-user processes are sampled about every 100 ms;
  all processes every tenth snapshot. Actual timestamps and unavailable reads
  are explicit. Collector work adds to the interval. Short-lived processes may
  escape sampling, and root-owned process I/O may be inaccessible.
- `observations/sample-NNNN/syscalls.log` in the traced batch includes epoch
  timestamps, durations, file operations, sync calls, locks and retry sleeps.
  Read/write buffers are not dumped. Open/close calls identify raw I/O file descriptors.
- Live trace and telemetry files are stored on tmpfs, then copied to the run
  directory outside timing. Per-trace file size is capped at 64 MiB, telemetry
  at 128 MiB per scenario batch. Exceeding a limit fails the diagnostic.

Interpret long sync/write calls together with disk and scheduler counters. Failed
`fcntl` locks followed by retry sleeps support contention. Runqueue-wait growth
supports scheduling delay. A long syscall alone does not establish an EBS fault.
No captured stall is inconclusive. No sample is discarded to obtain acceptance.

The syscall timing semantics are documented in the
[strace manual](https://man7.org/linux/man-pages/man1/strace.1.html).
Per-thread CPU and runqueue counters follow the
[Linux scheduler statistics documentation](https://www.kernel.org/doc/html/latest/scheduler/sched-stats.html).

## Launch and collect

Use the existing authenticated named profile; no credentials belong in the checkout.

```sh
AWS_PROFILE=visiongrep-bench python3 benchmarks/bench.py plan --profile cloud-standard --mode diagnose --cloud /Users/uday/.cache/visiongrep-bench/aws/cloud.json --max-hours 1 --budget-usd 0.26
AWS_PROFILE=visiongrep-bench python3 benchmarks/bench.py run --profile cloud-standard --mode diagnose --cloud /Users/uday/.cache/visiongrep-bench/aws/cloud.json --max-hours 1 --budget-usd 0.26 --wait
```

`--wait` monitors instance termination and collects artifacts in the same
authenticated session. Interrupting the local waiter does not cancel the cloud
worker; use the existing `cancel RUN` or resume with `status RUN` / `collect RUN`.
The output verdict is `diagnostic_complete`, or invalid/cancelled/incomplete if
execution fails. Completion means the protocol ran, not that a cause was proven.
