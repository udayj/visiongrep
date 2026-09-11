# VisionGrep benchmarks

Requires Python 3.12+. The reference is the release tag `v0.2.0`.
The tag is resolved once into a commit SHA in each run configuration and measurement
contract; comparison rejects a baseline with a different reference. Do not move release
tags. The historical `benchmark-foundation-v1` and its recordings remain unchanged.
Candidates must be committed;
working-tree changes are excluded. Model or preprocessing changes need separate evaluation.

## Prepare and validate

Runtime files live outside Git in `~/.cache/visiongrep-bench/`. Preparation downloads
verified models and images. Keep the corpus checksum lock in `cache/corpora/` with recordings.

```sh
python3 benchmarks/bench.py prepare --corpus 500
# Optional: --import-models /path/to/existing/visiongrep/models
python3 benchmarks/bench.py run --mode validate --validation-samples 3 --detach
python3 benchmarks/bench.py status /path/run
python3 benchmarks/bench.py logs /path/run
python3 benchmarks/bench.py report /path/run
```

Validation compares the reference against itself and checks quality, behavior and timing.
Omit `--validation-samples` for the full budget. Validation does not record a foundation.
Use `plan` instead of `run` to inspect configuration without starting work.

## Record a foundation and compare

Plan at least three recording sessions. Run local sessions sequentially with consistent
power settings and minimal competing work. Retain all planned sessions and samples.

```sh
# Run once per session; retain each output directory.
python3 benchmarks/bench.py run --mode record --profile local-quick
python3 benchmarks/bench.py foundation /path/run1 /path/run2 /path/run3 --output /path/foundation.json
python3 benchmarks/bench.py run --candidate COMMIT --baseline /path/foundation.json --detach
```

Local and cloud foundations are separate. Choose an aggregate with verdict `calibrated`
and pass its full path through `--baseline`; selection is not automatic. Saved local
foundations can be found with `ls ~/.cache/visiongrep-bench/local-foundation-*.json`.

Each aggregate records its analysis policy, source contracts and report hashes. After
an analysis change, reaggregate compatible recordings into a new output file. Use the
local directories listed in the aggregate's `reports` field, or collected cloud
`RUN/remote` directories. Originals remain immutable. Accepted prior harnesses are
listed in [calibration.py](harness/calibration.py) and
[local_screening.py](harness/local_screening.py); unknown harnesses and incompatible
measurement contracts require new recordings.

Compatibility checks cover the reference, profile, corpus, models, machine and build
settings. Microcode is diagnostic metadata. The harness fingerprint includes benchmark
Python and JSON files, including tests; Markdown and Git commit IDs are excluded.

## Profiles and acceptance

| Profile | Images | Samples per binary/scenario | Coverage |
|---|---:|---:|---|
| local-quick | 500 | 9 | Indexing, queries, modifications, persistent sequences, quality |
| cloud-standard | 500 | 21 | All 16 timing subcases, quality |
| cloud-scale | 10,000 | 3 | Indexing, queries, modifications, deletions, renames, persistent sequences |

Both persistent sequences start one fresh `serve --stdio` process per sample, against
an already indexed corpus. The process remains alive for five sequential requests:

- `persistent_text`: first novel text request, three further novel queries, then a
  repeat of the last query as a cached control. `wall_ms` is the median of the three
  warm novel round trips.
- `persistent_updates`: first text request, three successive 1% pixel replacements
  using the same query, then an unchanged cached control. The first update loads the
  vision session; `wall_ms` is the median of the remaining two updates with that session
  retained. Every request still refreshes the directory and reads the index.

`first_response_ms` includes process startup; `first_update_ms` records the initial
vision load; `cached_response_ms` records the control. Raw per-request latencies and
results are retained. Timing covers request write/flush through receipt of a complete
response line, excluding mutations, parsing, and behavior checks. Peak RSS covers the
whole server lifetime, including first use. Each response checks result membership,
finite scores, index membership, normalized embeddings, expected pixels, and metadata;
comparisons check every response's ranking and scores. Internal phase timings are not
available for serve; compiled binary provenance comes from the untimed seed warmup.
The independent statistical sample is a process sequence, not an
individual request. First-use and control timings are descriptive; the primary latency
gate uses the warm median. One-shot and persistent scenarios share one foundation per
profile; the new measurement contract requires fresh recordings.

Comparisons precheck novel and cached text with nine samples each, plus no-cache
embedding on cloud-standard, using the full corpus. Recording has no separate precheck.

**Cloud — `cloud-session-median-v1`:** requires at least three independent instances.
For each precheck scenario, every recording session's median must be within ±10% of the
median of session medians. Comparison precheck medians must fit the same bounds.
Within-session reference CV must not exceed 10%.

**Local — `local-median-v3`:** groups each session into three consecutive triples and
uses the median of batch medians. A hierarchical bootstrap estimates a 95% interval.
Every scenario requires median uncertainty at most 10% and batch median deviation at
most 15%. Prechecks require the same precision and stability, with the median inside
the saved interval expanded by 10%. Unstable local aggregation returns `inconclusive`.

Candidates and references alternate execution order on the same machine. Paired median
ratios and bootstrap intervals determine acceptance:

- A supported gain is at least 5%, with an interval excluding zero.
- Every timing and resource interval must exclude regressions over 5%.
- Quality and behavior checks must pass; missing evidence cannot pass.
- Only cloud-standard can return `qualifies`. A local pass is `promising`.
  Definite failures return `does_not_qualify`; uncertainty returns `inconclusive`.

Calibration bounds are drift limits, not confidence intervals. Small session counts
limit conclusions about host variability and tail latency. Reports retain raw samples,
phase timings, peak child memory, index size, quality checks and HTML summaries.
Worker-summed phase timings are not additive wall time; p95 is a rough estimate.

## Cloud operation

See [infrastructure setup](infra/README.md). Supply an out-of-tree settings file.
Use a fresh instance for each recording session; slots 1–3 support concurrent runs.

```sh
python3 benchmarks/bench.py plan --profile cloud-standard --mode record --cloud /path/cloud.json
python3 benchmarks/bench.py run --profile cloud-standard --mode record --cloud /path/cloud.json --cloud-slot 1
python3 benchmarks/bench.py status /path/run
python3 benchmarks/bench.py collect /path/run
python3 benchmarks/bench.py foundation /path/run1/remote /path/run2/remote /path/run3/remote --output /path/cloud-foundation.json
python3 benchmarks/bench.py run --profile cloud-standard --candidate COMMIT --baseline /path/cloud-foundation.json --cloud /path/cloud.json
```

Cloud launches return without waiting for measurement. Let launch finish and print the
run directory. Add `--wait` to wait for termination and collect automatically;
interrupting that waiter leaves the worker running. Use `cancel RUN` to stop a job.
Artifacts persist in private S3 and collect under `RUN/remote/`. Collect terminated
jobs to release their slots. For expired interrupted launches, use
`cloud-reconcile --cloud SETTINGS --cloud-slot N`.

The default USD 1 budget and four-hour maximum select the shorter deadline at the
configured planning rate of USD 0.26/hour. Budgets apply per job and are not exact
billing caps; transfer, storage, requests, tax and cleanup grace are additional.
Use `prepare --corpus 10000` before cloud-scale. Avoid cloud uploads during local runs.

## Measurement and quality

Each invocation is a fresh process with restored inputs and warm filesystem-cache
preparation. Setup, uploads and behavior checks are outside timing. External image
queries infer embeddings; indexed image queries reuse stored vectors. Cloud storage
must settle before timing. Incomplete or cancelled runs cannot pass.

COCO manifests fix image selections and licenses. The 500-image suite includes 200
caption queries and 240 category prompts across 80 intents. No-match evaluation filters
all 500 scores at threshold 0.25. Annotation-defined absence does not establish semantic
absence or model generalization; the 10,000-image suite does not reuse these labels.
Comparisons require normalized embeddings, score/vector agreement within `1e-4`, exact
ranking agreement and behavior parity.

## Diagnostics and development

```sh
python3 benchmarks/audit_local.py ~/.cache/visiongrep-bench/runs > /tmp/local-audit.json
python3 -m unittest discover -s benchmarks/tests -v
python3 -m compileall -q benchmarks/harness benchmarks/bench.py
```

Storage diagnostics use `--mode diagnose --profile cloud-standard --max-hours 1` with
cloud settings. The 800-invocation protocol produces telemetry and syscall traces,
checks behavior and omits quality inference. Diagnostic reports cannot form foundations.
Offline tests do not replace live validation. See [historical tooling](HISTORICAL.md)
and [model results](RESULTS.md) for archived work.
