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
Keep run-specific investigation notes and audit outputs in that cache as well. Only
curated foundation summaries belong in `benchmarks/results/`.

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

Recalculate an existing raw report with the current analysis without changing it:

```sh
python3 benchmarks/bench.py reanalyze /path/run/remote --output /path/run/median-analysis
# For a comparison, also supply a foundation recalculated with the current method:
python3 benchmarks/bench.py reanalyze /path/comparison/remote --output /path/comparison/median-analysis --baseline /path/foundation.json
```

`foundation` also recalculates each source recording before aggregation. Old timing
verdicts, including CV-only failures, do not decide eligibility. Complete samples,
correctness, quality, precision, and stability do. Outputs are immutable and record
source paths, report hashes, and original contracts. Every raw sample is retained.
There is one current analysis method; no legacy-policy selection or hash allowlists.

Experimental conditions must match: reference commit, profile/sample budget, corpus,
models, quality judgments, machine, and build settings. Original harness hashes remain
provenance; analysis labels and full-harness hashes are not experimental conditions.
Reanalysis is an explicit operation on raw evidence, not proof that arbitrary versions
of measurement code are equivalent. Audit the source measurement protocol before combining
recordings; actual measurement changes require fresh evidence. This change preserves
the one-shot invocations and persistent sequence definitions. Microcode remains diagnostic.

## Profiles and acceptance

| Profile | Images | Samples per binary/scenario | Coverage |
|---|---:|---:|---|
| local-quick | 500 | 9 | Indexing, queries, modifications, persistent sequences, quality |
| cloud-standard | 500 | 21 | All 16 timing subcases, quality |
| cloud-scale | 10,000 | 9 | Indexing, queries, modifications, deletions, renames, persistent sequences |

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

**One typical-latency method for local and cloud runs:**

- Estimate a session's typical latency with the median of its raw samples. For a
  foundation, use the median of session medians, giving each instance equal weight.
- Estimate precision with 5,000 deterministic block-bootstrap draws. Resample
  consecutive triples of samples within each session and, for multiple sessions,
  resample sessions as well. Blocks retain short-range dependence. The approximate
  95% percentile interval must stay within 10% of the point estimate on both sides.
- Check sustained drift separately: split each session into three equal chronological
  windows (7 samples each for cloud-standard; 3 for local-quick). Every window median
  must stay within 15% of its own session median. Each session median must also stay
  within 15% of the foundation median. Do not drop slow windows or sessions.
- At least 9 samples in complete triples are needed. Existing 3-sample scale recordings
  remain diagnostics and cannot certify precision or drift. Future cloud-scale runs
  use 9 samples. Foundation aggregation requires at least 3 sessions, from distinct
  instances for cloud profiles.
- Comparison prechecks test median precision/stability and require the reference
  median inside the saved foundation median ±10%. Uncertain timing produces
  `inconclusive`, not a correctness failure. It cannot erase a definite regression,
  missing correctness evidence, or failed quality checks.

The 10% precision and 15% drift limits carry over from local screening; they are
engineering tolerances, not statistical confidence levels. They were not selected to
fit a cloud recording. Three drift windows preserve the local policy's chronological
structure as sample budgets grow; transient clusters remain in the uncertainty estimate
and diagnostics rather than being treated as a sustained shift by themselves.

For comparisons, estimate improvement as `1 - median(candidate) / median(reference)`.
Resample paired consecutive triples, keeping candidate/reference pairs aligned. This
measures a change in typical latency, rather than the median of per-pair ratios.
Use the same procedure for peak memory and index size. Existing suite-adjusted interval
levels and acceptance gates remain: a supported gain is at least 5% with its interval
excluding zero; every latency/resource interval must exclude regressions over 5%.
Quality and behavior must pass. Cloud qualification needs at least 20 pairs and the
complete cloud-standard suite. Local success is `promising`; cloud-scale is supplemental.

CV, MAD, p95, and raw values remain diagnostics. p95 is a rough empirical estimate at
21 samples and is omitted below 20; it is not a tail-latency guarantee. One persistent
sequence is one statistical sample: requests within a process are not extra independent
observations. First-use and cached-control latencies remain separate diagnostics.

These bootstrap intervals are approximate. Short blocks assume dependence beyond three
successive samples is limited; 9–21 samples and three instances cannot establish exact
95% coverage or characterize rare tails. Drift gates catch sustained regime changes,
not every possible dependence pattern. For tighter intervals, pre-plan more samples or
independent sessions and retain the entire fixed budget; do not repeatedly sample until
an interval passes or pool incompatible conditions.

Background: [block resampling preserves dependence](https://stat.cmu.edu/~cshalizi/uADA/16/lectures/26.pdf);
[CV describes spread relative to the mean](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/coefvari.htm),
not precision of a median estimate.

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
jobs to release their slots. Collection can be retried after an interrupted download,
even after EC2 removes the terminated instance record. It resumes the S3 sync and
leaves any slot lease belonging to a newer run untouched. AWS lookup errors still
fail rather than being treated as a missing instance. For expired interrupted launches, use
`cloud-reconcile --cloud SETTINGS --cloud-slot N`.

The default USD 1 budget and four-hour maximum select the shorter deadline at the
configured planning rate of USD 0.26/hour. Budgets apply per job and are not exact
billing caps; transfer, storage, requests, tax and cleanup grace are additional.
Use `prepare --corpus 10000` before cloud-scale. Avoid cloud uploads during local runs.

## Measurement and quality

Each one-shot invocation or persistent sequence is a fresh process with restored inputs and warm filesystem-cache
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
