# VisionGrep benchmarks

The purpose is to identify meaningful application improvements. Environment calibration
is a coarse stability check; paired candidate/reference measurements determine gains
and regressions. Do not tune acceptance rules to individual candidates.

Requires Python 3.12+. The application reference is `benchmark-foundation-v1`
(`8b518ed86ff4f29e9c071931c91efa3e5ec31b44`). Commit candidates before measuring;
working-tree changes are excluded. Model or preprocessing changes need a separate evaluation.

## Preparation and first validation

Runtime files stay outside Git in `~/.cache/visiongrep-bench/`. Preparation downloads
verified models and images; measured runs reuse them. Retain the corpus checksum lock
under `cache/corpora/` with the recordings.

```sh
python3 benchmarks/bench.py prepare --corpus 500
# Optionally add --import-models /path/to/existing/visiongrep/models
python3 benchmarks/bench.py plan --mode validate --validation-samples 3
python3 benchmarks/bench.py run --mode validate --validation-samples 3 --detach
python3 benchmarks/bench.py status /absolute/path/to/run
python3 benchmarks/bench.py logs /absolute/path/to/run
python3 benchmarks/bench.py report /absolute/path/to/run
```

Validation compares the foundation with itself and checks quality, behavior and timing
drift. Three pairs give a short rehearsal; omit `--validation-samples` for the full
budget. Validation does not record a foundation.

For a new foundation, plan at least three recording sessions before measuring. Run local
sessions sequentially with consistent power settings and minimal competing work. Use a
fresh instance for each cloud session. Include all planned sessions without replacing
noisy runs or discarding samples. Aggregate them and compare a committed candidate:

```sh
# Run once per planned session; retain each output directory.
python3 benchmarks/bench.py run --mode record --profile local-quick
python3 benchmarks/bench.py foundation /path/run1 /path/run2 /path/run3 --output /path/foundation.json
python3 benchmarks/bench.py run --candidate COMMIT --baseline /path/foundation.json --detach
```

Save local aggregates outside the checkout, for example under
`~/.cache/visiongrep-bench/local-foundation-<date>.json`. A later session on the same
machine can find saved aggregates with:

```sh
ls ~/.cache/visiongrep-bench/local-foundation-*.json
```

Select an aggregate with verdict `calibrated` and pass its full path with `--baseline`.
The runner checks its measurement contract before use; it does not automatically select
the newest file. Raw session reports remain under `~/.cache/visiongrep-bench/runs/`.
These files are local to the machine and are not supplied by a new Git checkout.

The runner checks the reference commit, profile, corpus, models, machine and build
settings before using a saved foundation. CPU microcode remains in diagnostic metadata
but does not require an exact match: AWS manages host firmware, and each candidate is
compared with the reference on the same instance. Local and cloud foundations are separate.
The harness fingerprint hashes the sorted paths and contents of all Python and JSON
files under `benchmarks/`, including tests and fixtures. It excludes Markdown and the
current Git commit. Documentation-only changes do not require new recordings.

## Profiles and decisions

| Profile | Images | Samples per binary/scenario | Coverage |
|---|---:|---:|---|
| local-quick | 500 | 9 | Indexing, text/image queries, modifications, quality |
| cloud-standard | 500 | 21 | All 14 timing subcases, quality |
| cloud-scale | 10,000 | 3 | Indexing, queries, modifications, deletions, renames |

Comparisons precheck the foundation with three batches of three samples for novel and
cached text, plus no-cache embedding on cloud-standard. Prechecks use the full corpus.
Recording has no separate precheck. Cloud policy `cloud-session-median-v1` requires
at least three independent instances. For novel text, cached text, and (cloud-standard)
no-cache embedding, it takes the median of each complete session, then the median of
those session medians. Every session median must lie within ±10% of that center;
these same bounds check the complete precheck's median on future comparison runs.
Individual three-sample batches do not have to fit an outlier fence. All raw samples
are retained, and within-session reference CV over 10% still invalidates a cloud run.
These bounds are practical drift limits, not confidence intervals or a relaxation of
the 5% candidate improvement/regression rules. Three hosts provide limited evidence
about fleet-wide variability.

Local policy `local-median-v3` groups each session's nine samples into three consecutive
triples. It estimates typical latency as the median of batch medians across sessions.
An approximate 95% bootstrap interval resamples sessions, then batches within sessions.
Every scenario must meet both limits:

- Median uncertainty: the larger distance to an interval endpoint is at most 10% of
  the median.
- Batch stability: every batch median is within 15% of the aggregate median.

Aggregation evaluates all sessions together and retains raw samples and CV diagnostics.
It returns `calibrated`, or `inconclusive` without usable bounds. Local prechecks require
the same precision and stability, with the current median inside the saved interval
expanded by 10%. Do not adjust limits per candidate.

Analysis-only reaggregation is supported for the known v3 harness fingerprint
`f6750c4af39c4f55de442f197e217394fe24f9ef01c40d87e17bfa38e95feec9`
(local and cloud), plus the already supported local v2 nine-sample harness. The
`foundation` command accepts these original recording directories, preserves their
source contracts and report hashes, and writes the current analysis contract. It does
not rewrite raw reports. Unknown harnesses, old five-sample local records, and other
measurement contract changes still require fresh recordings.

After an analysis update, regenerate aggregates before comparison, using the same
three planned sessions. For cloud, pass each collected `RUN/remote` directory. For
local, use the original directories listed in the saved aggregate's `reports` field.
Always choose a new output filename; original aggregates remain immutable. Local
median estimates and thresholds are unchanged by the cloud calibration update.

With few sessions, correlated measurements or changing conditions can make the interval
optimistic. This policy screens typical latency, not tail latency or occasional stalls.
Cloud confirmation is required for performance claims. Audit saved reports with:

```sh
python3 benchmarks/audit_local.py ~/.cache/visiongrep-bench/runs > /tmp/local-audit.json
```

Candidates and fresh references alternate execution order. Comparisons use paired median
ratios and bootstrap intervals adjusted across timing scenarios.

A supported gain is at least 5% with an interval excluding zero. Every timing and
resource interval must exclude regressions over 5%. Quality and behavior must pass;
missing checks cannot pass. Definite failures produce `does_not_qualify`; uncertainty
produces `inconclusive`. Only the full cloud-standard suite can return `qualifies`.
A stable local pass is `promising`, a shortlist outcome that does not launch a cloud run.
Local-quick and cloud-scale cannot qualify candidates, even with more samples.

Reports retain raw observations, process latency, phase timings, peak child memory,
index size, quality checks and HTML summaries. Worker-summed phase times are not additive
wall time. At these sample counts, p95 is only a rough estimate.

## Measurement conditions

Each invocation is a fresh process with restored inputs and warm filesystem-cache
preparation. Setup, checkpointing, uploads and behavior checks are outside timing.
External image queries infer embeddings each time; indexed image queries reuse stored
vectors and exclude themselves. Cloud runs flush setup writes and require storage to
settle before timing; failure invalidates the run.

## Corpus and absence evidence

COCO manifests in `corpora/` fix the image selections and licenses. The 500-image suite
uses 200 caption queries and 240 category prompts across 80 intents. Absent galleries
exclude images annotated with the queried category; paraphrases are grouped by intent.
No-match evaluation filters all 500 scores at threshold 0.25, outside timing.

Annotation-defined absence is not proof of semantic absence. These results do not
establish model generalization; the 10,000-image suite does not reuse these absence labels.
Comparisons require normalized embeddings, score/vector agreement within `1e-4`, exact
ranking agreement and behavior parity. Investigate unexpected quality changes.

## Cloud, reports, and cost

See [infrastructure setup](infra/README.md) for prerequisites. Cloud profiles
require an explicit out-of-tree settings file. Plans never provision resources.

```sh
python3 benchmarks/bench.py plan --profile cloud-standard --mode validate --cloud /path/cloud.json
python3 benchmarks/bench.py run --profile cloud-standard --mode validate --cloud /path/cloud.json
python3 benchmarks/bench.py status /path/run
python3 benchmarks/bench.py collect /path/run
```

Use `prepare --corpus 10000` before running `--profile cloud-scale`.

The default USD 1 budget and four-hour maximum select the shorter deadline at the planning
rate of USD 0.26/hour. This is not an exact billing cap; transfer, retained storage, requests,
tax and cleanup grace are additional. Pilot measurements are needed for duration estimates.

Cloud jobs survive disconnects. `--wait` waits for termination and collects artifacts;
interrupting the waiter does not stop the worker. Use `cancel RUN` to stop it.
Reports persist in private S3 and collect under `RUN/remote/`.

`--cloud-slot 1`, `2` or `3` permits three concurrent jobs on separate instances;
budgets apply per job. Collect terminated jobs to release their slots. For expired,
uncertain launches, use `cloud-reconcile --cloud SETTINGS` with the original
`--cloud-slot`. Avoid cloud uploads during local measurements. Incomplete or cancelled
runs cannot pass.

For storage investigations, use `--mode diagnose --profile cloud-standard --max-hours 1`
with cloud settings. The fixed 800-invocation protocol retains behavior checks and omits
quality inference. Reports include `telemetry.jsonl` and traced `syscalls.log` files.
Tracing adds overhead. `diagnostic_complete` confirms protocol completion, not a cause;
diagnostic reports cannot calibrate a foundation.

## Development and history

```sh
python3 -m unittest discover -s benchmarks/tests -v
python3 -m compileall -q benchmarks/harness benchmarks/bench.py
```

Offline tests do not replace live validation. [Historical tooling](HISTORICAL.md) and
[model results](RESULTS.md) document earlier experiments; reproduction code remains in
the foundation tag.
