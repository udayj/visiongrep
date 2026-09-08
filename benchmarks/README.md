# VisionGrep benchmarks

Python 3.12+ development-only orchestration. No Python enters the shipped binary. The
immutable application reference is `benchmark-foundation-v1`, commit
`8b518ed86ff4f29e9c071931c91efa3e5ec31b44`. The independently fingerprinted harness measures
both exported commits. Commit application experiments before measuring; working-tree edits
are not included. Model/preprocessing changes are outside the initial comparison contract.

## Preparation and first validation

Runtime files default to `~/.cache/visiongrep-bench/`, outside the checkout. All commands
accept explicit runtime paths where relevant. Preparation imports or downloads verified
models once. The first corpus preparation freezes image checksums in
`cache/corpora/<selection-sha>.json`; retain this lock with foundation evidence. Later
preparation reuses and verifies the same bytes. Measured runs never download artifacts.
The 500-image source checksums are also pinned in `corpora/image-checksums.json`; the
remaining scale images receive checksums on preparation. Interrupted scale preparation
resumes from per-download receipts without re-fetching completed images.

```sh
python3 benchmarks/bench.py prepare --corpus 500
# Optionally add --import-models /path/to/existing/visiongrep/models
python3 benchmarks/bench.py plan --mode validate --validation-samples 3
python3 benchmarks/bench.py run --mode validate --validation-samples 3 --detach
python3 benchmarks/bench.py status /absolute/path/to/run
python3 benchmarks/bench.py logs /absolute/path/to/run
python3 benchmarks/bench.py report /absolute/path/to/run
```

The next deliberate step is same-commit validation. It compares the foundation with itself,
checks quality parity, and flags over 5% timing drift. It is an operational rehearsal, not
proof of statistical equivalence. No foundation timings are recorded by installation.
For a quick rehearsal, `--validation-samples 3` uses three pairs per profile scenario and
one three-sample calibration batch per cloud calibration scenario; local calibration
always retains three batches. Quality and behavior checks
remain enabled. The short calibration checks variation within its batch; it cannot measure
variation between batches. This option is rejected for comparison and foundation recording.
Omit it to use the profile defaults, or use `--validation-samples 21` for longer validation.
Standard cloud candidate comparisons always retain 21 pairs and three calibration batches.

After validation, execute `--mode record` in at least three separate sessions. Cloud records
must come from three fresh instances. Merge the harness before recording these references;
keep the application tag `benchmark-foundation-v1` unchanged. Recording measures only the
foundation binary, with the profile's normal sample count and one quality pass when enabled.
It skips the separate calibration precheck and derives reference bounds from its recorded
scenario samples. Aggregate the successful sessions and compare candidates:

```sh
python3 benchmarks/bench.py foundation /path/run1 /path/run2 /path/run3 --output /path/foundation.json
python3 benchmarks/bench.py run --candidate COMMIT --baseline /path/foundation.json --detach
```

Builds use the same routine locally and on EC2. Cache keys include the commit, source
digest, actual Rust/Cargo/compiler/linker identities, release profile, target, and SDK.
Binary digests are verified on reuse. Inherited build overrides (including
`RUSTUP_TOOLCHAIN`, Rust flags, Cargo profile settings, and native compiler flags) are
removed; the commit's pinned toolchain is selected explicitly. An isolated Cargo home
reuses downloaded dependencies. Cargo configuration files are rejected until their
overrides can be supported explicitly. The effective build fingerprint must match between
foundation and candidate and in the saved foundation contract. Use consistent
power settings and minimal competing work locally. The local profile uses the normal
application worker cap, not a four-core limit on the MacBook. Local/cloud measurements have
separate foundations. Changing the harness, profile, corpus lock, model, or environment
invalidates the old measurement contract and requires calibration.

## Profiles and decisions

| Profile | Images | Samples per binary/scenario | Coverage |
|---|---:|---:|---|
| local-quick | 500 | 9 | Indexing, text/image queries, modifications, quality |
| cloud-standard | 500 | 21 | All 14 timing subcases, quality |
| cloud-scale | 10,000 | 3 | Indexing, queries, modifications, deletions, renames |

Before comparisons, three batches of three foundation samples measure novel and cached
text, plus no-cache embedding when that scenario is in the profile (cloud-standard).
Prechecks use the profile's full corpus: 500 images for local/standard, 10,000 for scale,
so their measurements match the recorded reference. Scale prechecks therefore also need
to prepare indexes for 10,000 images.

Foundation recording has no separate calibration invocations. Aggregation groups each
reference scenario's measurements into consecutive, non-overlapping triples: seven batch
medians from 21 samples, three from local-quick's nine, or one from cloud-scale's three. Any final
one or two samples still participate in the full-session stability check, but cannot form a
three-sample batch. Cloud bounds use these batch medians across at least three sessions, with
median +/- max(3 scaled MAD, 3% of median). Excessive spread or outlying batch medians
refuse the cloud foundation. All new cloud comparison batch medians must fit its bounds. Reference CV
over 10% invalidates a cloud run. These starting tolerances must not be loosened to pass a candidate.

Local policy `local-median-v3` uses the same fixed nine-sample budget, but judges the
precision of typical latency rather than requiring uniform individual invocations. It
reports the median of batch medians, an approximate 95% hierarchical bootstrap interval,
raw CV, and maximum batch deviation. Local aggregation requires median uncertainty within
10% and batch deviation within 15%, across every recorded scenario. Raw spikes are retained
as diagnostics. Aggregation reassesses all sessions together rather than rejecting a
session merely because its own median estimate is imprecise. Choose at least three sessions
in advance and include all of them, without replacement.

Schema-3 aggregation emits `calibrated` with reference bounds or `inconclusive` with reasons
and no usable bounds. Local prechecks compare the current median with the reference median
interval expanded by 10%, while independently checking current precision and batch stability.
Only matching contracts can be compared. Explicit policy-only reanalysis of the known v2
nine-sample harness is supported, with its source contract and report hashes preserved;
unknown harnesses and old five-sample records require fresh recordings. See
[evidence, compatibility and limits](LOCAL_SCREENING.md).

A stable local comparison can be `promising`: a supported improvement of at least 5%, with
paired intervals excluding regressions over 5% in every local timing and resource check.
Quality and behavior must pass. This is a shortlist for considering cloud validation, never
cloud qualification or an automatic cloud launch. Clear regressions/no useful gain are
`does_not_qualify`; uncertain results remain `inconclusive`. The candidate is paired with
newly measured foundation invocations, not compared only with old recorded medians.

Samples alternate F/C and C/F. Comparisons use paired median ratios and deterministic
bootstrap intervals with Bonferroni-adjusted alpha across timing scenarios. Fixed budgets
avoid repeated peeking until significance. At 21 samples, p95 is the second-slowest value
and is labeled a rough estimate. Fewer than 20 pairs are screening evidence only.

Qualification uses the entire standard suite; there is no required target scenario:

1. Measure every standard scenario.
2. Establish at least one supported improvement of 5% or more.
3. Pass the regression check for every other scenario (and the improved scenarios).
4. Pass quality, behavior, memory, and index-size checks.

A supported improvement means a measured gain of at least 5% with an interval excluding
zero; it does not mean the entire interval exceeds 5%. Intervals remain adjusted across
all timing scenarios, including when selecting the strongest improvement. Regression checks
retain their separate 5% tolerance: the interval must exclude a slowdown greater than 5%.
Memory/index-size checks use the same regression tolerance. Reports name every supported
improvement when a candidate qualifies. Missing scenarios or resource checks cannot pass.
Definite regressions fail; unresolved checks are inconclusive. Local-quick and cloud-scale
are screening profiles and cannot qualify a candidate, even with more samples.
Screening preserves definitive failures and their reasons, including quality, behavior,
timing (when the local timing screen is stable), and resource regressions.

After each invocation, outside timing, behavior checks require the expected result count
and eligible image paths, including image-query self-exclusion. Index checks compare
membership and file metadata against the mutated corpus and embeddings against the
original image embeddings, mapping replacements and renames to their expected source.
Every measured pair also compares foundation/candidate exit codes, returned rankings,
scores, and all persisted embeddings. Missing or failed checks block qualification and
same-commit validation; failures also prevent recording a foundation. These checks use
the existing fixed-model embedding/score tolerance of `1e-4` and exact ranking agreement.
Reports from older harness/build contracts require recalibration.

Reports include external process latency, median/p95/spread, isolated child peak RSS,
checkpointed index bytes and bytes/image, full-index images/second, every application phase,
retrieval/no-match metrics, embedding/score/ranking parity and estimated spend. Parallel
decoding/preprocessing phases sum worker durations and are not additive wall time.

## Scenario state

Models and images are installed. Explicit pre-reading prepares a warm filesystem-cache
workload; OS residency is not guaranteed. Every invocation is a fresh process. Warmup
prepares artifact verification sidecars; normal application validation remains timed.

Each binary owns independent staged images and a seed index. Each sample restores the same
index path/root and mutation state. Setup, checkpointing and uploads stay outside timing.
Scenarios: absent index; forced reindex; no-cache; novel/cached text; external image first/
repeated; indexed image; modified indexed query image; 1% additions; 1% modifications;
1% deletions; 1% renames; and read-only corpus with an external index.

Cloud measurements explicitly flush the scenario filesystem and require 250 ms
of observed disk inactivity, zero writeback, and at most 16 MiB dirty memory before
launching the timed CLI. The 30-second setup deadline includes flushing. Settling
evidence is retained per observation; failure invalidates the run. This applies
to normal recording, validation and comparisons as well as diagnostics.
See `SQLITE_DIAGNOSTIC.md` for the validating experiment. SQLite durability and
the foundation variation threshold are unchanged.

External query images are transient: repeated queries still infer embeddings. Indexed
query images reuse their stored vector and exclude themselves. Modifications replace pixels
with a fixed different source image and do not accumulate across repetitions.

## Corpus and absence evidence

`corpora/coco-500.json` and `coco-10000.json` freeze COCO IDs, URLs, pixel dimensions and
original licenses from a checksummed official annotation archive. The local subset covers
all 80 annotated categories. The larger selection contains that subset, the rest of val2017,
and a fixed train2017 selection. Selection never uses model scores. This is a runtime and
retrieval benchmark, not an uncontaminated model-generalization evaluation.

`corpora/quality-500.json` contains 200 caption queries and 240 category prompts grouped into
80 intents. Each absent gallery excludes every image annotated with that category from the
500-image corpus. The present counterpart uses the full corpus. Paraphrases are averaged
within intent, never counted as independent concepts.
No-match reports include intent-bootstrap intervals, conditional on this fixed image corpus.

The CLI returns all 500 scores and the evaluator restricts them to the absent gallery at
threshold 0.25. This avoids re-embedding almost-identical galleries and measures abstention
on that score set, not a separate timed invocation of every absent gallery. Top-K-only
results are rejected. False positives mean any accepted result in an absent gallery;
false negatives mean no accepted result in a present gallery. Retrieval metrics separately
measure correct-image retrieval.

These negatives are **annotation-defined, not human-certified semantic absence**. COCO may
miss objects; caption relevance can be incomplete. Unmentioned caption content is never
treated as proof of absence. Future semantic-accuracy claims need reviewed hard negatives
and holdout intents. The 10,000-image suite does not reuse 500-image absence labels.

Regenerate selections with `corpora/build_manifests.py ANNOTATION_ZIP`. Image bytes, models,
indexes and raw logs stay outside Git. Image vectors must stay normalized and within 1e-4
absolute tolerance, with identical top-10 rankings and quality metrics. Unexpected quality
changes require investigation rather than automatically declaring them improvements.

## Cloud, reports, and cost

See `infra/README.md` for the pinned AMI and CloudFormation prerequisites. Cloud profiles
require an explicit out-of-tree settings file. Plans never provision resources.

```sh
python3 benchmarks/bench.py prepare --corpus 10000
python3 benchmarks/bench.py plan --profile cloud-standard --mode validate --cloud /path/cloud.json
python3 benchmarks/bench.py run --profile cloud-standard --mode validate --cloud /path/cloud.json
python3 benchmarks/bench.py status /path/run
python3 benchmarks/bench.py collect /path/run
```

The default USD 1 budget and four-hour maximum select the shorter deadline at the planning
rate of USD 0.26/hour. This is not an exact billing cap; transfer, retained storage, requests,
tax and cleanup grace are additional. Pilot measurements are needed for duration estimates.

Local reports persist in the run directory. Cloud reports go to private S3 `runs/RUN_ID/`
and collect under `RUN/remote/`. Config, status, raw observations, JSON, and standalone HTML
are retained. Cloud jobs survive disconnects and have independent expiry. One run per local
run root is permitted; do not run simultaneous local measurements on the same Mac.
Cloud runs use `--cloud-slot 1`, `2`, or `3` (default `1`): each slot permits one
active run on its own fresh instance and volume. Launch in separate terminals or
background processes using different slots. Collection releases only that slot
using a conditional lease deletion. Slot one remains compatible with older handles.
Budgets and deadlines apply per run: three simultaneous runs can consume three
times the individual budget. Each instance has its own benchmark directory and
build cache; shared S3 artifact objects are content-addressed.
Collect a terminated cloud job to release its lock.
Use `cancel RUN` to stop work and `cloud-reconcile --cloud SETTINGS` for expired uncertain
launches; pass the original `--cloud-slot` when reconciling slots 2 or 3.
Deadlines/cancellation preserve incomplete reports and cannot pass. `--wait` waits
for termination and collects automatically in the authenticated launch session.

After this update, record both foundations anew with the final merged harness:
at least three local sessions sequentially, and at least three cloud sessions on
distinct fresh instances (slots 1–3 may run concurrently). Complete cloud launches
before starting local timing so bundle uploads do not compete with measurements
on the Mac. Preserve older foundations as historical evidence. Never combine
different contracts or accept diagnostic reports as foundation recordings.

## Development and history

```sh
python3 -m unittest discover -s benchmarks/tests -v
python3 -m compileall -q benchmarks/harness benchmarks/bench.py
```

Offline checks cover decisions, manifests, scenario restoration, RSS, deadlines and
cancellation. They do not replace same-commit or paid infrastructure validation.
`HISTORICAL.md`, `RESULTS.md` and `results/` preserve earlier experiments. Their old heavy
workflow and timing runner were removed; exact reproduction code remains in the foundation
tag. Historical retrieval/model-contract adapters are not in routine candidate runs.
