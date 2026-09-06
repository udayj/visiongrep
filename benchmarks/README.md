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
python3 benchmarks/bench.py plan --mode validate
python3 benchmarks/bench.py run --mode validate --detach
python3 benchmarks/bench.py status /absolute/path/to/run
python3 benchmarks/bench.py logs /absolute/path/to/run
python3 benchmarks/bench.py report /absolute/path/to/run
```

The next deliberate step is same-commit validation. It compares the foundation with itself,
checks quality parity, and flags over 5% timing drift. It is an operational rehearsal, not
proof of statistical equivalence. No foundation timings are recorded by installation.

After validation, execute `--mode record` in at least three separate sessions. Cloud records
must come from three fresh instances. Aggregate them and compare candidates:

```sh
python3 benchmarks/bench.py foundation /path/run1 /path/run2 /path/run3 --output /path/foundation.json
python3 benchmarks/bench.py run --candidate COMMIT --baseline /path/foundation.json --target novel_text --detach
```

Builds are cached by commit and OS/architecture with verified binary digests. Use consistent
power settings and minimal competing work locally. The local profile uses the normal
application worker cap, not a four-core limit on the MacBook. Local/cloud measurements have
separate foundations. Changing the harness, profile, corpus lock, model, or environment
invalidates the old measurement contract and requires calibration.

## Profiles and decisions

| Profile | Images | Samples per binary/scenario | Coverage |
|---|---:|---:|---|
| local-quick | 500 | 5 | Indexing, text/image queries, modifications, quality |
| cloud-standard | 500 | 21 | All 14 timing subcases, quality |
| cloud-scale | 10,000 | 3 | Indexing, queries, modifications, deletions, renames |

Before comparisons, three batches of three foundation samples measure novel text, cached
text, and full embedding on the first 500 images, including for the scale profile. Historical
bounds use median +/- max(3 scaled MAD, 3% of median). Excessive spread refuses calibration;
all new batch medians must fit the bounds. Reference CV over 10% invalidates a run. These
starting tolerances must be evaluated during validation, not loosened to pass a candidate.

Samples alternate F/C and C/F. Comparisons use paired median ratios and deterministic
bootstrap intervals with Bonferroni-adjusted alpha across timing scenarios. Fixed budgets
avoid repeated peeking until significance. At 21 samples, p95 is the second-slowest value
and is labeled a rough estimate. Fewer than 20 pairs are screening evidence only.

Declare the target scenario before launching. Qualification requires at least 5% measured
improvement, an interval excluding zero, unchanged quality/behavior, and exclusion of more
than 5% timing regression elsewhere. This does not mean the whole interval exceeds 5%.
Definite regressions fail; unresolved differences are inconclusive. Scale-only results
cannot establish overall quality or qualify a candidate.
Memory/index-size comparisons also guard against more than 5% regressions before qualification.

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
run root/cloud bucket is permitted. Collect a terminated cloud job to release its lock.
Use `cancel RUN` to stop work and `cloud-reconcile --cloud SETTINGS` for expired uncertain
launches. Deadlines/cancellation preserve incomplete reports and cannot pass.

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
