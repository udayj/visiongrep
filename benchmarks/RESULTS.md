# Retrieval and performance foundation results

Evidence recorded on 2026-08-31. Metrics in this report are corpus- and runner-specific; raw cosine
similarity is not a probability or confidence. The corresponding machine-readable summary is
`results/remote_foundation.json`.

## Architecture and functional changes

- The default encoder is the pinned 256 px DataComp CLIP family used by rclip v3.3.0, with the
  matching OpenCLIP tokenizer and Pillow 12.3-compatible evaluation preprocessing.
- Opt-in timing emits one JSON document with environment/cache metadata, individual phases, and
  total wall time. Normal output and logging are unchanged when timing is disabled.
- Image discovery is sorted once and merge-walked against one ordered metadata query. Reconciliation
  and image updates remain transactional and retain native Unix path bytes and nanosecond mtimes.
- Indexing uses at most four scoped preprocessing workers, ordered result restoration, batches of
  eight, and one ONNX inference consumer. Channels and memory are bounded independently of corpus
  size; inference and batch-cardinality failures abort.
- Ranking uses a deterministic bounded heap and retains exact score-descending/path-ascending tie
  semantics.
- Verified-install manifests avoid rehashing unchanged model files. Downloads are streamed through
  SHA-256, checked before atomic installation, and can still be fully checked with `--verify-models`.
- `--index-path PATH` keeps the compatibility default while enabling a root-bound, out-of-tree index
  for read-only image trees. Staged reindexes live beside the selected destination.
- Persisted schema version 3 records the canonical search root and independent image/query embedding
  contracts, so query-only contract changes do not discard image embeddings.

## Model contract evidence

The manually triggered model-contract workflow ran on Linux x86_64 with Python ONNX Runtime 1.24.2
and the Rust CPU ONNX implementation. It compared the rclip ONNX conversion with the original
OpenCLIP safetensors, all at immutable revisions.

| Component | Immutable identity | SHA-256 |
|---|---|---|
| OpenCLIP source weights | `4afec35ffe57a943d569ff7ee888061830164da8` | `92c26d60d3200ed5ed040dff31a8d19f8140648da8007216c25744c478deef27` |
| rclip vision ONNX | `17b9d07433aad73f70d338d8a1c7a4cef83887e0` | `3f7e6f94e5a34bc7ee8aba84aec0f963f56974ab405fbcd334c8e1c3f832bd2c` |
| rclip text ONNX | `17b9d07433aad73f70d338d8a1c7a4cef83887e0` | `ee267cd64f0f77362670ae0140476ed51ee8c5a761d41636e09997f2fdddcacc` |
| OpenCLIP tokenizer JSON | `4afec35ffe57a943d569ff7ee888061830164da8` | `72ed5c96db5729294468543e4bc75fce14ca63f58e37300290189ba1c1e52b85` |
| rclip BPE vocabulary | `17b9d07433aad73f70d338d8a1c7a4cef83887e0` | `924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a` |

| Check | Recorded result | Gate |
|---|---:|---:|
| Token IDs | exact | exact |
| Preprocessed fixture tensors | maximum absolute error 0 | exact |
| Normalized text embeddings | max abs 2.38e-7; min cosine 0.99999988 | 1e-4; 0.99999 |
| Normalized image embeddings | max abs 5.66e-7; min cosine 0.99999994 | 1e-4; 0.99999 |
| Cosine scores | max abs 3.05e-7 | 1e-4 |
| Golden top-K rankings | exact | exact |
| Batched versus single inference | exact within fixture tolerance | same ranking/threshold behavior |

Model-contract workflow: run `33380886362`, measured commit
`e7d11874e7582b44274af8e48fc080b2475bc371`.

## Retrieval quality versus pinned rclip

The shared corpus contains 500 rows from the pinned COCO Caption 2017 validation revision plus two
deterministic CC0 text-heavy fixtures. It uses 114 positive and 10 deliberately absent queries.
rclip is pinned to v3.3.0 / commit `3dcec2de5e23311473f6fb6433e602aa4f4ca812`.

| System | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 | P@1 | P@5 | P@10 | FP absent | FN positive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VisionGrep baseline | .4912 | .7105 | .8421 | .5957 | .6545 | .4912 | .1421 | .0842 | .5000 | .1842 |
| rclip 3.3.0 | .5789 | .8596 | .8947 | .6830 | .7345 | .5789 | .1719 | .0895 | .3000 | .1754 |
| VisionGrep DataComp | .5789 | .8596 | .8947 | .6838 | .7353 | .5789 | .1719 | .0895 | .3000 | .1754 |

At the existing 0.25 threshold, DataComp matches rclip on Recall, precision, and both no-match error
rates, while MRR and nDCG are slightly higher. VisionGrep coverage is .9737 with .5946 top-1 precision
when covered and .3000 absent-query false-positive rate. At thresholds .30 and .35, those values
are respectively (.6667, .6447, .1000) and (.1316, .9333, 0). Ten absent queries are not enough
to responsibly recalibrate the default, so 0.25 remains explicitly provisional.

The golden PNG fixtures establish implementation parity within the documented tight tolerance. On
the JPEG COCO corpus, rclip's Pillow/libjpeg decoding and Rust's `image` JPEG decoding introduce
small pixel-dependent differences:
top-1 paths match for 98.39% of queries, top-10 membership for 91.13%, full ordered top-10 for
39.52%, threshold-accepted path sets for 91.94%, and no-match decisions for 99.19%. The maximum
score difference on shared paths is .00520. Only 2 of 124 top-1 paths and 1 of 124 no-match decisions
differ; this is reported as evidence rather than hidden behind a weakened correctness assertion.

## End-to-end retrieval observations

Linux GitHub-hosted runner: 4 logical CPUs, 16.77 GB memory, Linux 6.17 Azure x86_64, Rust 1.89.0.
The rclip environment used Python 3.12.14, Pillow 12.3.0, NumPy 2.1.3, and Python ONNX Runtime 1.28.0;
the VisionGrep binary is compiled against pinned ONNX Runtime 1.24.2 with the CPU provider.

| System | First index + query | Images/s | Cached query | Peak RSS | Index bytes |
|---|---:|---:|---:|---:|---:|
| VisionGrep baseline | 35,555 ms | 14.12 | 16.43 ms | 507,132 KiB | 2,605,056 |
| rclip 3.3.0 | 40,251 ms | 12.47 | 50.64 ms | 965,892 KiB | 2,195,456 |
| VisionGrep DataComp | 40,058 ms | 12.53 | 11.22 ms | 659,172 KiB | 2,613,248 |

VisionGrep DataComp's first-index result is 0.48% faster than rclip in this run, with 31.8% lower
peak RSS and a 19.0% larger SQLite index. Its exact cached query is 4.5x faster. The rclip adapter's
50.62 ms median novel-query latency is session-warm in one Python process, while VisionGrep's 567.11
ms median is process-cold and includes ONNX session construction; those novel-query values are not
a fair cross-system speed comparison.

## Measured implementation choices

- On 256 generated images with the previous 224 px baseline contract, bounded preprocessing plus
  batch-8 inference reduced median total indexing wall time from 14,360 ms to 10,534 ms (26.6%) over
  seven samples with byte-identical persisted embeddings. Four preprocessing workers delivered
  510.5 images/s versus 193.6 with one worker.
- On the DataComp encoder, 21-sample batch tests measured 20.33, 20.79, 20.93, 21.64, and 23.46
  images/s at batch sizes 1, 2, 4, 8, and 16. Batch 8 is the conservative choice: batch 16 improved
  median throughput by only 8.4%, doubled tensor memory, and raised p95 batch latency from 399.6 ms
  to 1,022.4 ms.
- The deterministic heap matched full-sort results exactly in every 21-sample case. At 100,000
  normalized embeddings it was 1.16-1.17x faster at threshold .25 and 1.42-1.47x faster at threshold
  -1 for K values 5, 10, and 100.
- The 21-sample reconciliation microbenchmark compared the former per-discovered-path SQL lookup
  with the ordered bulk pass. At 10k paths, bulk medians were 2.49 ms unchanged and 2.34 ms at 1%
  changed versus 6.34 and 6.09 ms (2.54-2.60x). At 100k, bulk medians were 42.75 and 43.19 ms
  versus 90.73 and 93.50 ms (2.12-2.16x). All plans matched exactly.
- In seven process-cold novel-query samples requiring the text model and tokenizer, verified
  manifests reduced median artifact validation from 170.00 ms to 0.108 ms. Median end-to-end wall
  time fell from 806.69 ms with explicit full SHA-256 verification to 623.78 ms (22.7%).

## Phase timing scenarios

The 21-sample remote scenario run records process-cold invocations with filesystem cache state
labelled warm/uncontrolled. GitHub-hosted runners do not expose safe cache eviction. Model-absent
network time is a separate one-sample observation and is never mixed into installed-model medians.

| Scenario | Samples | Total median | Total p95 |
|---|---:|---:|---:|
| Models installed, index absent | 21 | 40,024 ms | 40,152 ms |
| `--no-cache` | 21 | 40,083 ms | 40,164 ms |
| Unchanged index, novel query | 21 | 545.59 ms | 561.02 ms |
| Unchanged index, exact cached query | 21 | 7.62 ms | 7.70 ms |
| Approximately 1% changed (5/502 images) | 21 | 1,307.36 ms | 1,344.25 ms |
| Read-only tree, out-of-tree index | 21 | 40,055 ms | 40,206 ms |
| Models and index absent; network-variable | 1 | 78,488 ms | not reported |

Vision inference dominated first-index scenarios at a 36,225 ms median. For the 1%-changed case,
vision inference was 367.08 ms and session construction was 840.01 ms; for a novel warm-index
query, session construction was 474.42 ms of the 545.59 ms total. The exact cached-query path
loaded/deserialized vectors in 2.17 ms and completed in 7.62 ms without constructing a model
session. The one model-absent sample spent 38,455 ms downloading immutable artifacts.

Full per-phase medians and p95 values, environment metadata, cache state, and model checksums are in
`results/phase_timing.json`. Heavy workflow run `33382447637` measured commit
`0efc7e458500bcddece17662ef67414cd48d924d`. A later final-code seven-sample measurement records
the verified artifact-marker fast path because that phase's marker-hit attribution was corrected
after the scenario run; scenario total-wall measurements are unaffected.

## Apple Silicon Core ML decision

Run `33380886290` used a real `macos-15` arm64 GitHub-hosted runner and 502 images over five repeated
end-to-end samples. Core ML passed the correctness gate: maximum normalized-embedding error
7.38e-7, minimum cosine 0.99999994, exact top-K paths/order, identical threshold/no-match decisions,
and deterministic repeats.

| Provider | Median first index + query | Median throughput | Median peak RSS |
|---|---:|---:|---:|
| CPU ONNX | 32,074 ms | 15.65 images/s | 1,063,168 KiB |
| Core ML | 74,687 ms | 6.72 images/s | 697,808 KiB |

Core ML is deferred: throughput was 57.1% lower and first-index wall time was 132.9% higher. The
performance gate required either 20% higher throughput or 15% lower wall time. No Core ML production
code or dependency is retained; CPU ONNX remains the reference.

## Optimization disposition

| Area | Decision | Evidence or reason |
|---|---|---|
| DataComp default model | Implemented | OpenCLIP parity and aggregate rclip quality/no-match gates pass |
| Phase timing | Implemented | Opt-in JSON; normal CLI behavior is unchanged |
| Parallel preprocessing and batch-8 inference | Implemented | 26.6% baseline indexing wall reduction; deterministic output |
| Bulk reconciliation | Implemented | One ordered metadata pass and merge walk; transactional tests |
| Verified artifact manifests | Implemented | Common load avoids full hashes; explicit verification retained |
| Bounded top-K heap | Implemented | Exact semantics; up to 1.47x selection speedup at 100k |
| Out-of-tree root-bound index | Implemented | Read-only-tree scenario and wrong-root tests |
| Stream scoring from SQLite | Deferred | Keep the simpler materialized index API unless phase profiles show a material bottleneck |
| Core ML production path | Measured, deferred | Correct, but substantially slower end to end on Apple Silicon |
| Filesystem-cache-cold claims | Blocked | Hosted runners provide no safe cache eviction mechanism |
| ANN/HNSW, OCR, prompt expansion, multi-crop, GUI, daemon | Out of scope | Explicit branch scope |

## Reproduction and quality gates

Use the commands in `benchmarks/README.md` for the pinned corpus and system adapters. The canonical
isolated entry points are the `Model contract smoke test`, `Heavy retrieval and performance
benchmark`, and `Apple Silicon Core ML experiment` manual workflows.

The Rust 1.89 quality gate is:

```sh
cargo fmt --all
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo build --release
```

The model-contract workflow additionally runs the five named ignored Rust tests documented in
`tests/fixtures/README.md`. Standard tests cover contract invalidation, bulk reconciliation, stale
and renamed files, native paths, deterministic top-K ties, threshold boundaries, manifests and full
verification, bounded preprocessing, batch failure propagation, index-path conflicts, wrong-root
rejection, staged atomicity, and read-only roots.

## Limitations and follow-up

- The quality corpus is deliberately small and COCO-caption judgements are incomplete; precision is
  conservative because unlabelled images may also be relevant.
- The default threshold remains provisional until a larger labelled positive/absent corpus supports
  calibration.
- JPEG decoder differences prevent byte-identical rclip/VisionGrep rankings on the full COCO corpus,
  despite exact pinned preprocessing and golden PNG model parity.
- Warm novel-query latency is not directly comparable between the process-cold Rust CLI and rclip's
  in-process Python adapter.
- Larger 10k/50k real-corpus index-size and sustained indexing runs are appropriate follow-up work;
  the committed 10k/100k data is synthetic and restricted to ranking/reconciliation microbenchmarks.

No model weights, datasets, downloaded images, SQLite indexes, caches, generated binaries, virtual
environments, or large raw benchmark artifacts are committed.
