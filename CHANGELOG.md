# Changelog

## Unreleased

- Add `visiongrep index DIR`, which downloads missing models, updates the index and prepares both
  models without running a query, so the first search after it only embeds the query.
- Report indexing progress as `Indexed N/TOTAL images` lines every 256 images when stderr is not a
  terminal, instead of printing nothing until indexing finishes.
- Document use with AI agents in the README.
- Remove the `cargo binstall` metadata; VisionGrep is not published to crates.io.

## 0.4.1

- Publish prebuilt macOS and Linux binaries with each GitHub release, installable with a shell
  installer or `cargo binstall visiongrep`. Add crates.io package metadata for
  `cargo install visiongrep`.

## 0.4.0

- Use a dynamically quantized INT8 vision model with the final projection and first transformer
  block kept in FP32. On COCO val2017 text-to-image retrieval, recall changed by -0.0035,
  -0.0031, and -0.0044 at 1, 5, and 10 respectively. AWS c7a.xlarge end-to-end searches
  are 61.5% faster for external images (871.4 → 335.6 ms) and 59.5% faster after 1% of
  images change (990.0 → 400.6 ms). Text queries are unchanged. Existing indexes re-embed
  images once on first use.

## 0.3.0

- Add `visiongrep serve --socket DIRECTORY`, an opt-in server that keeps the index and
  model sessions loaded. Ordinary cached searches for the same directory and index use it
  automatically while it is running and fall back to one-shot searches otherwise.
  `--no-cache`, `--reindex`, `--verify-models` and `--timing` always run one-shot.
- `serve` now requires exactly one of `--stdio` or `--socket`.
- Refuse `--reindex` while a socket server is running for the same index.
- Save optimized ONNX graphs under `models/optimized/` on first use and reuse compatible
  graphs in later processes to reduce model startup time. The cache adds about 578 MiB of
  disk and falls back to the source models when a graph cannot be used or saved.
- Replace the benchmark harness with `benchmarks/bench.py`: `prepare`, paired
  `compare` and `quality` runs, plus `aws-run`, `aws-fetch` and `aws-list` for
  self-terminating EC2 comparisons.

## 0.2.0

- Add `visiongrep serve --stdio DIRECTORY` for sequential JSONL searches in one process.
  Keep SQLite and lazily initialized text/vision sessions open between requests while
  refreshing the image index before every search.
- Support text and image requests, optional request IDs, per-request limits and
  thresholds, flushed responses, and recoverable request errors.
- Extend the benchmark profiles with persistent text and image-update sequences,
  separating first-use latency from warm queries and recording process peak memory.
  Use the `v0.2.0` release commit as the new benchmark reference.

## 0.1.0

- Initial command-line text and image search with a persistent SQLite image index.
