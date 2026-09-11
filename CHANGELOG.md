# Changelog

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
