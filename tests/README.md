# CLI end-to-end tests

The 25 regular CLI tests and five model correctness tests run by default with `cargo test`, alongside
the deterministic unit tests. The four performance benchmarks and the CLI test that deliberately
starts without models and downloads everything remain ignored.

## Local setup

`XDG_CACHE_HOME` is an optional environment variable selecting the cache base directory. VisionGrep
stores models in `$XDG_CACHE_HOME/visiongrep/models/`; if it is unset or empty, it uses
`~/.cache/visiongrep/models/`. Tests use the same lookup rules. Existing pinned DataComp artifacts
are reused; the older `clip_*.onnx` models are not compatible substitutes.

Prepare models once, then run the regular suite:

```sh
cargo build --release
sh scripts/prepare-test-models.sh
cargo test --workspace -- --test-threads=1
```

The setup script runs the actual CLI on one generated image with no index cache. This downloads
only missing/invalid artifacts and verifies them using production code. It requires no Python or
extra Rust dependencies. Run it again when pinned artifacts change. With models already installed,
subsequent test runs need neither the setup command nor network access.

The model correctness tests cover single versus batched inference, dynamic batch dimensions,
image/text golden vectors, and reference scores/rankings/thresholds. They load the shared installed
artifacts directly through `VisionSession::load` / `TextSession::load`, which do not download or
replace artifacts. Each test owns its sessions and any generated images. Symlinks are needed by the
CLI fixtures to isolate cache writes, but are unnecessary for these read-only model tests.

To keep test models separate from your normal cache, set `XDG_CACHE_HOME` before both setup and tests:

```sh
export XDG_CACHE_HOME=/absolute/path/to/test-cache
```

Run just the regular CLI E2Es in release mode:

```sh
cargo test --release --test text_queries --test image_queries -- --test-threads=1
```

## Isolation, parallelism, and CI

Tests launch the real executable with small PNG corpora and inspect its output, exit codes, timing
reports, and SQLite state. Each test gets a temporary corpus, index, and cache directory. Its model
files are symlinks to the shared installation; verification manifests and mutations stay isolated.
Missing fixture models fail with setup instructions instead of silently skipping a test. An
unavailable proxy makes accidental download attempts fail.

Rust runs tests concurrently by default. The fixtures support this without shared cache writes,
but each running CLI process may load its own model sessions. Use `--test-threads=1` to bound memory
and CPU use. No test depends on another test running first, even when execution is sequential.

Push/PR CI prepares models **before** testing on both Linux and macOS. Each matrix job has a separate
runner. Each runner restores model artifacts with `actions/cache`, then runs the setup script
once to verify the cache and download any missing artifacts. It runs the whole regular suite
serially. A cache hit requires no model downloads; a cache miss downloads one set per runner, not
per test. On the first run both OS jobs may download a set because they cannot share live files.

The manually triggered model-contract workflow also runs these E2Es. Its optional
`include_download_test` input additionally exercises a genuinely empty model cache. Run that test
locally with network access (about 610 MB):

```sh
cargo test --release --test text_queries \
  text_query_without_index_or_models_downloads_and_caches_artifacts -- --ignored --nocapture
```

The cold-download test is not a setup step: its artifacts are temporary and intentionally isolated.

## JSON contract assertions

`JsonSearchResult` is a test-side consumer of the executable's JSON output. It deliberately checks
field names and rejects unexpected fields independently of the internal Rust ranking type. Sharing
the production type here could allow a schema change to update both producer and assertion together.
Tests of internal Rust APIs use the production types directly; these E2Es test the process boundary.

## Coverage

| Scenario | Checks |
| --- | --- |
| Text query, no index or models | Downloads and verifies all three artifacts, creates index/query cache, repeat uses no models |
| Text query, no index, models installed | Recursive discovery, results, persisted images/query, no downloads |
| Text query, warm index | Repeated query uses no inference; novel query runs only text inference; warm search also works after removing models |
| Changed corpus | Deleted image removed, modified image re-embedded, new image added, unchanged embedding preserved |
| Reindex | Re-embeds unchanged images, replaces old query cache; failed text/image query preserves the previous database |
| No cache | No index created; existing index remains byte-identical; even malformed SQLite is ignored |
| External index | Relative path resolves from cwd, reuse and reindex work, no root-local index created; another root is rejected |
| Output formats | JSON schema, finite descending scores, matching text/paths/NUL output, quiet stderr, paths with spaces, early pipe closure |
| Ranking controls | Top-K limit and no-match threshold with exit code 1 |
| Image query, no index | External query builds index; no-cache internal query excludes itself without creating an index |
| Indexed image query | Reuses embedding without models, excludes itself before top-K, resolves symlink aliases |
| External image query | Uses vision inference, remains outside persisted index, responds to query-file changes |
| Invalid or empty inputs | Exit 1 for empty corpus; exit 2 for missing roots, non-directory roots, invalid query images and conflicting/invalid arguments |
| Corrupt corpus images | New and previously indexed corrupt images are skipped; stale embeddings are removed and repaired files return; no-cache preserves the existing index; all-corrupt corpora return exit 1 without vision inference |
| Ingestion batches | Nine valid images cross the vision batch boundary with and without cache; 257 files cross the database batch boundary with only two valid images requiring inference |
| Unavailable download | Exit 2, empty stdout, no partial artifacts or cached embeddings |
| Timing output | Timing JSON on stderr preserves result JSON on stdout; invalid timing destinations return exit 2 |
| Explicit model verification | `--verify-models` works with installed artifacts and creates the query cache without downloads |

Inline unit tests also cover non-finite embeddings and malformed embedding byte lengths, invalid
custom index paths, preservation of newer database schemas, artifact content-length/body-size
mismatches, text path escaping, and preservation/classification of typed errors.
Scanner tests cover missing/disappearing roots, unreadable nested directories, pre-epoch timestamps,
supported extensions and casing, ignored symlinks, sorted recursive discovery, and size/mtime snapshots.
Permission and pre-epoch assertions report a skip when the environment cannot represent the condition
(for example, root can read mode-000 directories).

Synthetic images test CLI and cache behavior without depending on subjective semantic rankings.
Text/image embedding accuracy against OpenCLIP is covered separately by the golden-vector tests
in [fixtures/README.md](fixtures/README.md).
