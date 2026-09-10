# visiongrep

Search local images with a natural language description or a reference image. VisionGrep is a
Rust CLI that runs CLIP inference locally and caches embeddings in SQLite for repeated searches.
It recursively searches JPEG, PNG, WebP, and BMP files without following symbolic links.

## Build and run

Requires Rust/Cargo and a Unix platform (macOS or Linux).

```sh
cargo build --release
./target/release/visiongrep "a dog on a beach" ./photos
./target/release/visiongrep --image ./reference.jpg ./photos
```

Required model artifacts are downloaded automatically on first use. They are stored in
`$XDG_CACHE_HOME/visiongrep/models/`, or `~/.cache/visiongrep/models/` when `XDG_CACHE_HOME` is unset
or empty. Once the required models are cached, searches can run offline. An image query excludes
the query image itself from results.

## Options

```sh
./target/release/visiongrep "sunset" ./photos --top 10 --threshold 0.3
./target/release/visiongrep "sunset" ./photos --json
./target/release/visiongrep "sunset" ./photos --paths-only
./target/release/visiongrep "sunset" ./photos --index-path ./photos.db
./target/release/visiongrep "sunset" ./photos --no-cache
./target/release/visiongrep --help
```

The default is up to five results with a minimum cosine similarity of `0.25`. Scores are
similarities, not probabilities. Results go to stdout; progress and diagnostics go to stderr.
Use `--null` for exact paths separated by NUL bytes, or `--quiet` to suppress progress.

Exit codes are `0` for matches, `1` for no matches, and `2` for an operational or argument error.

## Persistent stdio searches

```sh
./target/release/visiongrep serve --stdio --index-path photos.db ./photos
```

Keep the process open and send one JSON object per line on stdin:

```json
{"id":1,"query":"a dog on the beach","top":3}
{"id":2,"image":"./reference.jpg","threshold":0.3}
```

Each request receives one flushed JSON line on stdout, in request order:

```json
{"id":1,"results":[{"score":0.31,"path":"./photos/beach-dog.jpg"}]}
{"id":2,"results":[]}
```

Provide exactly one non-blank `query` or non-empty `image` path. `top` defaults to 5 and must
be positive; `threshold` defaults to 0.25 and accepts -1 through 1. The optional `id` is a
string or unsigned 64-bit integer and is echoed back (omitted/null IDs return null). Unknown
fields are rejected. Paths follow the usual CLI rules: relative paths use the process's working
directory, and JSON requires UTF-8 paths. To search for the literal word `serve` in one-shot
mode, place a search option first: `visiongrep --top 5 serve ./photos`.

Before each search, the service scans for additions, modifications and deletions and updates the
index. SQLite stays open; text and vision models load only when needed and remain resident until
exit. Image vectors are read from SQLite for each request. Keeping both models resident uses more
memory than one-shot mode. The search directory is resolved at startup.

Request failures return `{"id":1,"error":"..."}` and the service continues. Malformed JSON or
schema errors return a null ID. Progress and warnings go to stderr. Close stdin to shut down;
EOF and a closed output pipe exit with code 0, even if individual requests failed or had no matches.
Startup/I/O failures and request lines exceeding 1 MiB (including the newline) exit with code 2.
Stop the service before replacing or rebuilding its index with `--reindex`.

## Index cache and reindexing

The default index is `.visiongrep.db` inside the searched directory. Repeated searches reuse
unchanged image embeddings; modification time and file size determine which images need updating.
Use `--index-path PATH` to choose another location, `--no-cache` to bypass the index, or `--reindex`
to rebuild it before returning search results. A custom index belongs to one search root.

**Known limitation: running `--reindex` concurrently with a cached search or another `--reindex`
using the same index can corrupt the SQLite database.** Reindexing replaces the database file
without coordinating other processes that may still have it open. Atomic file replacement does
not make concurrent database access safe. Wait for all users of that index to finish before
starting `--reindex`, and let it finish before accessing the index again. This also applies to
indexes selected with `--index-path`; concurrency protection is not implemented yet.

## Development

```sh
cargo fmt --all --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace -- --test-threads=1
```

The full test suite needs cached model artifacts. See [test setup and coverage](tests/README.md)
for preparation instructions and [benchmarks](benchmarks/README.md) for performance measurements.
