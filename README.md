# visiongrep

Search local images with a natural language description or a reference image. VisionGrep is a
Rust CLI that runs CLIP inference locally and caches embeddings in SQLite for repeated searches.
It recursively searches JPEG, PNG, WebP, and BMP files without following symbolic links.

It is built to be called by scripts and AI coding agents as well as people: results go to stdout
in stable text, JSON or NUL-separated formats, diagnostics go to stderr, and exit codes separate
"no match" from errors. See [Use with AI agents](#use-with-ai-agents).

## Install

Prebuilt binaries are available for Apple Silicon macOS and Linux (x86_64 and arm64, glibc 2.39
or newer, such as Ubuntu 24.04). Intel Macs are not supported because ONNX Runtime publishes no
prebuilt CPU binaries for them. Install the latest release into `~/.cargo/bin`:

```sh
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/udayj/visiongrep/releases/latest/download/visiongrep-installer.sh | sh
```

```sh
visiongrep "a dog on a beach" ./photos
visiongrep --image ./reference.jpg ./photos
```

To build from a checkout instead, run `cargo build --release` and use
`./target/release/visiongrep`. Building needs Rust 1.89 or newer and a C compiler; the optimized
build takes several minutes.

Required model artifacts are downloaded automatically on first use. They are stored in
`$XDG_CACHE_HOME/visiongrep/models/`, or `~/.cache/visiongrep/models/` when `XDG_CACHE_HOME` is unset
or empty. Once the required models are cached, searches can run offline. An image query excludes
the query image itself from results.

The DataComp CLIP vision model uses dynamic INT8 quantization, with the final projection and first
transformer block kept in FP32. The vision model is about 119 MB (down from 352 MB); the FP32 text
model and tokenizer add about 257 MB. Changing the vision model causes existing indexes to re-embed
images once on their next use.

On first use, VisionGrep also saves an optimized ONNX graph under `models/optimized/` for each
model it loads. Later processes reuse compatible graphs to reduce model startup work. The vision
and text graphs are cached separately; the tokenizer is unchanged. The first use still pays the
optimization and save cost. On Apple Silicon, the optimized vision and text graphs measured about
119 MB and 254 MB respectively, adding about 373 MB of disk usage; sizes can vary by platform.
Already-running services reuse their loaded sessions.

Optimized graphs are tied to the source model, ONNX Runtime build and local hardware configuration.
If an optimized graph cannot be used or saved, VisionGrep falls back to the original model.
Removing `models/optimized/` causes the graphs to be rebuilt on subsequent use.
`--no-cache` bypasses the SQLite index, while model artifacts and optimized graphs remain cached.

## Options

```sh
visiongrep "sunset" ./photos --top 10 --threshold 0.3
visiongrep "sunset" ./photos --json
visiongrep "sunset" ./photos --paths-only
visiongrep "sunset" ./photos --index-path ./photos.db
visiongrep "sunset" ./photos --no-cache
visiongrep --help
```

The default is up to five results with a minimum cosine similarity of `0.25`. Scores are
similarities, not probabilities. Results go to stdout; progress and diagnostics go to stderr.
Use `--null` for exact paths separated by NUL bytes, or `--quiet` to suppress progress.

Exit codes are `0` for matches, `1` for no matches, and `2` for an operational or argument error.

## Prepare a folder ahead of time

```sh
visiongrep index ./photos
```

`index` downloads any missing models, embeds new or changed images, and prepares both models'
optimized graphs, without running a query. Afterwards the first search only needs to embed its
query. It accepts `--index-path PATH` and `--quiet`, exits `0` on success and `2` on error, and
writes nothing to stdout. Run it again at any time; unchanged images are skipped.

Indexing saves its progress every 256 images, so an interrupted `index` or search resumes where
it stopped when run again. An interrupted model download restarts that file. When stderr is not a
terminal, progress is reported as `Indexed N/TOTAL images` lines instead of a progress bar.

## Use with AI agents

Coding agents with a shell, such as Claude Code or Codex, can call `visiongrep` directly. The
first use of a folder is the slow part: it downloads about 375 MB of models and embeds every image,
which can take minutes for a few thousand photos and exceed an agent's command timeout. Run
`visiongrep index DIR` once yourself before handing the folder to an agent, or let the agent run
it with a long timeout. After that, a new query takes well under a second and a repeated query
takes a few milliseconds. If an agent's command is cut off while indexing, running it again
continues from the saved progress.

For many queries against one folder, start `visiongrep serve --socket DIR` in the background.
Ordinary searches then use the loaded models automatically, with no change to the command.

Output for agents:

- `--json` prints an array of `{"path": ..., "score": ...}` objects, highest score first.
- `--paths-only` prints one path per line; `-0` separates exact paths with NUL bytes.
- Exit code `1` means the search worked and nothing passed the threshold. The default threshold is
  deliberately conservative, so an empty result is more likely than an unrelated guess. Use
  `--threshold -1` to always get the top results.
- Progress, warnings and errors go to stderr only; `--quiet` suppresses progress.

Concurrent searches can share an index. Do not run `--reindex` while other searches use the same
index (see the known limitation below).

A snippet to adapt for `AGENTS.md` or `CLAUDE.md`:

```markdown
To find images by content, use `visiongrep "<description>" <dir> --json --top 10`, or
`visiongrep --image <file> <dir> --json` for similar images. Exit code 1 means no image matched;
retry with a different description or `--threshold 0.2` before concluding nothing matches.
Scores are raw cosine similarities (around 0.25–0.35 for good matches), not probabilities.
Never pass `--reindex`.
```

## Persistent socket server

Start an opt-in server for a search directory to reuse its loaded models across ordinary CLI
searches:

```sh
visiongrep serve --socket ./photos &
visiongrep "a dog on a beach" ./photos
```

The server runs in the foreground until stopped. Searches use it automatically when the search
directory and resolved index path match; otherwise they run as normal one-shot searches. Pass the
same `--index-path` to the server and searches when using a custom index. The server loads models
as needed and keeps them in memory (about 400–600 MiB with both models loaded). `--no-cache`,
`--reindex`, `--verify-models`, and `--timing` always run one-shot searches. `--reindex` refuses to
run while a socket server is listening for the same root and index; stop the server first.

## Persistent stdio searches

```sh
visiongrep serve --stdio --index-path photos.db ./photos
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
directory, and JSON requires UTF-8 paths. To search for the literal word `serve` or `index` in
one-shot mode, place a search option first: `visiongrep --top 5 serve ./photos`.

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

## Releasing

Releases are built by [dist](https://opensource.axo.dev/cargo-dist/) from
`dist-workspace.toml`. Bump the version in `Cargo.toml`, update `CHANGELOG.md`, merge to `main`,
then push a matching tag such as `v0.4.1`. The release workflow builds the three target archives,
the shell installer and checksums, and publishes them as a GitHub release. Run `dist generate`
after changing the dist configuration or upgrading dist. The crate is not published to crates.io.
