# visiongrep — retrieval foundation specification

## What it is

A CLI tool that searches a folder of images using natural language. You describe what you're looking for; it returns ranked matches. CLIP zero-shot similarity via ONNX Runtime. Single Rust binary, no Python, no GPU required.

visiongrep is a Rust-native visual grep for local folders, scripts, and AI agents.
It is designed for repeated non-interactive use: fast startup, warm-cache search,
stable machine-readable output, deterministic exit codes, and no Python/PyTorch
runtime.

```
visiongrep "red car parked near a building" ./photos/
```

---

## Non-goals for v0.1

- No video support
- No GUI
- No server/daemon mode
- No image-to-image search
- No re-ranking or multi-modal fusion
- No Windows support (Linux + macOS only for now)

---

## Project layout

```
visiongrep/
├── Cargo.toml
├── AGENTS.md
├── src/
│   ├── main.rs          # lightweight process boundary and exit codes
│   ├── application.rs   # cached/no-cache search orchestration
│   ├── embedding.rs     # image preprocessing and embedding normalization
│   ├── error.rs         # VisionGrepError enum (thiserror)
│   ├── pillow_resize.rs # Pillow-compatible bicubic resize contract
│   ├── ranking.rs       # cosine similarity and deterministic bounded top-K
│   ├── timing.rs        # opt-in machine-readable phase measurements
│   ├── cli/
│   │   ├── mod.rs       # CLI subsystem facade
│   │   ├── args.rs      # clap parsing and typed command construction
│   │   └── terminal.rs  # result output and progress-event presentation
│   ├── index/
│   │   ├── mod.rs       # index subsystem facade
│   │   ├── scan.rs      # recursive image discovery and metadata snapshots
│   │   ├── store.rs     # SQLite schema, cache, and persisted vectors
│   │   └── ingest.rs    # batch embedding and typed progress events
│   └── model/
│       ├── mod.rs       # model subsystem facade
│       ├── artifacts.rs # model installation, verification, and transfer events
│       └── runtime.rs   # tokenizer and ONNX session/inference ownership
├── benchmarks/          # isolated Python/reference and performance tooling
└── THIRD_PARTY_NOTICES.md
```

---

## Dependencies

```toml
[dependencies]
ort = "=2.0.0-rc.12"             # ONNX Runtime 1.24 bindings
image = "0.25"                   # image decoding (JPEG, PNG, WEBP, BMP)
tokenizers = "0.22"              # OpenCLIP text tokenizer (HuggingFace)
ndarray = "0.17"                 # tensor assembly
rusqlite = { version = "0.39", features = ["bundled"] }
clap = { version = "4", features = ["derive"] }
thiserror = "2"
indicatif = "0.18"               # progress bars
reqwest = { version = "0.12", features = ["blocking", "rustls-tls"], default-features = false }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
sha2 = "0.10"                   # artifact checksums
tempfile = "3"                  # unique same-directory downloads and staged indexes
walkdir = "2"                    # recursive directory traversal
```

No `torch`. No `transformers`. No Python interop.

---

## Models

Two pinned ONNX files and their exact tokenizer, downloaded on first run:

| File | Source | Size |
|---|---|---|
| `datacomp_vision.onnx` | rclip conversion of `laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K` | 351,826,068 bytes |
| `datacomp_text.onnx` | matching rclip text encoder conversion | 254,344,274 bytes |
| `datacomp_tokenizer.json` | pinned original OpenCLIP tokenizer contract | 2,224,081 bytes |

Download destination: `~/.cache/visiongrep/models/`

The model URLs use immutable Hugging Face revisions. Models, tokenizer, source OpenCLIP weights,
conversion revisions, exact sizes, and SHA-256 values are recorded in
`benchmarks/retrieval/benchmark_manifest.json`. Downloaded artifacts are hashed while streaming to
a sibling temporary file and atomically installed only after size and checksum validation.

Each installed artifact has a versioned verified-install manifest tied to the immutable revision,
expected size/checksum, and local file identity. Normal loading trusts a matching marker without
re-reading hundreds of megabytes; `--verify-models` forces a complete hash. Missing, stale,
malformed, or mismatched markers fall back to full verification and can never make a partial
download valid. Licenses and attribution are in `THIRD_PARTY_NOTICES.md` and `third_party/`.

The matching `tokenizer.json` is downloaded alongside the text model. Model artifacts live in
`~/.cache/visiongrep/models/`, outside the searched folder.

---

## CLI interface

```
visiongrep [OPTIONS] <QUERY> <PATH>

Arguments:
  <QUERY>   Natural language description of what to find
  <PATH>    Directory to search (searched recursively)

Options:
  -n, --top <N>          Number of results to return [default: 5]
  -t, --threshold <F>    Minimum raw CLIP cosine similarity -1.0–1.0 [default: 0.25]
  --json                 Output results as JSON
  --paths-only           Output only matching paths, one per line
  -0, --null             Output exact paths separated by NUL bytes
  --reindex              Force re-embedding of all images, ignoring cache
  --no-cache             Skip reading and writing the index cache
  --index-path <PATH>    Use an explicit index outside the searched tree
  --verify-models        Fully hash every model artifact needed by this search
  --timing               Emit one phase-timing JSON document to stderr
  --timing-file <PATH>   Write --timing JSON to PATH instead
  -q, --quiet            Suppress progress output (useful in scripts)
  -h, --help             Print help
  -V, --version          Print version
```

### Default output (human-readable)

```
score  path
0.312  /photos/street/img_0042.jpg
0.287  /photos/travel/rome_01.png
0.261  /photos/misc/scan003.jpg
```

Tab-separated, score to 3 decimal places, sorted descending. Tabs, newlines, backslashes, and
control characters in paths are escaped. No header when `--quiet` is set.
`--json` and `--paths-only` are mutually exclusive. If both are passed, return a CLI error.

### JSON output (`--json`)

```json
[
  {"score": 0.312, "path": "/photos/street/img_0042.jpg"},
  {"score": 0.287, "path": "/photos/travel/rome_01.png"}
]
```

JSON requires UTF-8 paths. A non-UTF-8 result path produces a clear operational error rather than
lossy output.

### Paths-only output (`--paths-only`)

```
/photos/street/img_0042.jpg
/photos/travel/rome_01.png
```

Use `-0` / `--null` for exact native path bytes separated by NUL. This is the safe format for paths
that may contain newlines or other record-separator characters.

Exit code 0 if any results found above threshold, 1 if no matches, 2 on error.

Scores are raw cosine similarities, not probabilities or confidence percentages. The default
threshold is deliberately conservative for non-interactive and agent use, where an empty result is
preferable to an unrelated guess. It is provisional until calibrated against a representative
positive/negative query set. Pass `--threshold 0` to restore the original non-negative behavior,
or `--threshold -1` to force an exhaustive top-N result.

---

## Indexing and cache

On first run against a directory, visiongrep embeds every image and stores the result in a SQLite
database at `<PATH>/.visiongrep.db`. Subsequent runs load embeddings from cache and only process new
or modified files (detected by nanosecond mtime + file size). Deleted or renamed paths are removed.

Schema (keep it minimal):

```sql
CREATE TABLE images (
  path       BLOB PRIMARY KEY, -- root-relative native Unix path bytes
  mtime_ns   INTEGER NOT NULL,
  size       INTEGER NOT NULL,
  embedding  BLOB NOT NULL CHECK(length(embedding) = 2048) -- 512 little-endian f32 values
);

CREATE TABLE queries (
  query      TEXT PRIMARY KEY,
  embedding  BLOB NOT NULL CHECK(length(embedding) = 2048)
);
```

Embeddings are stored as explicit little-endian bytes. No compression. No external vector DB.

SQLite `user_version` identifies the embedding contract. Any model, tokenizer, output, or image
preprocessing change that could alter vectors must increment this version. Older incompatible
caches are cleared and rebuilt automatically; newer cache versions fail safely instead of being
silently misread.

Exact query embeddings are cached in the same database. On an unchanged folder, a repeated query
loads neither ONNX model: the command reads cached image/query vectors and performs dot products.
Novel queries load only the text model. New or changed images load only the vision model unless the
query is also novel.

The schema also has a `metadata(key TEXT PRIMARY KEY, value BLOB NOT NULL)` table. Schema version 3
stores the canonical native-byte search root plus separate image-embedding and query-embedding
contracts. A wrong-root custom index is rejected before mutation. Image-contract changes clear
image and query vectors; query-only contract changes preserve image vectors.

`--reindex` builds and verifies a unique sibling database using bounded write transactions, then
atomically replaces the active index. A failed rebuild leaves the previous image and query caches
untouched, and concurrent readers see either the previous complete index or the replacement.
`--no-cache` skips reading and writing entirely — embeds everything fresh each run. Useful for scripting against small folders.
The two flags are mutually exclusive.

`--index-path` resolves relative paths from the process working directory, leaves the image tree
untouched, and supports cached search over a read-only tree. Reindex temporary files are created
beside the selected destination for a same-filesystem atomic rename. Back up the SQLite file only
while no writer is active; the index is deliberately bound to one canonical root and is not a
portable image bundle. SQLite's five-second busy timeout makes concurrent-writer failure explicit.

Incremental reconciliation loads cached `(path, mtime_ns, size)` rows in one ordered query and
merge-walks them against byte-sorted discovery results. Stale deletions and embedding updates are
transactional. Native Unix path bytes, nanosecond mtime, size semantics, and deterministic ordering
are preserved.

---

## Core logic

### Image embedding

1. Read dimensions before allocating the decoded image; skip zero-sized images, images over 100 MP,
   or images whose estimated peak working memory exceeds 512 MiB
2. Decode with `image` and apply EXIF orientation
3. Convert to RGB
4. Resize the short edge to 256 with Pillow 12.3-compatible bicubic coefficients and rounding
5. Take the centered 256×256 crop using Python round-half-to-even crop coordinates
6. Normalize with CLIP mean `[0.48145466, 0.4578275, 0.40821073]` and std `[0.26862954, 0.26130258, 0.27577711]`
7. Run through `datacomp_vision.onnx` — dynamic input `[N, 3, 256, 256]`, output `[N, 512]`
8. L2-normalize the output vector

Tiny images are upscaled; portrait and landscape images retain their geometry. The pre-allocation
working-memory check includes decode, orientation, RGB conversion, resized image, crop, and `f32`
tensor buffers.

### Text embedding

1. Tokenize with the matching OpenCLIP tokenizer: include special tokens, truncate to 77 positions,
   and right-pad to 77 with ID `0` and token `<|endoftext|>`
2. Run through `datacomp_text.onnx` — one `input_ids [1, 77]` input and output `[1, 512]`
3. L2-normalize the output vector

### Search

Cosine similarity = dot product of two L2-normalized vectors (just a dot product, since both are unit vectors).

Use a bounded deterministic heap to select up to N images above threshold, then return them in
score-descending/path-ascending order. If none meet
the threshold, emit no result rows and exit 1.

Text embedding runs once for a novel query and is reused for exact repeated queries. Indexing uses
up to four scoped preprocessing workers, an atomic work index, ordered result restoration, batches
of eight images, and one inference consumer. Memory is bounded by the 256-image persistence chunk,
eight-image inference batches, and four preprocessing workers. Recoverable decode/resource failures
produce explicit skip events; worker, inference, cardinality, runtime, and database failures abort.

---

## First-run experience

```
$ visiongrep "sunset over water" ./photos/

Downloading DataComp CLIP vision model (352 MB)...
[████████████████████░░░░░░░░░░] 312/352 MB  4.2 MB/s  eta 9s

Downloading DataComp CLIP text model (254 MB)...
[██████████████████████████████] 254/254 MB  done

Indexing 1,432 images...
[██████████████████████████████] 1432/1432  done  (43s)

score  path
0.334  ./photos/2023/mallorca_sunset.jpg
0.291  ./photos/landscapes/dusk_lake.png
```

On subsequent runs against the same directory, indexing is skipped (only new images are embedded) and the model is already cached. Warm-cache time to results should be measured and reported once representative image folders are available.

---

## Benchmark reporting

`benchmarks/README.md` specifies the pinned COCO/reproducible-fixture corpus, rclip v3.3.0 adapter,
metrics, phase scenarios, cache-state labels, licensing, and exact reproduction commands. Small
aggregate evidence is stored in `benchmarks/results/remote_foundation.json`; raw vectors, per-query
logs, models, indexes, and dataset images remain runner artifacts or temporary files.

Normal CI does not download models. Manually triggered model-contract CI verifies OpenCLIP, ONNX,
tokenizer, golden vectors, scores, rankings, and batching. Heavy retrieval/timing and real Apple
Silicon Core ML experiments are also manual workflows with read-only repository permissions.

---

## Install/runtime footprint reporting

Release notes and README benchmarks should report:

- Release binary size
- Fresh install size excluding model weights
- First-run model download size
- Disk usage of `.visiongrep.db` for 1k / 10k / 50k images
- Cold start time
- Warm-cache query time

---

## Error handling

- Unsupported extensions are excluded during discovery
- Corrupt, unreadable, oversized, or invalid-dimension images: emit an explicit skip event and
  warning unless `--quiet`; remove any stale cached record
- Download failure: exit with clear message, suggest checking connection
- Checksum mismatch: delete partial file, exit with message
- Permission error on cache dir: exit with a typed error; never silently change persistence behavior
- ORT session failure: exit with error message and suggest filing an issue
- Downstream pipe closure: terminate normally without printing an operational error

No panics in release builds. All errors use a single `VisionGrepError` enum in `error.rs`:

```rust
#[derive(Debug, thiserror::Error)]
pub enum VisionGrepError {
    #[error("Failed to download model from {url}: {source}")]
    DownloadRequest {
        url: String,
        #[source]
        source: reqwest::Error,
    },

    #[error("Checksum mismatch for {file}: expected {expected}, got {actual}")]
    Checksum { file: String, expected: String, actual: String },

    #[error("Failed to decode image {path}: {source}")]
    ImageDecode { path: PathBuf, #[source] source: image::ImageError },

    #[error("ONNX inference error: {0}")]
    Inference(#[from] ort::Error),

    #[error("Index error: {0}")]
    Index(#[from] rusqlite::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Failed to load tokenizer: {source}")]
    TokenizerLoad {
        #[source]
        source: tokenizers::Error,
    },

    #[error("Failed to encode query with tokenizer: {source}")]
    TokenizerEncode {
        query: String,
        #[source]
        source: tokenizers::Error,
    },
}
```

Internal fallible operations return `Result<T, VisionGrepError>`. `application.rs` coordinates
subsystems and forwards typed index/model events without depending on terminal presentation.
`cli/terminal.rs` owns stdout/stderr and progress rendering, while the lightweight `main.rs` process
boundary maps matches, no matches, and operational failures to exit codes. No `unwrap()` or
`expect()` appears outside `#[cfg(test)]` blocks.

---

## What success looks like for v0.1

- `cargo build --release` produces a single binary
- Binary runs on Linux x86_64 and macOS arm64 without any additional installs
- First run downloads models, indexes a folder of 1000 images, returns results
- Subsequent runs against the same folder reuse cached image embeddings
- Repeated exact queries reuse cached text embeddings without loading ONNX
- Unrelated queries can return no results and exit 1 under the default threshold
- `--json` output is stable and parseable
- `--paths-only` output is stable and script-friendly
- Zero unsafe code outside of the ORT bindings (which are already unsafe internally)

---

## Out of scope until explicitly requested

- `--model` flag for alternate models
- MCP server wrapper
- Windows build
- Video frame extraction
- Image-to-image search (`--image` flag instead of query text)
- Streaming/watch mode
- Config file
