# visiongrep — spec v0.1

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
│   ├── ranking.rs       # cosine similarity, thresholding, and top-K ranking
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
└── README.md
```

---

## Dependencies

```toml
[dependencies]
ort = "2"                        # ONNX Runtime bindings
image = "0.25"                   # image decoding (JPEG, PNG, WEBP, BMP)
tokenizers = "0.19"              # CLIP text tokenizer (HuggingFace)
ndarray = "0.15"                 # tensor ops
rusqlite = { version = "0.31", features = ["bundled"] }
clap = { version = "4", features = ["derive"] }
thiserror = "2"
indicatif = "0.17"               # progress bars
reqwest = { version = "0.12", features = ["blocking", "rustls-tls"], default-features = false }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
sha2 = "0.10"                   # artifact checksums
tempfile = "3"                  # unique same-directory artifact downloads
walkdir = "2"                    # recursive directory traversal
```

No `torch`. No `transformers`. No Python interop.

---

## Models

Two pinned ONNX files, downloaded on first run:

| File | Source | Size |
|---|---|---|
| `clip_vision.onnx` | `Qdrant/clip-ViT-B-32-vision` on HuggingFace | ~352 MB |
| `clip_text.onnx` | `Qdrant/clip-ViT-B-32-text` on HuggingFace | ~254 MB |

Download destination: `~/.cache/visiongrep/models/`

The model URLs use immutable Hugging Face revisions. Both models and the matching tokenizer are
verified with pinned SHA-256 checksums before atomic installation. If the cache directory already
contains valid files, skip download silently.

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

`--reindex` keeps the previous cache until model artifacts and the vision session are usable, then
clears and rebuilds its contents. Image updates are written in bounded transactions.
`--no-cache` skips reading and writing entirely — embeds everything fresh each run. Useful for scripting against small folders.
The two flags are mutually exclusive.

---

## Core logic

### Image embedding

1. Read dimensions before allocating the decoded image; skip zero-sized images, images over 100 MP,
   or images whose estimated peak working memory exceeds 512 MiB
2. Decode with `image` and apply EXIF orientation
3. Convert to RGB
4. Take the centered square crop without stretching the image
5. Resize the square to 224×224 with Catmull-Rom bicubic filtering
6. Normalize with CLIP mean `[0.48145466, 0.4578275, 0.40821073]` and std `[0.26862954, 0.26130258, 0.27577711]`
7. Run through `clip_vision.onnx` — input shape `[1, 3, 224, 224]`, output shape `[1, 512]`
8. L2-normalize the output vector

The crop-first implementation is geometrically equivalent to the model's resize-short-edge then
center-crop policy for ordinary images while avoiding a potentially enormous intermediate image for
pathological panorama aspect ratios. Tiny images are upscaled; portrait and landscape images retain
their geometry.

### Text embedding

1. Tokenize query using the matching downloaded `tokenizer.json` (BPE, max 77 tokens, pad/truncate)
2. Run through `clip_text.onnx` — input `input_ids [1, 77]` and `attention_mask [1, 77]`, output `[1, 512]`
3. L2-normalize the output vector

### Search

Cosine similarity = dot product of two L2-normalized vectors (just a dot product, since both are unit vectors).

Rank all indexed images by similarity to query vector, return up to N above threshold. If none meet
the threshold, emit no result rows and exit 1.

Text embedding runs once for a novel query and is reused for exact repeated queries. Image decoding,
preprocessing, and inference are currently sequential; batching or parallel preprocessing should be
added only after representative indexing measurements identify it as the bottleneck.

---

## First-run experience

```
$ visiongrep "sunset over water" ./photos/

Downloading CLIP vision model (352 MB)...
[████████████████████░░░░░░░░░░] 312/352 MB  4.2 MB/s  eta 9s

Downloading CLIP text model (254 MB)...
[██████████████████████████████] 254/254 MB  done

Indexing 1,432 images...
[██████████████████████████████] 1432/1432  done  (43s)

score  path
0.334  ./photos/2023/mallorca_sunset.jpg
0.291  ./photos/landscapes/dusk_lake.png
```

On subsequent runs against the same directory, indexing is skipped (only new images are embedded) and the model is already cached. Warm-cache time to results should be measured and reported once representative image folders are available.

---

## Benchmark reporting plan

v0.1 should report benchmark numbers once representative image folders are available.
Do not treat these as guaranteed targets before measurement; they are the metrics that
should appear in the README and launch post:

- Warm cached search over 1,000 images
- Warm cached search over 10,000 images
- Startup overhead before model/query work
- Index throughput in images/sec on representative datasets
- Incremental re-run against an unchanged folder, confirming no image re-embedding
- Repeated-query time on an unchanged folder, confirming no ONNX session is loaded
- No-match precision over deliberately unrelated queries at the default threshold

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

- Unsupported image format: skip silently, warn with filename if not `--quiet`
- Corrupt image: skip silently, warn
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
