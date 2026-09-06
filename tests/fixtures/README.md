# CLIP golden vectors

For the text/image CLI scenario matrix, cache lifecycle tests, and first-download test, see
[the end-to-end test guide](../README.md).

`datacomp_golden.json` is the active DataComp contract fixture. It was generated from the pinned
original OpenCLIP safetensors with `open_clip_torch` 3.3.0, then checked against the pinned rclip
ONNX conversion. It contains exact token IDs plus normalized text and image embeddings. The image
cases cover landscape, portrait, and square geometry, and the queries include ordinary, long-form,
screenshot, and Unicode text. Contract revisions and checksums are embedded in the fixture.

Vectors are stored as little-endian `f32` bytes encoded in hexadecimal to keep the fixture compact
and preserve the reference bits without decimal formatting noise.

The five ONNX-backed correctness tests run by default locally and in push/PR CI. They reuse the
pinned model artifacts installed by the preparation step and never initiate downloads. The
manually triggered model-contract workflow additionally validates the original reference models.
With the artifacts installed, run individual correctness tests with:

```text
cargo test --release text_embeddings_match_openclip_golden_vectors
cargo test --release image_embeddings_match_openclip_golden_vectors
cargo test --release cosine_scores_rankings_and_thresholds_match_openclip
cargo test --release batched_and_single_image_inference_match
cargo test --release vision_model_contract_supports_dynamic_batches
```

The image-query CLI integration test runs by default and needs only the pinned vision model under
`$XDG_CACHE_HOME/visiongrep/models/datacomp_vision.onnx` (or `~/.cache/visiongrep/models/` when
`XDG_CACHE_HOME` is unset). It creates an isolated cache with no text
artifacts and checks external queries, indexed-query reuse, changed files, no-cache searches, and
reindex atomicity:

```text
cargo test --release --test image_queries
```

`clip_text_golden.json` is retained only as provenance for the pre-DataComp Qdrant baseline used by
the comparative benchmark. It is not the current product contract.
