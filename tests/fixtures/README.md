# CLIP golden vectors

`datacomp_golden.json` is the active DataComp contract fixture. It was generated from the pinned
original OpenCLIP safetensors with `open_clip_torch` 3.3.0, then checked against the pinned rclip
ONNX conversion. It contains exact token IDs plus normalized text and image embeddings. The image
cases cover landscape, portrait, and square geometry, and the queries include ordinary, long-form,
screenshot, and Unicode text. Contract revisions and checksums are embedded in the fixture.

Vectors are stored as little-endian `f32` bytes encoded in hexadecimal to keep the fixture compact
and preserve the reference bits without decimal formatting noise.

The ONNX-backed Rust tests are ignored by default because they require the pinned model artifacts in
VisionGrep's cache. The manually triggered model-contract workflow runs all of them. In an isolated
remote environment with the artifacts installed, run:

```text
cargo test --release text_embeddings_match_openclip_golden_vectors -- --ignored
cargo test --release image_embeddings_match_openclip_golden_vectors -- --ignored
cargo test --release cosine_scores_rankings_and_thresholds_match_openclip -- --ignored
cargo test --release batched_and_single_image_inference_match -- --ignored
cargo test --release vision_model_contract_supports_dynamic_batches -- --ignored
```

The image-query CLI integration test needs only the pinned vision model under
`$XDG_CACHE_HOME/visiongrep/models/datacomp_vision.onnx`. It creates an isolated cache with no text
artifacts and checks external queries, indexed-query reuse, changed files, no-cache searches, and
reindex atomicity:

```text
cargo test --release --test image_queries image_query_end_to_end -- --ignored
```

`clip_text_golden.json` is retained only as provenance for the pre-DataComp Qdrant baseline used by
the comparative benchmark. It is not the current product contract.
