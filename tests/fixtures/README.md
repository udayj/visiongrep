# CLIP text golden vectors

`clip_text_golden.json` protects the text-model contract independently of the Rust implementation.
It was generated with Python ONNX Runtime 1.22.1 and `tokenizers` 0.22.2 from the exact model and
tokenizer SHA-256 values recorded in the fixture. Padding follows Qdrant/FastEmbed: fixed length 77,
pad ID 1, and pad token `<|endoftext|>`.

The cases cover a one-word query, a typical sentence, a query just below the token limit, a
truncated query, and Unicode text. Each reference output is L2-normalized before serialization.
Vectors are stored as little-endian `f32` bytes encoded in hexadecimal to keep the fixture compact
and preserve the reference bits without decimal formatting noise.

The Rust integration test is ignored by default because it requires the pinned 250 MB text model
in VisionGrep's cache. Run it explicitly with:

```text
cargo test model::runtime::tests::text_embeddings_match_qdrant_fastembed_golden_vectors -- --ignored --exact
```
