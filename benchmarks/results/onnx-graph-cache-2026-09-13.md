# Optimized ONNX graph startup: Linux diagnostic

Run `20260913-144636-c52f7944` measured compatible, prebuilt graph reuse on one Linux
`c7a.xlarge` instance (AMD EPYC 9R14, 4 CPUs, 8 GiB), using ONNX Runtime 1.24.2 CPU
and a 500-image corpus. Typical startup latency improved by 6–10% on this configuration.

## Provenance and method

- Reference: `v0.2.0`, commit `4c8c613c8af16591860252683d29f4f0edaee723`.
- Candidate application: `651b4e4a01dd0c90421f9e68a074afb451899c2f`.
- Benchmark harness: `d65b89e247aa442bb9488f1ee3ecc1d025dff1e3`, mode `graph-cache`.
- Raw report: `runs/20260913-144636-c52f7944/report.json` in the configured private
  benchmark artifact bucket; collected locally as `RUN/remote/report.json`.
- Report SHA-256: `6bfb693c1b2af0b29301c3ae028bd05e3d6b384c57ea0489f9bfb930fc0fb99f`.

Each of four scenarios used 21 fixed reference/candidate pairs, alternating execution
order, with no sample exclusions or repeats. Both binaries ran on the same instance.
Downloads, compilation, indexing and graph creation occurred before timed samples.
Each binary's applicable model files were pre-read; filesystem cache residency was not
guaranteed. CPU idle/steal and storage-settling checks passed.

Improvement is `1 - median(candidate) / median(reference)`. Intervals are approximate,
pointwise 95% paired block-bootstrap intervals (5,000 draws, consecutive blocks of three).
The diagnostic materiality threshold was 5%.

## Latency results

| Endpoint | Reference median (ms) | Candidate median (ms) | Improvement | 95% interval |
|---|---:|---:|---:|---:|
| One-shot novel text | 764.57 | 715.41 | 6.43% | 6.33–7.35% |
| One-shot external image | 929.51 | 835.65 | 10.10% | 8.93–10.89% |
| Service first response | 721.40 | 665.24 | 7.79% | 7.37–8.52% |
| Service warm novel query | 31.10 | 31.03 | 0.24% | −0.06–2.33% |
| Service cached response | 4.24 | 4.18 | 1.35% | −0.13–3.05% |
| One-shot cached text | 9.49 | 9.48 | 0.03% | −0.82–0.50% |

Service first response includes process startup and the first request/response over
stdio. The warm metric is each process's median of three subsequent novel requests;
the cached control repeats the last request. All three startup gains remained above
5% in both execution-order groups and all chronological thirds. Warm novel requests
showed order sensitivity (16.67% versus 0.07% subgroup gains); no warm-query gain is
established. These intervals do not provide a joint confidence level across endpoints.

## Preparation, resources and correctness

Both optimized graphs were created before timing and remained unchanged, including
final SHA-256 verification. They added 605,872,980 bytes (577.8 MiB). Setup's combined
model-session phase took approximately 4.4 seconds for the candidate versus 1.14 seconds
for the reference, including source loading, optimization and serialization. This is
not a measurement of graph generation alone; an unprepared first use still pays it.

Median peak RSS stayed near 380 MiB for text/service, 600 MiB for external-image queries,
and 14 MiB for cached CLI queries. SQLite indexes remained near 2 MiB. Phase diagnostics
showed text session construction decreasing from 678.56 to 629.47 ms and vision session
construction from 820.32 to 731.81 ms. Text inference alone increased from 23.11 to
27.35 ms; individual phases were not separately statistically qualified.

All 168 sample records passed behavior checks. Paired rankings matched exactly; scores
and embedding components agreed within `1e-4`, with normalized embeddings and matching
index contents. No output-equivalence regression was detected. The full retrieval-quality
suite was not run, so this does not establish unchanged aggregate recall or nDCG.

## Qualification status

This is a completed single-instance diagnostic, not a calibrated cloud foundation or a
full cloud-standard comparison. It does not replace or qualify the pending v0.2.0 cloud
foundation. Selecting this implementation as a future reference requires separate full
recordings and foundation calibration; the existing qualification gates are unchanged.
