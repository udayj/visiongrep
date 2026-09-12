# Cloud v0.2.0 recordings: collection and aggregation

All three recordings were collected successfully on 2026-09-12. No new cloud measurements were launched. Source report hashes match the aggregate provenance; raw recordings were not rewritten.

Aggregate: `/Users/uday/.cache/visiongrep-bench/cloud-foundation-v0.2.0-median.json`

**Verdict: inconclusive; not yet a usable calibrated foundation.** All behavior checks passed. Fourteen of sixteen scenarios meet the current precision and drift criteria. Deletion latency has a sustained difference between instances; persistent text has insufficient precision across instances. No samples were excluded and no thresholds were changed.

The corpus contains 500 images. Each scenario has 21 samples on each of three distinct instances (63 total); 1,008 scenario samples overall. Each persistent sequence counts as one sample. Estimates give equal weight to session medians. Intervals are approximate 95% hierarchical block-bootstrap intervals.

| Scenario | Median ms | Approx. 95% interval ms | Session medians ms (runs 1 / 2 / 3) | Median peak RSS MiB¹ | Status |
|---|---:|---|---|---:|---|
| added_1pct | 1050.04 | 1044.50–1058.73 | 1057.28 / 1050.04 / 1045.76 | 606.0 | Pass |
| cached_text | 9.58 | 9.47–9.79 | 9.58 / 9.73 / 9.49 | 14.3 | Pass |
| deleted_1pct | 14.92 | 13.94–18.74 | 13.96 / 14.92 / 18.72 | 14.3 | median uncertainty exceeds 10%; session median drift exceeds 15% |
| external_image_first | 923.40 | 918.29–927.11 | 923.94 / 923.40 / 919.13 | 599.5 | Pass |
| external_image_repeated | 920.59 | 916.26–930.31 | 920.59 / 924.76 / 917.62 | 599.5 | Pass |
| index_absent | 17312.41 | 17265.44–17354.69 | 17331.92 / 17312.41 / 17276.68 | 638.9 | Pass |
| indexed_image | 9.62 | 9.44–9.92 | 9.62 / 9.89 / 9.44 | 14.5 | Pass |
| modified_1pct | 1052.18 | 1043.13–1054.61 | 1052.55 / 1052.18 / 1044.08 | 605.4 | Pass |
| modified_query_image | 930.56 | 917.26–933.24 | 930.56 / 930.93 / 919.37 | 599.9 | Pass |
| no_cache | 17272.85 | 17217.61–17302.92 | 17272.85 / 17299.87 / 17220.41 | 637.2 | Pass |
| novel_text | 760.76 | 757.89–766.89 | 764.49 / 760.25 / 760.76 | 381.6 | Pass |
| persistent_text | 30.38 | 29.76–33.95 | 29.77 / 30.38 / 33.94 | 377.9 | median uncertainty exceeds 10% |
| persistent_updates | 180.27 | 177.49–181.59 | 180.27 / 178.34 / 181.29 | 924.6 | Pass |
| read_only | 9.63 | 9.47–9.85 | 9.63 / 9.84 / 9.47 | 14.1 | Pass |
| reindex | 17383.72 | 17292.26–17463.55 | 17383.72 / 17460.16 / 17293.37 | 639.3 | Pass |
| renamed_1pct | 1061.39 | 1053.14–1068.89 | 1061.39 / 1064.42 / 1054.99 | 606.0 | Pass |

¹ Median of the three session medians of per-process peak RSS; descriptive, not a regression comparison.

Deletion: median uncertainty 25.63%; maximum session deviation 25.51%, above the 15% drift limit. Its within-session batch deviation is only 2.89%, consistent with a sustained instance difference rather than occasional spikes.

Persistent text: median uncertainty 11.77%, above the 10% precision limit. Maximum session deviation is 11.73% (below the 15% drift limit); within-session batch deviation is 2.26%. The recordings do not identify a physical cause for the between-instance differences.

| Persistent diagnostic | Text sequence | Update sequence |
|---|---:|---:|
| First response (ms) | 719.51 | 720.89 |
| First update, vision session loaded lazily (ms) | — | 894.63 |
| Cached response (ms) | 4.36 | 4.35 |
| Index bytes | 2,109,440.00 | 2,097,152.00 |

Quality metrics are identical across all three recordings: 440 checks per recording (1,320 executions of the same fixed evaluation, not 1,320 independent queries). Recall@1 66.5%, Recall@5 88.5%, Recall@10 95.5%, MRR@10 0.7511, nDCG@10 0.8000. At threshold 0.25, annotation-defined absent false-positive rate is 39.58% and present false-negative rate is 9.58%. These are measured product quality levels, not claims of zero classification error.

Raw per-session CV, p95, and other spread diagnostics remain in the aggregate `raw_summaries`. These diagnostics do not alone reject the foundation.

Keep these recordings. Do not promote this aggregate to the calibrated foundation yet. Investigate the deletion path and instance conditions before deciding on further independent-instance measurements. More repetitions on an already sampled instance would not resolve the observed between-instance difference. No exact additional sample count is justified by these three instances alone.

Collection fix: missing EC2 records are represented explicitly; S3 synchronization can resume; missing or newer slot leases do not block repeat collection, and only the matching lease is conditionally deleted. EC2 lookup failures still propagate. All 97 benchmark tests passed.

## Investigation of the timing difference

The phase records identify database writes as the main affected operation.

| Median time (ms) | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| Complete deletion search | 13.962 | 14.916 | 18.721 |
| Delete stale database entries | 4.317 | 5.010 | 8.818 |
| Write a new text query to the database | 4.047 | 4.650 | 8.865 |
| Write added images to the database | 4.708 | 5.043 | 8.954 |
| Write modified images to the database | 4.396 | 4.985 | 8.933 |
| Persistent warm text sequence | 29.769 | 30.376 | 33.940 |
| Persistent cached response | 4.363 | 4.412 | 4.193 |

For deletion, the run 3 increase relative to run 2 is 3.805 ms for the complete search. The increase in the stale-entry phase is 3.808 ms. Separate medians do not form an exact accounting identity, but this close agreement locates the delay. Discovery, change detection, and scoring remain approximately unchanged. The stale-entry phase includes the SQLite DELETE transaction and its commit (`src/index/store.rs`, `apply_reconciliation`).

The same delay occurs in database writes for new text queries and changed images. This is evidence of a shared database-write cost, rather than a deletion-only code defect. Larger scenarios also incur the delay, but other work makes it a small percentage of their total time.

Each new persistent text request writes its embedding to SQLite before the response. A cached request skips both text inference and this write (`src/application/query.rs`, `prepare_text`). Run 3 has slower new requests but faster cached controls. Its extra database-write time in the one-shot phase records is consistent with the persistent difference. Persistent requests have no phase timings, so these recordings cannot separate their inference time from their write time directly.

The source commit, model files, corpus, CPU model, kernel, and configured storage type are the same. Storage is configured as gp3 with 3,000 IOPS and 125 MiB/s. Run 1 has an older recorded CPU microcode version. Runs 2 and 3 have the same microcode but different write times, so that difference alone cannot explain the result. The storage-quiet check occurs before each measured process. Its median sync time is about 7.7 ms in all three runs. It does not measure the latency of each later SQLite commit.

The evidence supports a difference in database-write latency between the cloud instances. It does not identify the underlying storage or operating-system cause. There are no per-commit system-call timings or block-device latency traces in these recordings. Do not claim that an EBS fault, CPU scheduling, or SQLite synchronization is proven.

The analysis correctly detects this difference. All three runs use the same product commit, so this is not evidence of a regression between product versions. Keep every sample. Do not change the thresholds to accept this aggregate. Before more foundation runs, add focused timing for SQLite commit/synchronization and separate inference from database writes in persistent requests. The next diagnostic measurement should test the identified write path. Repeating the full suite without these measurements can reproduce the difference without explaining its cause.
