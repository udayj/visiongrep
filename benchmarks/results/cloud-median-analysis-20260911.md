# Cloud recording: typical-latency reanalysis

Run: `20260911-170542-99ffaccc`. Application reference: `4c8c613c8af16591860252683d29f4f0edaee723` (`v0.2.0`).

Raw report SHA-256: `f2850426d89e5af639a1abd7605ba4a87d5877853e6f69249807ca43f6bb2ab4`.

The original verdict remains `invalid`. A separate current-method analysis returns `foundation_recorded`: this session can contribute to a foundation, but does not alone establish one.

Only `persistent_text` exceeded the former 10% raw-CV cutoff. Five of its 21 process sequences had warm medians around 35–39 ms; most were around 29–30 ms. Slower sequences include short clusters, rather than a sustained shift across the run. Their cached control requests remained around 4.2–4.5 ms. The variation is localized to warm text-query work, but internal serve phase timings and CPU scheduling/frequency telemetry were not recorded; a physical cause cannot be established from this evidence.

All 336 timing samples (16 scenarios × 21 sequences/invocations) and all 440 quality queries were retained. Behavior checks passed. Each persistent process sequence contributes one sample, not five requests. Original results, resource measurements, and quality data are unchanged.

| Scenario | Median ms | Approx. 95% median interval ms | Raw CV | Max three-window drift |
|---|---:|---:|---:|---:|
| added_1pct | 1057.284 | 1052.081–1059.866 | 0.94% | 0.36% |
| cached_text | 9.578 | 9.559–9.665 | 1.05% | 0.92% |
| deleted_1pct | 13.962 | 13.906–14.160 | 3.65% | 2.89% |
| external_image_first | 923.943 | 922.000–927.597 | 0.86% | 0.38% |
| external_image_repeated | 920.588 | 918.470–926.786 | 0.96% | 0.67% |
| index_absent | 17331.922 | 17287.366–17346.972 | 0.44% | 0.22% |
| indexed_image | 9.617 | 9.590–9.659 | 1.17% | 0.40% |
| modified_1pct | 1052.550 | 1048.801–1056.446 | 0.76% | 0.70% |
| modified_query_image | 930.562 | 920.163–934.703 | 1.48% | 1.18% |
| no_cache | 17272.849 | 17253.361–17287.489 | 0.21% | 0.10% |
| novel_text | 764.493 | 762.835–768.129 | 0.87% | 0.53% |
| persistent_text | 29.769 | 29.653–30.141 | 10.58% | 0.99% |
| persistent_updates | 180.272 | 177.194–181.639 | 1.74% | 1.91% |
| read_only | 9.633 | 9.566–9.696 | 1.02% | 0.29% |
| reindex | 17383.720 | 17324.042–17488.654 | 1.21% | 0.60% |
| renamed_1pct | 1061.394 | 1057.626–1063.159 | 1.02% | 0.27% |

The shared method resamples consecutive triples of raw samples, uses three chronological windows for sustained drift, and retains session boundaries when combining instances. The inherited 10% precision and 15% drift tolerances were not changed to fit this recording. CV, MAD, and p95 remain diagnostics. These are approximate bootstrap intervals with limited evidence about long-range dependence and rare tails; see the benchmark README for assumptions and comparison gates.

Two further complete recordings from distinct cloud instances under the same reference, profile, corpus, models, and build/environment conditions are needed for the minimum three-instance foundation. The first session does not need to be discarded or repeated. All three must satisfy the joint precision and stability checks; three sessions are a minimum, not a guarantee. No new cloud measurements were launched during this analysis.

The existing three local v0.2.0 recordings were also recalculated under this method and remain calibrated. Older derived outputs and all raw recordings remain unchanged.
