# Cloud acceptance review

Keep all three recordings and the local foundation. Do not change acceptance thresholds or promote the cloud aggregate from these data alone.

## Existing evidence

Recalculation with all 21 samples per run gives:

| Scenario | Run | Median uncertainty | Maximum batch deviation |
|---|---|---:|---:|
| Deletion | 1 | 1.42% | 2.89% |
| Deletion | 2 | 1.56% | 1.57% |
| Deletion | 3 | 1.40% | 2.09% |
| Persistent text | 1 | 1.25% | 0.99% |
| Persistent text | 2 | 8.73% | 2.26% |
| Persistent text | 3 | 0.42% | 0.18% |

Every individual session passes. The aggregate fails because absolute levels differ between machines. A persistent sequence remains one sample.

An older cloud same-code test exists at `/Users/uday/.cache/visiongrep-bench/runs/20260907-132908-74309f94/remote/report.json`. It used commit `8b518ed86ff4f29e9c071931c91efa3e5ec31b44`, 14 scenarios, and three pairs per scenario. It has no persistent sequences. Recalculation gives `inconclusive`: the current method needs more samples. Its original success label cannot validate the present method. Raw reports were not rewritten.

## Method assessment

Absolute latency across machines and matched improvement within a machine are different quantities. A future revision should report them separately. This does not require two foundations or multiple policy implementations.

The harness already alternates reference/candidate order and resamples aligned pairs in blocks. Controlling shared conditions has an independent statistical basis: [NIST blocking guidance](https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm). Alternating order is not random allocation and cannot exclude every periodic effect.

A paired interval is conditional on the sampled environment. Reference-only recordings cannot establish how a candidate responds to different storage conditions. A shared additive delay does not cancel from a percentage improvement. For example, 100 ms versus 90 ms gives 10%. Adding 100 ms to both gives 5%. The current paired estimator reproduces this arithmetic. This is a constructed example, not a measurement.

Conversely, correlated variation can leave a paired effect precise while individual timings vary. Thus, hard gates on each role's absolute precision can be unnecessarily restrictive. This warrants a review of the gates, but does not justify removing environmental protection from the current evidence alone.

## Focused diagnostic proposal

No diagnostic was launched.

Use foundation commit `4c8c613c8af16591860252683d29f4f0edaee723` in both comparison positions. Keep the same 500-image corpus and storage configuration. Test deletion, persistent text, and cached text. Use a fixed 21 matched pairs per scenario with balanced order. Keep correctness checks and every sample. This budget does not guarantee a precise result.

Measure the unchanged product binary. In a separate traced pass, record database writes and synchronization calls with their duration. Associate persistent writes with request boundaries. Trace overhead must not enter qualification measurements. Existing one-shot phases already locate the deletion and write operations. No cache feature is needed.

A useful same-code result requires its paired confidence interval to lie within the existing ±5% material-change margin, plus passing correctness and within-run drift checks. A point estimate inside this margin is insufficient. The current validate mode checks point estimates; its success label must not substitute for an interval-based equivalence check.

A failed or uncertain test is not permission to discard samples or repeat until it passes. Inspect the trace and order effects first. A successful test supports the comparison mechanism on that machine. It does not prove that candidate improvements generalize across machines. These data do not establish an exact number of further runs.

## Outcome

No acceptance code or thresholds changed. No new cloud runs. No local foundation changes. The cloud aggregate remains inconclusive. The existing recordings remain useful descriptive reference data. The previous analysis and collection edits remain in the working tree.
