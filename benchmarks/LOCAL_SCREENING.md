# Local screening policy v2

This changes the fixed budget and distinguishes timing uncertainty from failed execution;
it does not widen tolerances. Local defaults use nine samples per scenario, making three
complete consecutive triples. Compare/validate calibration also uses three triples per
calibration scenario. Choose at least three independent recording sessions in advance and
aggregate all of them. No adaptive retries, discarded outliers, synthetic samples, pooled
leftovers or replacing sessions until bounds pass.

Raw CV and calibration batch CV retain their 10% limits. Aggregation retains median ±
max(3 × 1.4826 × MAD, 3% × median), the 15% radius guard and the outlying-batch guard.
These are conservative heuristics, not coverage-guaranteed prediction intervals. Nine
samples improve batch coverage but do not establish equivalence or reliable tail latency.
Local runs cannot qualify a candidate.

Timing noise and same-commit timing drift produce `inconclusive` after the fixed measurement
and quality budget finishes. Both binaries' timing variability is examined. Quality,
behavior and definite resource failures keep their failure verdicts. Execution errors,
incomplete samples, missing quality evaluation and incompatible contracts remain blocking.
Noisy recordings can enter aggregation, where all samples still face the stability checks.

Local reports/aggregation use schema 2 and profile policy `local-batches-v2`. Inconclusive
aggregation retains source hashes and batch medians but exposes no bounds; comparison
rejects it. Old reports are never rewritten. Every retained local recording predates v2
and requires fresh recording. The global harness digest also changes for cloud, requiring
fresh recording despite unchanged cloud measurement settings and qualification policy.

## All retained local evidence

`results/local-noise-audit-20260908.json` audits **all 12 local reports** under
`~/.cache/visiongrep-bench/runs`, including failures. It retains original report SHA-256,
verdict/reasons, contract, all wall/application-wall/phase samples and the new raw-noise
diagnostic. The other five directories have cloud configurations. Reproduce read-only:

```sh
python3 benchmarks/audit_local.py ~/.cache/visiongrep-bench/runs > /tmp/local-audit.json
```

| Session | Historical verdict | Raw CV above 10% |
|---|---|---|
| 20260907-131710-3359d12f | validation_passed | none; three pairs |
| 20260908-120901-f78f6c5c | invalid | cached_text 12.59% |
| 20260908-122007-f42d2998 | foundation_recorded | none |
| 20260908-123051-b9ab938a | foundation_recorded | none |
| 20260908-123614-7055109c | foundation_recorded | none |
| 20260908-184338-01efa9fe | invalid | cached_text 10.61% |
| 20260908-185252-2278cfb6 | foundation_recorded | none |
| 20260908-185657-6748ea27 | foundation_recorded | none |
| 20260908-190119-9dd5c719 | foundation_recorded | none |
| 20260908-190735-9a2190df | foundation_recorded | none |
| 20260908-214925-9a3f9c5f | invalid | modified_1pct 14.52% |
| 20260908-215550-4c60b373 | invalid | novel_text 13.42% |

All four historical noise failures remain flagged by the unchanged raw-CV diagnostic.
Each recording has five samples (one complete triple); the validation has three.
**All 12 are ineligible under v2.** No missing samples are fabricated and no historical
session is claimed to pass with nine. This audit cannot estimate v2 false-positive or
false-negative rates; fresh recordings must validate its empirical behavior. Contracts
must not be pooled to substitute for independent batches.

The initially selected novel-text medians (203.494, 205.295, 216.844 ms) illustrate MAD's
instability with just three points. Replacing the last session changes which scenario
fails: cached medians 9.168, 8.811, 8.368 ms give about 4.6% CV but an 18% robust radius.
Neither selection becomes a new foundation. More complete batches address the shortage
of evidence; preserving conservative guards may still produce an inconclusive result.

## Harness interference and limitations

The helper times the CLI subprocess directly using `perf_counter`; the runner's 100 ms
polling is outside that interval. Setup, seed copies, warm pre-reads, behavior checks and
report serialization occur outside the measured child. Pre-reads intentionally implement
warm-cache policy. Reset writes can still cause background I/O, and heartbeat fsync every
five seconds can overlap a child. Historical local reports have no I/O/heartbeat trace to
establish causation. No arbitrary sleeps, local sync or measurement-boundary changes are
justified by this evidence.

Across all eleven record sessions, external minus application wall time has a cached_text
median of 4.477 ms (range 3.999–6.666), and indexed_image median of 4.469 ms (4.099–5.433).
These gaps include launch/teardown and timing-file work; they are not pure timer error and
must not be subtracted. Novel_text's gap reaches 47.674 ms. The last two failed sessions
also show increased model-session construction: modified_1pct reaches 237.330 ms and
novel_text 205.906 ms. Thus variation is not confined to external overhead. Worker-summed
decoding/preprocessing phases cannot be added to reconstruct elapsed wall time.

An absolute-plus-relative allowance was considered but deferred. These samples do not
isolate a stable measurement-error floor from real launch/application variation. Choosing
a millisecond allowance to pass cached-text failures would fit thresholds after observing
the data. Until controlled fresh measurements justify one, v2 retains existing limits and
labels uncertainty honestly. No performance improvement is claimed.

## Fresh fixed-budget experiment, 2026-09-08

Before publishing the change, three sequential, independent local recording processes
were run with this command, once per session:

```sh
python3 benchmarks/bench.py run --mode record --profile local-quick --max-hours 1
```

The budget of three sessions was fixed before starting. No run was replaced or retried.
All three retained nine samples for every scenario (45 timing samples each), passed all
scenario behavior checks and completed the 440-query foundation quality evaluation.

| Session | Verdict | Timing limitation |
|---|---|---|
| 20260908-222242-b61c82f7 | inconclusive | novel_text raw CV 16.293% |
| 20260908-222700-45d8895f | inconclusive | cached_text raw CV 17.880% |
| 20260908-223123-029c221a | foundation_recorded | all scenario CVs below 10% |

The exact aggregation command was:

```sh
python3 benchmarks/bench.py foundation \
  /Users/uday/.cache/visiongrep-bench/runs/20260908-222242-b61c82f7 \
  /Users/uday/.cache/visiongrep-bench/runs/20260908-222700-45d8895f \
  /Users/uday/.cache/visiongrep-bench/runs/20260908-223123-029c221a \
  --output /Users/uday/.cache/visiongrep-bench/local-foundation-v2-20260908.json
```

It completed with verdict **inconclusive** and empty calibration bounds. It retained all
nine batch medians per calibration scenario and all three source report hashes. Reasons:
the two raw-CV failures above and an outlying novel-text calibration batch. This experiment
successfully exercised recording and aggregation, but **did not establish a usable local
foundation**. The limits were not adjusted after observing these failures. The new policy
preserves complete evidence and reports uncertainty; it does not guarantee calibration
on this machine under its current conditions.

All three share harness SHA-256
`6d323d1709693c70c61aae7f924703faa1e51b0828808aa86fce76434d52b79c`.
Their original report SHA-256 values, in table order, are:

```text
523ffc3c9ce8fa8827ce8de395d1940cb99102059ac219f89821f62420171428
b56439ddde95ec7780e71d0dee1bb7172f46467fe78a5ee26505f82bed72454a
8cac7355e7495b48631541c9ddd0c8ed881771cbee26d7c74fa1da3bb420d981
```

This Markdown-only evidence update does not change the measurement contract digest.
The earlier JSON audit remains an immutable snapshot of the twelve historical reports;
the three fresh sessions are documented here rather than rewriting that snapshot.

## CI and administration

Rust and offline benchmark workflows now trigger only on pushes to main, retaining
benchmark path filters. The separate manual model-contract smoke workflow remains manual.
No cloud run is launched by this change.

The GitHub connector returned an empty repository ruleset list on 2026-09-08. Classic main
branch protection returned HTTP 403 (integration lacks administration access); the browser
was signed out and could not display settings. No protection was changed. An administrator
must check whether classic protection requires `Rust / ubuntu-latest`, `Rust / macos-15`,
or `offline`, and remove only those corresponding CI requirements if present. Preserve
reviews, force-push restrictions and unrelated protections.
