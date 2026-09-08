# Local median screening v3

Local benchmarking is inexpensive candidate triage before considering cloud runs. Its
purpose is to reject inappropriate changes and identify plausible gains on this Mac, not
to prove a five-percent improvement transfers to another machine. Cloud acquisition,
noise limits, sample budgets and qualification rules are unchanged.

## Recording and uncertainty

The budget remains nine samples per scenario per session, grouped into three consecutive,
non-overlapping triples. Plan at least three independent sessions before observing results;
include all completed sessions without replacements or adaptive retries. All raw samples,
including spikes, remain in the original reports and raw CV remains visible.

The local calibration statistic is the **median of the three-sample batch medians**.
For aggregation, pool the batch medians but retain their session membership. Report:

- This typical-latency estimate and an approximate 95% percentile interval, using 5,000
  deterministic bootstrap draws that resample sessions, then batches within each session.
- Relative median uncertainty: the larger distance from the median to either interval
  endpoint divided by the median. Require at most **10%**.
- Batch stability: maximum absolute deviation of any batch median from the aggregate
  median, divided by that median. Require at most **15%**.
- Per-session medians, every batch median, and raw summary statistics including CV.

The 10% precision budget is a coarse local screening resolution. The 15% guard is now
applied directly to observed batch displacement, not to a MAD-derived radius that can
reject otherwise close batches. Raw CV and exceeding a narrow MAD band no longer reject
a foundation. These are explicit policy choices for coarse triage, not estimates of a
machine's noise floor. They are not tuned per scenario or candidate, and there is no
absolute millisecond allowance.

Aggregation evaluates precision and stability over all sessions for **all five scenarios**.
It does not require every individual session's interval to pass first: additional sessions
can improve precision. A sustained shift in a session or batch still blocks calibration.
Quality/behavior failures, incomplete observations and incompatible contracts remain
errors. An inconclusive report never exposes usable calibration bounds.

## Candidate triage

Before a comparison, the newly measured foundation must have adequate median precision
and batch stability. Its median must fall within the historical median interval expanded
by 10% on each side. Individual precheck batches need not fall within the old narrow MAD
band. The historical interval is a drift check, not a replacement for contemporaneous
measurements: candidates are still paired F/C and C/F with newly executed foundation runs.

Timing and resource comparisons retain paired median-ratio bootstrap intervals and their
existing multiple-scenario adjustment. A candidate is `promising` only if its local
measurements are stable, correctness checks pass, all local timing/resource comparisons
exist, their intervals exclude regressions greater than 5%, and at least one local timing
scenario improves by 5% or more with an interval excluding zero. `promising` is not
`qualifies`, does not cover unmeasured cloud scenarios, and never launches a cloud run.
Clear regressions or a confidently negligible gain are `does_not_qualify`; uncertainty
remains `inconclusive`. The 10% reference precision budget does not mean accepting a
candidate with a possible 10% paired regression.

With only three sessions and three batches each, bootstrap coverage is not guaranteed.
Intervals can be optimistic for correlated/nonstationary workloads or identical observed
batch medians. The batch drift guard is a separate check, not a proof of stationarity.
One isolated spike can leave the median unchanged; this deliberately targets typical
latency and does not certify tails or absence of occasional stalls. Raw timings remain
available for those questions. Cloud confirmation is required for performance claims.

## Compatibility and retained evidence

New local reports and foundations use schema 3 and policy `local-median-v3`. The comparison
contract is still checked in full. Reanalysis is allowed only for current measurement
contracts or the known nine-sample v2 harness SHA-256:

```text
6d323d1709693c70c61aae7f924703faa1e51b0828808aa86fce76434d52b79c
```

The v3 change leaves its acquisition code (process timing, resets, warming, sample order,
builds and quality execution) unchanged. A reanalyzed foundation preserves `source_contract`
and immutable source report hashes, and emits a current analysis contract with the policy
and harness digest updated. All remaining contract fields are preserved. This is an
explicit migration, not a general permission to ignore harness hashes. Unknown v2/v3
harnesses require fresh recordings; five-sample records cannot supply three full batches.
All source sessions must have identical contracts. Old foundation files are never edited.

At policy selection the retained population was 15 local reports: the 12 historical reports in
`results/local-noise-audit-20260908.json`, plus all three recent v2 sessions. The older
12 still have only three or five samples and are insufficient; their original verdicts
and raw CV diagnostics remain intact. All three eligible nine-sample sessions, including
both inconclusive sessions, are included in v3 reanalysis. No session is selected away.
Reproduce an audit of the entire directory with:

```sh
python3 benchmarks/audit_local.py ~/.cache/visiongrep-bench/runs > /tmp/local-v3-audit.json
```

The same three recent sessions that were inconclusive under v2 are **calibrated** under
v3's stated typical-latency policy:

| Scenario | Median ms | Approximate 95% median interval ms | Relative uncertainty | Max batch deviation |
|---|---:|---:|---:|---:|
| index_absent | 5055.446 | 4999.403–5083.817 | 1.11% | 1.30% |
| novel_text | 276.044 | 268.854–285.726 | 3.51% | 7.76% |
| cached_text | 9.648 | 9.338–10.451 | 8.32% | 9.22% |
| indexed_image | 9.561 | 8.791–10.287 | 8.06% | 11.07% |
| modified_1pct | 288.596 | 273.983–298.493 | 5.06% | 5.95% |

Use a new output filename; the v2 inconclusive artifact stays unchanged:

```sh
python3 benchmarks/bench.py foundation \
  /Users/uday/.cache/visiongrep-bench/runs/20260908-222242-b61c82f7 \
  /Users/uday/.cache/visiongrep-bench/runs/20260908-222700-45d8895f \
  /Users/uday/.cache/visiongrep-bench/runs/20260908-223123-029c221a \
  --output /Users/uday/.cache/visiongrep-bench/local-foundation-v3-20260908.json
```

These retrospective results test usability, not independent validation of the selected
policy. Regression tests additionally cover isolated raw spikes, sustained batch/session
drift, imprecise medians, missing/failed checks, contract migration, candidate regressions,
uncertain comparisons, and promising improvements. No real candidate is claimed to have
passed v3 merely because the reference calibrated.

## End-to-end same-commit check

One additional fixed-budget local comparison used the reanalyzed v3 foundation:

```sh
python3 benchmarks/bench.py run --mode compare --profile local-quick \
  --candidate benchmark-foundation-v1 \
  --baseline /Users/uday/.cache/visiongrep-bench/local-foundation-v3-20260908.json \
  --max-hours 1
```

Run `20260908-230035-86bb928f` accepted the full contract, completed all nine pairs in
each scenario and all 440 quality queries for each binary, and passed quality/behavior
parity. It did not falsely promote an unchanged candidate. The outcome was `inconclusive`
because the cached-text precheck median, 8.020 ms, fell below the historical reference's
8.404 ms lower bound. Its novel-text median, 242.834 ms, remained in range. Paired median
improvements were only 0.24–0.96%; cached-text's paired interval still extended to 5.81%.
Neither the precheck margin nor candidate rules were changed after seeing this result.

This confirms the baseline is usable by the runner, but not that every later session
will pass environmental drift checks. Even sub-millisecond differences can matter for
short operations. The reference calibrated under v3; later machine conditions can still
make a comparison inconclusive. The run's report SHA-256 is:

```text
e429830761eebd8891da5970cb21d1609efc00dc3f03f3a3b4bc2a3ae8372d04
```

The original evidence, causal limitations and administration blocker are preserved in
[the v2 record](LOCAL_SCREENING_V2.md). Classic branch protection still needs an
administrator's check; no unrelated protection or cloud policy is changed here.
