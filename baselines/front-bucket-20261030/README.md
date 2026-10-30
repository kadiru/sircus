# Front-bucket learning evidence — October 30, 2026

This is a separate **synthetic, provisional negative-Y profile**. Historical
positive-Y, rectangular-container and earlier bucket baselines remain unchanged.
No physical robot, Viam service, camera or audio connection was used by this study.
Native collision-path replay remains blocked.

The prespecified default is N=128, seed 20261030, checkpoint:
`7b9a76dddceca22490933f55edcdd82ebc596b2d6539f0d445b3be0f23c1d2e6`.
It was not selected using test performance. Its frozen matrix prediction captured
33/48 held-out layouts (68.75%,95% Wilson 54.7–80.1%), versus fixed 25/48 and
separately labeled 16-proposal search 37/48. The paired gain over fixed is
16.7 percentage points (paired-layout bootstrap 95% 6.25–29.17). This is insufficient
for a reliable-deployment claim; no physical transfer has been validated.

## Actual learning curves

Each row uses the same 48 independent held-out layouts and paired release seeds.
Three training seeds use nested data prefixes. Validation 16 chooses among three
predeclared ridge candidates; teacher labels use training layouts only. Inference
is a frozen feature expansion and matrix multiply, with no per-scene optimizer.

| Sampled N | Seed 20261030 captures | Seed 20261031 captures | Seed 20261032 captures | Mean | Layout interval | Seed range |
|---:|---:|---:|---:|---:|---:|---:|
|8|28/48|28/48|29/48|59.0%|45.0–71.8%|58.3–60.4%|
|24|31/48|32/48|34/48|67.4%|53.2–79.9%|64.6–70.8%|
|64|32/48|34/48|31/48|67.4%|53.2–79.2%|64.6–70.8%|
|128|33/48|34/48|33/48|69.4%|55.4–81.9%|68.8–70.8%|

This is 48 independent layouts, not 144 independent tests. Aggregate intervals
cluster by layout and envelope a bootstrap with a Wilson effective-layout bound;
seed variability is separate. Performance is not strictly monotonic and appears
to plateau around two-thirds. N counts sampled layouts: the third seed fits
8/24/63/125 valid labels; other seeds fit all sampled labels. Three layouts had
no valid label within the 16-proposal budget (all proposed dropper endpoints
unreachable); this does not prove intrinsic layout infeasibility. There are 381
valid labels and 291 teacher captures among 384 training layouts/6,144 proposals.

## Challenges and declared interventions

Default-checkpoint distance results:60 mm 22/24;100/160/220 mm 24/24;280 mm 17/24;
340/400/450 mm 0/24. Distance 340 remains within the trained layout bounds, exposing
a limitation even before extrapolation. Offset−200/+200 mm each gives 0/24, with
endpoint failures included. All failures remain in denominators. Distance groups
independently sample small board-position variations; they are descriptive
sampled challenges, not paired causal distance experiments.

The noise cohort has 24 independent layouts with three replicates. Default
capture counts at XY/gap 2/3,10/15,20/30 and 30/45 mm are 69/72,68/72,68/72,68/72.
These layouts are closer on average:207.1 mm forward distance,2/24 beyond 280 mm,
versus shared test 241.2 mm,16/48 beyond 280 mm. Its high capture rate is not general
held-out performance. The random manifest was frozen, not resampled. Use final
clustered intervals; generic stdout noise intervals treat attempts independently
and are not inference-quality. Original logs remain outside Git unchanged.

A separate paired intervention fixes U/V at center, edges and corners while
retaining each model's predicted roll/pitch/gap. For the default checkpoint,
center captures 31/48; U=+65 captures 31/48 with 14 endpoint failures; U=−65 and
V=±95/corners capture 0/48. Requested and simulated actual impact U/V, support
response and failure counts are recorded. Changing U/V changes trajectory,
reachability and compliance together; these differences cannot be attributed
solely to board suspension. Intervention results are not policy inference scores.

Post-study diagnostics compare the same 24 layouts per condition using the fixed
action and 16-proposal search, without fitting or selecting any checkpoint:

| Condition | Frozen default | Fixed | Finite search |
|---|---:|---:|---:|
|Distance 220 mm|24/24|24/24|24/24|
|Distance 340 mm|0/24|0/24|2/24|
|Distance 450 mm|0/24|0/24|0/24|
|Offset−200 mm|0/24|0/24|18/24|
|Offset+200 mm|0/24|0/24|12/24|

The offset controls reveal substantial out-of-distribution inference failures
where bounded search can find solutions. Zero finite-search successes do not
prove impossibility or an optimality bound.

## Assets, counts and provenance

Origin is arm 10 base; +X points toward arm 14; +Z up; operator/bucket are toward−Y.
Nominal board center[600,−150,350]mm, RPY[0,−30,90]°, bucket[600,−400]mm and table
placement are provisional. Measured board 200×300×5 mm and tapered bucket inner
177/150 mm, outer 200/158 mm remain unchanged. Reported height 175 mm is unresolved
as overall versus interior depth; simulated interior 175/floor 8 mm are assumptions.
The screenshot loading/dropping TCPs are not calibrated ball centers.

Protocol attempts:6,144 teachers;576 validation;576 frozen test;48 fixed baseline;
768 held-out search;2,376 frozen challenges;1,296 forced-contact;1,920 post-study
search;120 post-study fixed. Counts exclude gallery renderings/regression
inspections. There are 832 distinct core layouts; including the separate initial
24/6/8 run yields 846 distinct layouts (24 initial training layouts overlap the
first curve pool). Repeats/seeds are not additional independent layouts.

`initial-24-6-8/` preserves the earlier corrected-profile run separately: frozen
3/8, fixed 2/8, limited 4-proposal search 2/8. Its hold-out layouts and teacher budget
differ, so do not compare raw percentages as learning gains. Historical
positive-Y evidence stays in its original baseline; it is another distribution.

- `checkpoint.json`, `checkpoints/`: default and all12 frozen model contracts.
- `manifest.json`: predeclared layouts, seeds, search budgets and splits.
- `report.json`: authoritative final UI report, including14 actual replays,
  labeled interventions and diagnostics. Four replay clips use the same first
  predeclared held-out layout at N 8/24/64/128; all four miss and were not replaced.
- `summary.json`, `challenge-summary.json`: exactly the final report summaries.
- `core-*`, `report-core.json`, `report-pre-diagnostic.json`: earlier immutable
  evidence snapshots, explicitly separate from the final extension summaries.
- `test-scorecard.json`, `comparator-scorecard.json`: compact selected actions,
  endpoint checks, capture/retention outcomes and IDs; no raw candidate traces.
- `contact-intervention/`, `post-study-diagnostics/`: extension manifests,
  summaries and a compact diagnostic scorecard.
- `independent-model-verification.json`: all12 fits reconstructed from train-only
  labels with augmented least squares; maximum weight difference 1.89e−15.
- `SHA256SUMS`: every published file, including this README.

All 18 study regressions passed: source/dependency/asset guards, split separation,
no-search prediction, unavailable-label handling, clustered uncertainty, artifact
checksums, all 576 published frozen actions versus direct prediction, final-summary
consistency, and actual numerical replay within 0.0001 mm. These are numerical
checks, not physical calibration or native motion-path validation.

Reproduction and extension commands:
[`sircus/studies/README.md`](../../sircus/studies/README.md).
Verified Python 3.12.14, NumPy 2.5.4, SciPy 1.18.1. Raw teachers, traces and logs stay
under ignored `.runtime/overnight-study`; credentials, host calibration, captures,
queues, virtualenvs and platform binaries are absent. All numerical jobs finished.
