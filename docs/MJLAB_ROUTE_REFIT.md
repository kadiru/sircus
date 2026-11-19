# Does collecting both bounce routes improve predictions?

The [previous matched test](MJLAB_MATCHED_ROUTE_TEST.md) compared three
ways to learn how a known shot must change when the bucket moves:
a pooled model, a shared model with a route-specific correction, and two
separate route specialists. Each planner tried both a direct board-to-bucket
shot and a board-to-table-to-bucket shot. The
[paired collection](MJLAB_PAIRED_ROUTE_COLLECTION.md) then recorded new
shots at 11 positions that already supported both routes.

## Training and held-out comparison

1. Start with the original 20,330 training records. Add 342 valid paired
   releases: 137 direct captures, 154 table captures and 51 misses. Four
   invalid releases and the historical test shots do not enter training.
   The fit learns from verified captures. It matches compatible shots
   from different positions, weighting the two routes equally and each
   paired target position equally within its route. The model form, feature
   scaling and regularization are unchanged.
2. Freeze 50 test positions before evaluation, at least 5 mm from the
   inventoried placements. This excludes all previous training, collection
   and evaluation positions. Use split seed `2026111901`.
3. Give each original and refitted planner the same four saved starting
   shots per route from the **original** recipe library. Each planner
   adjusts those settings using its frozen coefficients. Select the first
   candidate that passes the same arm pose and path checks. Freeze all 600
   choices before release. Each choice has at most one release; a failed
   setup counts as a failed attempt. No feedback from test shots changes
   the next choice.
4. Score with the existing mjlab scene and 64 substeps. A capture requires
   a board rebound and one continuous second with the whole ball inside
   the bucket. The table route also requires a measured table rebound
   before whole-ball entry. A capture by the other route, an experimental
   arm-assisted capture or invalid physics does not count for the requested
   route. Repeat the same nine control shots before and after the test.
5. Independently reconstruct the split, fitting, selected poses, scoring,
   and matched original/refit results on CPU. Report captures out of 50
   for each planner and route, plus positions gained and lost.

The fixed starting-shot library isolates changes in the model fit. This
does not measure the benefit of using the new shots as starting points
during inference. The seven positions reused for paired data collection
are development positions for this refit; the old test records remain
unchanged.

## Reproduce

Use the audited Ubuntu outputs `fixed35-sixhour-v2`,
`intergrid-route-policies-v1`, `intergrid-matched-routes-v1` and
`intergrid-paired-collection-v1`, with their adjacent `-audit.json`
receipts. Use `corrected-grid-v1` as the runtime reference and the same
read-only scene assets. The producer is
`python -m sircus.mjlab.intergrid_route_refit_study --help`; the
independent verifier is
`python -m sircus.mjlab.intergrid_route_refit_audit --help`. Use a fresh
output directory and the shared GPU lock. The 4090 runs the regression
solves and native drops; CPU workers prepare the paths and the audit runs
with CUDA hidden. No Isaac files, services, or 4070 workloads are changed.

The Ubuntu archive `intergrid-route-refit-v1` contains the two checkpoints,
the exact training rows, all 50 layouts, frozen decisions, and native traces.
Keep these raw files outside Git.

## Audited results

The three refits used 9,226 measured captures from 83 distinct positions:
5,130 direct captures at 72 positions and 4,096 table captures at 22.
The fit formed 34,535 matched pairs. The additional 51 valid misses are
retained in the ledger but did not enter the fitting loss.

All 600 requests were accounted for: 444 releases and 156 setup failures.
All 18 control releases behaved as expected. The independent CPU audit
passed 114,690 checks. The mjlab test suite ran 2,272 tests with seven
skipped. Both generations used the same fixed starting-shot library.

| Planner | Direct captures, original → refit | Direct gains / losses | Table captures, original → refit | Table gains / losses | Either route, original → refit |
|---|---:|---:|---:|---:|---:|
| Pooled | 35/50 → 35/50 | 2 / 2 | 10/50 → 12/50 | 2 / 0 | 39/50 → 41/50 |
| Route-conditioned | 34/50 → 34/50 | 1 / 1 | 12/50 → 11/50 | 2 / 3 | 39/50 → 39/50 |
| Two specialists | 33/50 → 34/50 | 1 / 0 | 9/50 → 13/50 | 5 / 1 | 38/50 → 39/50 |

“Either route” gives both attempts at each position; it is not a
first-shot or route-selection result. Gains and losses count matched
positions captured by only one generation of the same planner.

| Planner | Direct setup failures | Table setup failures | Table captures per release |
|---|---:|---:|---:|
| Pooled | 6 → 6 | 13 → 15 | 10/37 → 12/35 |
| Route-conditioned | 8 → 8 | 18 → 22 | 12/32 → 11/28 |
| Two specialists | 9 → 8 | 18 → 25 | 9/32 → 13/25 |

For the table specialist, model predictions outside the usable candidate
set caused 13 → 14 setup failures. Proposals with no admitted pose or
path accounted for the other 5 → 11. Its released misses fell from 23 to
12. It found better-scoring flights among the shots it released, while
its setup coverage fell. Its net gain was four positions (exact paired
two-sided p = 0.219); this one split does not establish a reliable gain.
The pooled refit had the best coverage with both attempts available.
The study recorded seven direct captures on table requests and three
invalid trajectories; these count as failures for the requested route.
There were no arm-assisted captures. No model was promoted.

Next: keep these weights fixed and test whether adding the new measured
table shots to the starting-shot library reduces setup failures. Reserve
a new test split for that separate comparison.

- Frozen producer: `58837dd4edc26b7b5bee02f29c1b239284ed98b8`
- Original checkpoint SHA-256: `847b1119d4321ce362b628f7ef5f2ace3ed202cbb46b597bb9875bf1fc351c52`
- Refit checkpoint SHA-256: `3b43cb74946ab4dd53f110ffe1d9e07bff5409bd40966f1f128eb72009da6161`
- Training ledger SHA-256: `98161c042e377d904cfb92e07b158dcdf33206ad692ce4284156326b987a2b30`
- Report SHA-256: `ada19d180e2509cb9430c5247d698eef0a765e80f52466c36e73b20742384c8c`
- Evidence manifest SHA-256: `d3c527be9006ae990046459bc39162fac8dc8e9181620f3fbb72a989c0944a59`
- Audit SHA-256: `b28693fd997aa1f4c95a17b15bc4b35e927c9284933b5714d7e943e796c061ef`
