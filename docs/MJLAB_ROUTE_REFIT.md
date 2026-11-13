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
