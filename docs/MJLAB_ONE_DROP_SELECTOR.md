# Choosing a bounce route before one drop

Compare two planners at the same 50 fresh bucket positions. Direct-only
chooses a board-to-bucket shot. The route selector also considers a
board-to-table-to-bucket shot using the
[expanded table library](MJLAB_TABLE_SHOT_LIBRARY.md).

1. Freeze the original direct library, the expanded table library and
   the refitted specialist checkpoint. Use up to four starting shots
   for each route and take the first candidate for that route that passes
   the existing pose and path checks. Both planners share the exact
   direct candidate and its arm configuration.
2. If only one route passes setup, the selector chooses it. If both
   pass, compare capture rates at the five geographically nearest
   positions in the [matched refit study](MJLAB_ROUTE_REFIT.md) where
   both routes passed setup. Add one simulated count to both successes
   and failures for smoothing. Choose the higher rate, favoring direct
   on ties. Those earlier table attempts used the original starting
   library, so their outcomes serve as a proxy for the expanded library.
3. Freeze the test positions using seed `2026112101`, at least 5 mm from
   archived placements and each other. Freeze all choices before any
   test releases. Each planner drops at most once at each position.
4. Use the existing mjlab scene at 64 substeps. Count a standard capture
   when the ball rebounds from the board and stays fully inside the
   bucket for one continuous second. Score both planners on all 50
   positions, including setup failures. Save the chosen and measured
   routes, plus the paired gains and losses.
5. Run the same nine control shots before and after the test. Verify the
   selected routes, one-drop budget and all recorded outcomes in a
   separate CPU audit.

The producer is `python -m sircus.mjlab.intergrid_route_selector_study --help`
and the verifier is `python -m sircus.mjlab.intergrid_route_selector_audit --help`.
Use the audited archives listed in the [library comparison](MJLAB_TABLE_SHOT_LIBRARY.md)
and add `intergrid-table-library-v1` and its audit receipt. The new evidence
lives in `intergrid-route-selector-v1` on Ubuntu. Keep raw evidence outside
Git. Native simulation uses the 4090 and the existing shared lock.

## Audited results

| Planner | Feasible setups | Captures | Captures per drop |
|---|---:|---:|---:|
| Direct-only | 41/50 (82%) | 33/50 (66%) | 33/41 (80.5%) |
| Route selector | 47/50 (94%) | 40/50 (80%) | 40/47 (85.1%) |

The selector captured all 33 positions that direct-only captured and
added seven. It chose direct 36 times and captured 31 of those drops.
It chose table 11 times and captured nine. Five added captures came
from six positions that admitted only table. The other two came from
five positions where both routes passed setup and the selector chose
table. Both planners scored identically on all 36 duplicated direct
commands.

The paired capture difference was 14 percentage points, with a
two-sided exact paired p-value of 0.015625 on this 50-position split.
The gate used 19 older positions where both routes released. At each
new position, it used the five nearest of those 19. The selector could
consider four direct plus four table starting shots, compared with
four direct starting shots for direct-only. Both methods had the same
one-drop limit. The result measures adding table candidates and choosing
between feasible routes together.

Both control sets passed. The independent CPU audit passed 34,460
checks, accounting for all 100 attempts, 88 test drops and 18 controls.
One direct-only release was invalid and counts as a failure. The mjlab
suite ran 2,277 tests with seven skipped. Native simulation used the
4090. The 4070 and Isaac were left alone.

- Frozen producer `21cad439f6e4845b9c6def43df68c9b1ffd80746`
- Frozen checkpoint SHA-256 `3b43cb74946ab4dd53f110ffe1d9e07bff5409bd40966f1f128eb72009da6161`
- Report SHA-256 `391682a65763f7e7a95d20b1322bf15e205e30a9b05cc81157f6d6f416df806b`
- Evidence manifest SHA-256 `fa404bf492da0f21994f5460c81f8ab7cabe8311cf388a2e076db1dac9c6fe9f`
- Audit SHA-256 `80f491c47f76bc087f242dc62719c92519cc6c6bb3b08b0d3a26eb7743555fa3`
