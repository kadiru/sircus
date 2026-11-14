# Testing an expanded library of table-bounce shots

In the [matched model refit](MJLAB_ROUTE_REFIT.md), the table specialist
captured 13 of 50 positions with 25 setups. Its original weights captured
9 of 50 positions with 32 setups. This experiment uses the refitted
weights for both variants and tests whether more starting shots help
the arms find feasible settings.

- Start with the original library of verified table captures. For the
  expanded library, add the verified table captures from the
  [paired collection](MJLAB_PAIRED_ROUTE_COLLECTION.md). Preserve
  original shots and their evidence.
- Freeze 50 fresh bucket positions before evaluating. Sample the same
  tabletop strata using seed `2026112001`. Each position is at least 5 mm
  from all inventoried placements and every other test position.
- For each library, choose four starting shots using the same
  nearest-position rule and the same ranking within each source position.
  The frozen table specialist predicts adjustments. Use the first
  candidate that passes the existing pose and path checks, and freeze all
  100 choices before dropping any ball. Give each library at most one
  release per position.
- Use the existing mjlab scene and 64 substeps. A table capture requires
  a board rebound, a verified table rebound before whole-ball entry and
  one continuous second inside the bucket. Report setup and table captures
  out of all 50 positions per library. Report captures per release and
  gains and losses on the same positions.
- The CPU audit reconstructs the starting shots, predictions, selected
  poses and results. It independently rescores native evidence and the
  nine controls run before and after the comparison.

The Ubuntu archive is `intergrid-table-library-v1`. Use the audited
`fixed35-sixhour-v2`, `intergrid-route-policies-v1`,
`intergrid-matched-routes-v1`, `intergrid-paired-collection-v1` and
`intergrid-route-refit-v1` archives and their audit receipts. Keep raw
evidence outside Git. The producer is
`python -m sircus.mjlab.intergrid_library_study --help`; the CPU auditor is
`python -m sircus.mjlab.intergrid_library_audit --help`. The simulation
uses the 4090 via the shared lock.
