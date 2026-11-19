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

## Audited results

Both libraries used the same frozen refitted table specialist. The
expanded library added 154 verified shots, increasing source coverage
from 15 to 22 bucket positions.

| Library | Verified starting shots | Feasible setups | Table captures | Captures per release |
|---|---:|---:|---:|---:|
| Original | 3,942 | 23/50 (46%) | 10/50 (20%) | 10/23 (43.5%) |
| Expanded | 4,096 | 25/50 (50%) | 13/50 (26%) | 13/25 (52.0%) |

The expanded library found a setup at three positions where the original
failed and lost one setup the original found. It gained four capture
positions and lost one. The paired differences were +4 percentage
points for setup and +6 for capture. Exact paired two-sided tests gave
p = 0.625 and p = 0.375, respectively. The measured gains come from one
50-position comparison.

The original library had 15 positions with no candidate and 12 where
all candidates failed the arm checks. The expanded library had 17 and
eight, respectively. The expanded library used a newly added starting
shot in 14 of its 25 drops and captured on nine of those 14.

Both sets of controls passed. The independent CPU audit passed 27,966
checks across all 100 attempts, 48 test releases and 18 controls. All 25
released misses had valid physics. The mjlab regression suite ran 2,274
tests with seven skipped. Simulation used the 4090. The 4070 and Isaac
were left alone.

- Frozen producer `6210ae6bd0e0a109481b071c9f2afa613e65a026`
- Frozen checkpoint SHA-256 `3b43cb74946ab4dd53f110ffe1d9e07bff5409bd40966f1f128eb72009da6161`
- Report SHA-256 `b6cc2c800b34d8c1c3076ed48a0cfafcf84bf5f770d0c6c2b5360dda771d795a`
- Evidence manifest SHA-256 `c783202bc7f2164e1f71a8145bab16bf27b52f33bdd55c24ebb572b1b1531cb5`
- Audit SHA-256 `902dc6873988d5d43c4f4cd154ec77d9e7753f3a105070026377795f4606bc28`
