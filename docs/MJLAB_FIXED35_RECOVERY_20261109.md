# Fixed-grid recovery: completed handoff

Five of the 11 original failed grid locations now have a confirmed recipe,
each with **3/3 fresh nominal captures**. All **24/24 unchanged successful
parent recipes** captured again. This gives **29/35 locations with a verified
recipe** in this study; the existing frozen policy remains **24/35**. No policy
was trained, replaced or promoted.

The [protocol](../sircus/mjlab/FIXED35_RECOVERY.md) kept the canonical bucket
coordinates fixed, bucket-facing gripper orientation, native64 physics and
version 3 scoring. This is development search on an already inspected grid,
not held-out generalization, placement-noise validation or 128-substep testing.

## Completed execution

| Phase | Actual requests | Result |
|---|---:|---|
| Unchanged parent preservation | 24 | 24 captures |
| Bounded recipe search | 384 | 5 captured discoveries |
| Fresh confirmation | 15 | 15 captures; five recipes confirmed |
| Before/after controls | 18 | All gates passed |
| Total | 441 | Below the frozen 843-request cap |

Execution ended at finite candidate caps, not a time limit. The worker took
4,105.75 seconds; the complete pipeline took 4,135.29 seconds. The full ledger
contains 203 native releases, 238 setup refusals, and 147 released noncaptures.
**Three of those noncaptures had invalid physics** and remain separately marked;
they do not count as valid flight misses or successful evidence. The aggregate
56 captures include controls, parent replays and confirmation repeats, so
56/441 is not a policy capture rate.

A capture requires board contact, whole-ball entry and one continuous second
inside. Table/rim assistance counts; robot, floor and unattributed contacts
invalidate capture. Two newly confirmed recipes, at grid 01-06 and 02-06, had
table contacts in their discovery and each fresh repeat. Historical late robot
contacts are recorded but do not, by themselves, establish the initial cause
of a missed bucket.

## Five confirmed recipes

Positions and heights below are scene-coordinate millimetres; tilt is in
degrees. Grid IDs are zero-based row-column. The ball-release centre is the
recorded native value, not an assumed dropper height. Drop gap is a separate
action parameter and is not total release height. Values are rounded for
reading; the [versioned summary](../baselines/mjlab-fixed35-recovery-20261109/summary.json)
retains full-precision actions, native release centres and record identities.

| Grid | Bucket X/Y (mm) | Board X/Y (mm) | Board centre Z (mm) | Tilt (°) | Drop gap (mm) | Ball release centre Z (mm) | Fresh captures |
|---|---|---|---:|---:|---:|---:|---:|
| 00-00 | (100.0, -650.0) | (533.9, -231.2) | 307.0 | 26.09 | 398.8 | 743.3 | 3/3 |
| 01-06 | (1240.0, -425.0) | (597.9, 16.2) | 459.4 | 26.04 | 252.2 | 749.1 | 3/3 |
| 02-03 | (670.0, -200.0) | (559.1, -62.5) | 509.6 | 10.10 | 195.0 | 742.1 | 3/3 |
| 02-06 | (1240.0, -200.0) | (597.3, -15.2) | 472.2 | 27.24 | 162.6 | 672.3 | 3/3 |
| 04-04 | (860.0, 250.0) | (578.1, -23.5) | 460.0 | 25.99 | 252.8 | 750.3 | 3/3 |

## Six remaining locations

Four locations exhausted all 80 frozen proposals without a valid capture:

| Grid | Bucket X/Y (mm) | Search requests | Setup refusals | Released misses |
|---|---|---:|---:|---:|
| 01-05 | (1050.0, -425.0) | 80 | 34 | 46 |
| 03-04 | (860.0, 25.0) | 80 | 49 | 31 |
| 03-05 | (1050.0, 25.0) | 80 | 57 | 23 |
| 04-06 | (1240.0, 250.0) | 80 | 76 | 4 |

Two locations, grid 03-00 at **(100, 25) mm** and grid 03-06 at
**(1240, 25) mm**, remain **fixed-base clearance blocked** and were not searched.
Their conservative clearance proxy is below 5 mm; this is not proof of mesh
overlap or physical impossibility. Bucket positions and robot bases stayed
unchanged. Exhausting a finite search likewise does not prove the four other
locations impossible.

## Provenance and operational boundary

- Frozen producer: `7bcba3980a1baf24d6ac2505e35189b95daead64`.
- [Draft PR #120](https://github.com/kadiru/sircus/pull/120), based on PR #119.
- Contract fingerprint: `a7b58fc50ce6d7b28b0d2a2ab045f4031bc77d3c70145c01a670ec134a003bf5`.
- Raw report SHA-256: `2d62f6b43b54210b83c7323b5e46f0ccdf0817f36b67fbe43e1aa7618fb099d9`.
- Audit proof SHA-256: `460e0e5496b54e9426fb170d64b9c0687813eb28e3b7310d3e8f212eda211569`.
- Independent audit: **141,041 checks passed**. This verifies evidence accounting;
  it does not make the three invalid-physics requests valid.
- Isolation: **31/31 checks passed**. The owned 4090 worker exited; the 4090 was
  idle at the final check, and the 4070 workload had the same process. Historical
  reports/checkpoints and the Isaac setup were preserved.

The additive `#fixed35Recovery` interface shows all 11 original failures,
confirmation groups, exact recipes, and the 24 preservation replays. The full
request ledger is collapsed by default. Raw traces and runtime logs remain
local, outside Git. No physical commands or service restarts were performed.
Further search, bucket perturbations, training and policy promotion are separate
next steps; this task does not authorize or perform them.
