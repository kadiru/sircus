# Two remaining bucket positions: arm-contact search

**Neither position was recovered in this bounded search.** We diagnosed the
previous contacts, tried different arm joint configurations and board/release
positions, then tested 32 distinct admitted configurations per target at the
unchanged 64 substeps.

| Bucket X, Y (mm) | New captures / drops | New drops hitting an arm | Arm-free misses | Unchanged original |
|---|---:|---:|---:|---:|
| 1192.994, 147.423 | 0/32 | 30/32 | 2 | 0/3 |
| 1127.555, −38.348 | 0/32 | 32/32 | 0 | 0/3 |

All 64 new drops were valid simulated misses. None retained the ball through an
arm-assisted route either. No new setting qualified for three confirmation
repeats. This does not establish that either position is impossible.

The 40 previous search flights struck the board arm: at the first target all
20 first contacts matched link 3; at the second, 16 matched link 3, three link 4
and one link 2. Native force samples establish contact and timing. Link identity
comes from collision-geometry reconstruction at the saved ball and joint poses,
not a native per-link contact label.

The same reconstruction on the **new** drops passed 983 checks. At the first
target, link 3 was implicated in all 30 robot-contact shots. At the second,
27 first contacts matched link 3, three link 2 and two link 4.

The two arm-free shots had the same board center **(680.34, −11.04, 461.87) mm**,
tilt **25.47°** and drop gap **109.44 mm**, with board twists **135° and −135°**.
Both hit the board twice, at approximately 0.145 and 0.337 seconds. When their
centers descended through rim height, they were **142.5 and 146.1 mm short** of
the bucket center, with **26.8 and 68.8 mm lateral error**. Neither ball entered.
These are native trajectory measurements, not predicted capture margins.
Exact commands, measurement definitions and trace hashes are in
[followup-diagnosis.json](followup-diagnosis.json).

**Next proposed experiment, not run:** start from the clearer 135° board-twist
configuration and increase the drop gap, followed by modest tilt changes.
Both changes require renewed reach/path and rebound-clearance checks. Extra
energy could extend the flight or clear the board's second impact, but could
also put link 3 back in the ball's path. The second position still needs a
configuration that first clears that arm; no successful correction is claimed.

## What changed

Six diverse previously feasible commands per target supplied starting points.
The search varied joint branches at fixed Cartesian poses, board/release XY
along and across the direction to the bucket, and board twist around its normal.
It prioritized changes to the board arm's proximal joints rather than only wrist
flips. Board height, tilt, drop gap and the bucket remained fixed for each source.
All alternatives retained the same full-pose, endpoint and sampled-path checks.

Two CPU workers each ran for at most 900 seconds. They attempted 50 and 78
pose/path checks, producing 115 and 213 admitted joint-solution records before
deduplication and selection. These are preparation records, not ball drops.
Each target then used 32 distinct commands, frozen before any new native result.

| Selected alternative | First target | Second target |
|---|---:|---:|
| Same board/release pose, different joints | 7 | 7 |
| Shift along bucket direction | 5 | 2 |
| Shift across bucket direction | 10 | 12 |
| Twist board around its normal | 10 | 11 |

## Validation and preservation

- **97 native releases:** 64 search, six unchanged originals, nine preservation
  drops and 18 simulator controls. No setup rejection or invalid physics in the
  selected search drops.
- One unchanged successful setting from each of the three previously recovered
  positions passed three repeats: **9/9 captures**. The full earlier archive and
  its 42 confirmed settings were not overwritten or relabeled.
- Both nine-drop control blocks passed: each contained three direct captures,
  three table-assisted captures and three expected noncaptures.
- Independent preflight: **7,317 checks**. Final evidence audit: **26,471 checks**.
- Leased RTX 4090 time: **493.481 seconds**. The RTX 4070 workload and Isaac setup
  were untouched; no service was restarted.

Exact selected settings, joint angles, per-drop outcomes and hash provenance are
in [summary.json](summary.json). Computational producer:
`48f41dd66be3efaa24823eb68d0e2816e84521c0`. Raw traces, calibration and runtime
files remain outside Git. Two failed CPU-only preparations are preserved
separately; neither ran new native drops.

Results are published in the existing UI at `#remainingLinkClearance`.
The live planner and all historical video entries remain unchanged. This is
development search at two already-known failures, not retraining or a new
generalization score. Native flights hold arms at their checked endpoints;
sampled setup paths do not establish driven-arm or physical validation.

See [the protocol and reproduction instructions](../../sircus/mjlab/REMAINING_LINK_CLEARANCE.md).
