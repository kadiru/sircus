# Bounded demonstration recovery — November 9, 2026

The bounded search and actor-only imitation study completed **970 requests** in **96.8 worker minutes**. Independent CPU verification passed **283,251 checks**; all original direct-capture, table-assisted and expected-miss controls passed.

All four frozen models were evaluated once on the same **70 fresh bucket positions**, two per cell in a 7×5 table grid. Every setup rejection remains in the denominator.

| Frozen model | Captures / 70 | Capture rate | Setup rejections | Flight misses | Invalid physics | Clean captures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Recovery imitation | 37/70 | 52.9% | 16 | 17 | 0 | 14 |
| Parent imitation | 34/70 | 48.6% | 20 | 14 | 2 | 10 |
| Parent selected PPO | 34/70 | 48.6% | 18 | 17 | 1 | 9 |
| Strongest previous PPO weights | 31/70 | 44.3% | 22 | 17 | 0 | 8 |

Flight misses above exclude invalid-physics attempts. Both remain failures in the denominator; numerical/contact-domain failures are reported separately.

Invalid-physics TEST cases:

- Parent imitation / `recovery-test-r1-c5-n1`: `through_wall_after_entry`. Counted as failure, without retry.
- Parent imitation / `recovery-test-r2-c5-n1`: `through_wall_after_entry`. Counted as failure, without retry.
- Parent selected PPO / `recovery-test-r2-c5-n1`: `through_wall_after_entry`. Counted as failure, without retry.

Recovery imitation also missed both affected bucket positions. **None of its gained captures overlaps a reference invalid-physics failure**; the all70 comparison remains unchanged.

Matched outcomes show what changed at individual bucket positions:

| Reference for recovery imitation | Both capture | Recovery gains | Recovery losses | Neither captures |
| --- | ---: | ---: | ---: | ---: |
| Parent imitation | 29 | 8 | 5 | 28 |
| Parent selected PPO | 28 | 9 | 6 | 27 |
| Strongest previous PPO weights | 29 | 8 | 2 | 31 |

These are observed counts from one nominal attempt per model and position. The comparison does not measure repeated-shot reliability or guarantee an advantage on other layouts. No policy was automatically promoted.

## What changed

The former 70-position TEST set is now development history. Of the selected PPO’s 42 failures, **39** were eligible for bounded recovery search; **three fixed-base clearance failures** were explicitly skipped because changing the permitted shot coordinates cannot move the robot bases. Skipping them does not establish global impossibility.

- **624/624 search requests** completed: 16 full rounds over 39 layouts and 0 requests in the next round. The 16 proposals per layout were frozen before outcomes.
- Search stays inside the same five-coordinate policy support: board X/Y ±40 mm, height ±50 mm, tilt ±8°, and drop gap ±120 mm around the geometry anchor. Proposals vary height/tilt for setup refusals and gap/tilt/height for flight misses; previously observed alternate-model captures can seed a proposal.
- **48 fresh nominal repeats** tested selected discoveries. **24 new layouts** met the required two successful repeats out of two and supplied one teacher each.
- The actor dataset contains **92 teachers**: the original 68 TRAIN captures plus those confirmed development discoveries. The critic and exploration parameters remain unchanged.
- Recovery imitation starts from the parent imitation model and performs **800 full-batch actor-only Adam steps**, learning rate 0.001. This experiment performs no PPO training.
- All **280 TEST predictions** were frozen before any new TEST outcomes. There was no ball-flight search or outcome-based retry at inference; all methods share the unchanged 19-pose geometry prior.

Confirmed recoveries by the original failure category: action_bounds_rejected: **2**, finite_ik_unresolved: **5**, released_miss: **16**, strict_endpoint_clearance_or_pose_rejected: **1**.

Selected development recipes below each captured in both fresh nominal repeats. One example per original failure category is shown; all selected recipes and provenance are in the summary.

| Original failure | Bucket X, Y (mm) | Board height (mm) | Tilt (°) | Drop gap (mm) | Actual release-center height (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| action_bounds_rejected | 554.9, -162.6 | 272.0 | 24.09 | 204.4 | 513.8 |
| finite_ik_unresolved | 879.2, -146.4 | 420.0 | 20.00 | 200.0 | 657.5 |
| released_miss | 116.9, -596.7 | 340.1 | 30.64 | 349.4 | 727.0 |
| strict_endpoint_clearance_or_pose_rejected | 543.5, 140.0 | 314.1 | 33.53 | 101.3 | 453.0 |

The frozen tabletop surface is at simulation Z = 0 m; heights above the table were calculated by subtracting that surface datum from world Z. Tilt is measured from horizontal. Drop gap is requested release-center height minus board-center height minus ball radius; it is not the release-center height itself. Actual release-center height comes from the independently audited native time-zero observation.

## Interpretation and limits

A capture requires a measured board rebound, whole-ball entry and one continuous second inside. Rim and table-assisted routes count; robot, floor and unattributed contacts do not. These rules and the calibrated containment tolerance were unchanged.

New positions are at least 5 mm from coordinates in the frozen historical inventory, without filtering on IK or outcomes. The recovery actor has additional development demonstrations, so this is not an equal-data training-budget comparison. Historical results retain their original split definitions.

This is privileged-geometry, held-endpoint simulation at **64 substeps**. It does not train perception, grasping, carrying or arm transit. Finite IK rejection is not proof every shot at that bucket is unreachable. Two nominal repeat captures do not establish noise robustness; no 128-substep or hardware qualification is claimed.

## Evidence

- [Compact results, paired gains/losses, recovered settings and provenance](summary.json)
- [All 70 positions and four model outcomes](test-positions.csv)
- [Frozen recovery protocol](../../sircus/mjlab/COVERAGE_RECOVERY.md)

Frozen producer: `522409306b836e39d2108be91c5d57fcf1fd484b`.
Report SHA-256: `6a255223820656059fe16a5de2944ba2593f7ac3603b287f4955979cd4677504`.
Independent audit SHA-256: `e2ba653792b37d302425ba4af6cb22460c3d5b426274b3c1b3cbce270a857a9b`.
Contract fingerprint: `f54c341eba5258b427e4e1eb05a1b2e93bf76518d9f94a2fe8cf6bad744f6082`.

Raw trajectories, checkpoints, queues, host configuration and logs remain outside Git. The compact files are derived only after the complete independent proof passes.
