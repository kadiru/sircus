# Clearing the rebound path at two remaining bucket positions

The nearby-shot transfer study recovered three of five development misses.
Its 42 successful settings each passed three fresh confirmations. At the two
remaining bucket positions, every one of the 20 tested flights hit a robot.
This follow-up diagnoses those contacts and searches for different arm setups;
it does not retrain a policy or change the interactive planner.

| Target | Bucket X, Y (mm) |
|---|---:|
| 09 | 1192.994, 147.423 |
| 19 | 1127.555, −38.348 |

CPU reconstruction of all 40 previous search flights identified a unique
overlapping collider at each measured first robot-contact sample. At target 09,
all 20 first contacts were consistent with board-arm link 3 (0.390–0.520 s after
release). At target 19, 16 were consistent with link 3, three with link 4 and one
with link 2 (0.381–0.512 s). None of these first contacts implicated the dropper.
This diagnosis passed 621 source, trajectory and geometry checks. Geometry
names are reconstructed evidence, not native logged per-link contact IDs.

## Question and experiment

1. Read the immutable previous trajectories and robot-contact force samples.
   Reconstruct the robot collision geometry at the recorded joint positions
   to identify the links consistent with each first contact. Native force
   establishes that a robot contact occurred; the reconstructed geometry
   identifies the likely link. Keep those two evidence sources distinct.
2. Start from previously checked commands at each target. First vary joint
   configurations while preserving the Cartesian board and release geometry.
   Prioritize changes to the proximal arm joints, because flipping only the
   wrist can leave an obstructing middle link in the same place.
3. Also consider bounded shifts of the board and release geometry, with the
   bucket fixed. Recheck full arm poses, endpoint clearances and sampled setup
   paths for every candidate. A checked setup does not certify a clear ball
   flight; approximate flight predictions only rank the experiments.
4. Freeze up to 32 distinct admitted commands per target. Test them at the
   unchanged calibrated 64 substeps. Repeat each valid direct or table-assisted
   capture three additional times. Preserve robot-assisted retention and
   invalid-physics outcomes separately rather than hiding them as ordinary
   misses or counting them as direct captures.
5. Run three unchanged-original repeats per target, nine preservation drops
   (one exact successful setting from each of the three recovered positions,
   three repeats each), and nine matched controls before and after the tests.
   Maximum: 64 search + 192 confirmations + 6 originals + 9 preservation +
   18 controls = **289 releases**, within one hour of leased RTX 4090 time.
6. Independently audit the frozen requests, source hashes, setup proofs,
   saved native trajectories, one-second retention, routes and complete
   denominators before publishing results.

## Boundaries

- Previous settings, raw studies, scores and model checkpoints remain immutable.
- CPU preparation hides both GPUs. Native tests acquire the shared GPU lease
  and expose only the RTX 4090. An occupied GPU is waited for, never cleared by
  stopping somebody else's process. The 4070 and Isaac workload are untouched.
- These are two known development positions, not unseen-position evaluation.
- Ball dynamics are native simulation with arms held at checked endpoints.
  Sampled setup-path checks do not establish driven-arm execution, real robot
  behavior, perception performance or robustness to noise or other timesteps.
- UI publication adds a read-only results section, `#remainingLinkClearance`.
  There is no live planner promotion, service restart or physical action.

## Frozen menu and preparation limits

Six diverse source commands per target each receive two deterministic 16-start
IK searches. Changed-geometry proposals use radial XY offsets of ±40/80/120 mm,
lateral offsets of ±50/100/150 mm and board twists of ±20/45/180 degrees.
Board height, tilt and drop gap remain fixed. A board twist rotates the actual
board geometry around its normal; it is not merely a joint change at an identical
board pose. Preparation prioritizes board-arm joints 1–3, then dropper joints.

Two CPU workers each receive at most 96 full checks and 900 seconds. Checks
interleave fixed-pose, radial, lateral and twist proposals across the six sources.
A proposal receives at most 30 seconds; unfinished admissions are discarded and
the engine is reconstructed before continuing. Nested reference-path searches
retain their original IK options. Each
proposal may admit up to four joint branches. Selection omits joint differences
under 0.01 rad for an otherwise identical Cartesian command; fixed-command
alternatives require at least 0.05 rad difference from their source.

## Reproduction

Use the pinned mjlab environment, read-only original assets and the audited
`nearby-shot-transfer-v1` source plus its independent audit. New raw outputs,
calibration and logs belong outside Git.

```text
python -m sircus.mjlab.link_clearance_study
  --source-study <nearby-shot-transfer-v1> --source-audit <its audit>
  --diagnosis <independent link diagnosis JSON>
  --parent-study <fixed35-sixhour-v2> --parent-audit <its audit>
  --reference <corrected-grid-v1> --assets <original assets>
  --output <new empty output> --gpu-lock <shared scene.lock>
  --stage prepare|freeze|run
```

CPU preparation requires `CUDA_VISIBLE_DEVICES=''` and one BLAS/Torch thread per
process. After freezing, run the independent preparation audit and supply its
receipt via `--preflight-audit` for native execution. Only the authorized 4090 UUID
may be exposed. A final audit is required before publication.


Results will be appended after the independent audit.

## Preparation correction

The first CPU-only preparation exposed a duplicate `preserve_known` keyword
when the alternative-IK wrapper intercepted an existing reference-path solver.
No native drops ran. Its partial files and failure receipt remain separate.
The corrected wrapper delegates reference calls unchanged and customizes only
the new candidate solve. The new preparation interleaves joint-only and
geometry changes, bounds each proposal to 30 seconds, and rebuilds the checking
engine after a timed-out proposal. Timed-out checks are unresolved, not evidence
that a pose is impossible. Complete endpoint/path proofs remain required.
