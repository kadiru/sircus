# Moving the board to clear three missed shots

The preceding planner captured at 17 of 20 new bucket positions. At the three
missed positions, the ball bounced from the board into the bouncer arm's middle
segment. Changing joint configuration while keeping the board fixed mainly
flipped the wrist and did not move that obstruction. See
[the recorded diagnosis](HYBRID_MISS_CORRECTION.md).

This development search keeps those three bucket positions fixed and allows
the board center and heading to change. Board height, tilt and release gap
may also vary to preserve or recover the required flight range. The dropper
must remain above the board, and both arms retain the existing full-pose IK
and geometric path checks. This is bounded configuration search, not training
or a new unseen-position evaluation.

## Completed result — November 22, 2026

Moving the board and adjusting the shot recovered **two of the three targets**.
The original configurations still missed on all nine unchanged repeats.

| Bucket X, Y (mm) | New clear captures / search drops | Route | Fresh confirmation captures | Confirmed settings |
|---|---:|---|---:|---:|
| 948.4, −35.1 | 6/40 | Direct | 18/18 | 6 |
| 1195.0, 160.5 | 1/40 | Board → table → bucket | 3/3 | 1 |
| 1226.0, 183.4 | 0/40 | No qualifying capture | No repeats qualified | 0 |

Every qualifying search setting was repeated three additional times; all seven
passed 3/3. **Two additional arm-assisted captures** occurred at the first
target. Those retained-ball outcomes remain preserved separately. The frozen
clear-path confirmation rule did not repeat them, so they are not included
in the seven confirmed settings.

A setting includes board twist and the selected joint branch. Two confirmed
settings share their board center, tilt and gap but use different board twists;
the count is seven reproducible commands, not seven unrelated flight shapes.

The study completed **168 native releases**: 120 exploratory drops, nine
unchanged originals, 21 confirmations and 18 simulator controls. All 18
control checks passed: 12 expected captures and six expected noncaptures.
The independent audit passed **204,584 checks**. Leased 4090 time was
**640.767 seconds**, approximately 10.7 minutes.

This is selected development evidence. It does not change the earlier
17/20 policy evaluation, establish performance at new placements, or promote
a new planner. Existing successful settings remain preserved.

## Two confirmed examples

The table shows the selected successful setting at each recovered target.
Board coordinates and heights use the unchanged simulation frame. The gap
is the released ball's bottom Z minus board-center Z; ball-center release Z
also includes the ball radius. Heading is relative to facing the bucket.
Both examples use 180° board twist. Exact joint vectors and all seven
confirmed settings are in the [compact aggregate](../../baselines/mjlab-board-pose-recovery-20261122/summary.json).

| Bucket X, Y (mm) | Board X, Y (mm) | Board height (mm) | Tilt | Drop gap (mm) | Heading offset |
|---|---|---:|---:|---:|---:|
| 948.4, −35.1 | 636.9, 207.2 | 449.6 | 15.73° | 116.4 | −3.94° |
| 1195.0, 160.5 | 682.4, 2.1 | 461.9 | 25.47° | 109.4 | +6.95° |

The second example required a different board-arm joint solution found by
the multistart IK retry. These examples demonstrate complete shots at fixed
settings, not just numerically reachable endpoints.

## What still misses

At **(1226.0, 183.4) mm**, 26 of 40 flights contacted a robot. The other
14 cleared both arms and bounced on the table, but none entered the bucket
completely. The closest arm-free shot reached the bucket's side/rim; at its
closest recorded approach its center was **83.0 mm toward positive Y and
52.4 mm above the opening center**. That sample describes the recorded path;
it is not a whole-ball entry margin.

That shot used board center **(573.9, −8.4, 447.3) mm**, **20.34° tilt**,
**249.8 mm gap**, **+7.02° heading offset**, and **0° twist**. A useful next
local test is to reduce the positive heading offset while preserving this
translated board center and checking that the middle arm link stays clear.
The saved-trace diagnosis adds no new native trials.

## Protocol

1. Generate a deterministic menu around each original configuration. Use the
   existing approximate flight model to prioritize candidates; its predictions
   are not capture evidence.
2. Check arm reachability and paths on CPU, then predict ball-to-arm clearance.
   Preserve proposed, rejected, admitted and selected counts separately.
3. Freeze up to 40 admitted configurations per bucket before native execution.
   Run their ball flights on the RTX 4090 at 64 substeps.
4. Repeat every qualifying captured configuration three additional times.
   Keep those confirmation results separate from the exploratory captures.
5. Retest the three unchanged original configurations three times each and run
   matched simulator controls before and after the experiment. Audit saved
   native trajectories independently before publishing final counts.

The board center can move up to 150 mm toward/away from the bucket and 150 mm
sideways. Heading varies from −30° to +30° relative to facing the bucket;
0° and 180° board twist are tested. One family preserves the original board
height, tilt and gap. A second varies board height by −120 to +80 mm, tilt
from 12° to 44°, and drop gap from 80 to 550 mm, within the existing arm/action
bounds. Its 4,096 proposals per position use a seeded Halton sequence.

An outer wrist-reach bound removes obviously unreachable targets before
ranking. Up to 96 proposals from each family undergo existing full-pose IK
and geometric path checks. The 24 highest-ranked failures can receive a
second IK search with 16 starting points and 180 solver iterations. A failed
finite IK search is recorded as unresolved, not proof that a pose is
physically unreachable. Preparation is limited to 15 minutes elapsed per case,
with the three cases prepared in separate CPU processes.

Precise predicted arm and bucket-opening clearance ranks admitted candidates;
selection also spreads them across position, height, tilt, gap and heading
bins. Only one joint branch is selected for each identical Cartesian shot.
The final menu is frozen before any new native ball drop. Approximate flight
scores prioritize experiments; they do not substitute for capture evidence.

This run considered **14,344 unique parameter combinations** across the
three fixed bucket positions and made **72 additional IK retries** at
existing combinations. There were **648 setup checks**, producing
**216 admitted joint-solution records** (79/86/51 by target). Record counts
can include equivalent angle representations and are not counts of distinct
arm shapes. The actual physics experiment released **120 selected new shots**,
40 per target. The 14,344 CPU proposals were not 14,344 simulated ball drops.

A qualifying clear capture requires a board bounce, whole-ball entry and one
continuous second inside the bucket under the current direct/table-assisted
scorer. Robot-assisted retained outcomes remain separately identified. The
experiment does not relabel historical results or replace successful recipes.

## Scope and resources

The native experiment has a one-hour GPU budget and cooperatively acquires
the existing 4090 lease. The 4070 and Isaac workloads, source and setup remain
untouched. The simulation uses privileged geometry and holds the arms at
their checked endpoints during flight; geometric path checks do not establish
native driven-arm motion, camera perception or physical validation.

The new results panel is `#boardPoseRecovery`. Replays use recorded native ball
samples and the actual held arm poses, with normal-speed and quarter-speed
views. The interactive planner stays unchanged until a separately authorized
integration test. Raw observations, generated media and host runtime files
remain outside Git.

## Evidence and reproduction

Frozen computational producer:
`8d5a6aaca793b4975772672bc09d62cfb85ae974`.

```text
/home/kadir/sircus-mjlab/studies/board-pose-recovery-v2
/home/kadir/sircus-mjlab/output/board-pose-recovery-v2
/home/kadir/sircus-mjlab/output/board-pose-recovery-v2-audit.json
/home/kadir/sircus-mjlab/output/board-pose-recovery-v2-case19-diagnosis.json
```

| Artifact | SHA-256 |
|---|---|
| Report | `37a630386e4e2ee4b46a362174eb70de6666414b7a3010bc15b1b0a883b1b764` |
| Preparation | `cf3b558dc98368e45e679b51e89f4e3d2f007ed7645fe55abf8cd5afe5fae0a1` |
| Evidence manifest | `2e74bd8f04f8f1ad75768a5f5a1c37e5b3839e6bd23f24070553b31267f9501c` |
| Independent audit | `2961456b0b7b5d314a533af04cef8b6e40eaa5a7dcb38734ce3465335bf28563` |
| Remaining-target diagnosis | `5a7a92510ab01943693bbe2b8506eb36f8191783b633db37f1cad86083bbc4d2` |

To repeat, use the frozen source and pinned Ubuntu environment with a **new
output directory**. Run `board_pose_recovery_study` stages `prepare`, `freeze`,
then `run`, using the saved source/parent audit arguments. Hide CUDA during
preparation; select only the 4090 UUID and the cooperative GPU lock for native
execution. Do not overwrite the completed evidence. The CPU-only
`board_pose_recovery_audit` independently rebuilds screening and selection,
authenticates pose/path proofs, and rescores every saved native trajectory.

The 19 focused Python checks, 11 UI checks and producer GitHub CI passed. A separate broad
suite in the pinned Ubuntu environment remained incomplete after an unchanged
mock physical-trials test stalled; only that task-owned test process was
terminated. No hardware code or frozen computational source was changed to
bypass the stall.
