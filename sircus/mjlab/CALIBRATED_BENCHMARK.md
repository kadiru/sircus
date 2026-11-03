# Calibrated current-ball sweep versus PPO

This additive v2 experiment corrects two limitations in the frozen
[v1 benchmark](SWEEP_RL_BENCHMARK.md): a 1 µm containment test that confused
soft-contact penetration with capture loss, and a single nominal source/park
check that could reject a bucket layout before checking the requested action.
V1 artifacts, definitions and observations remain unchanged. The two versions
use different scoring and fresh evaluation coordinates; their percentages are
not a controlled before/after comparison.

## Capture controls before policy comparison

The exact compiled ball and 64-wedge open bucket are isolated from the robot
scene, retaining their contact materials. Fixed controls include free drops,
wall-face/vertex impacts at several heights, rim contacts, translated buckets
and outside misses. The fit grid has57 cases. Its predeclared contact-envelope
rule is `1.25 × maximum observed fit overlap + 8 µm`, capped below95% of the
minimum radial wall thickness and half the base thickness. Geometry precision
(2 µm) is separate from soft-contact tolerance. This calibrates the scorer to
this simulated solver and current ball; it is not physical material calibration.

Verification uses72 different controls at the benchmark timestep
(1/15,360 s), the finer timestep (1/30,720 s), and a reversed world ordering at
the benchmark timestep. All must pass the frozen fit envelope without refitting.
A coarser1/7,680 s run is diagnostic only. An independent CPU audit reconstructs
compiler-derived cavity planes and actual contact-pair distances using
`mj_geomDistance`, checks source/data hashes and timestep/options, and records
the collision observation convention. Contact distances are pre-integration
observations for interval[k,k+1]; position[k+1] is the post-integration state.

The new scorer and a separate implementation independently require:

- A measured board rebound, followed by downward passage through the open mouth.
- Whole-ball entry and one continuous observed second inside. No preceding
  interval is credited; partial exits reset retention. Initial-inside and
  side-wall arrivals cannot create a new valid entry.
- Compiled cavity geometry with the verified contact envelope at side/floor
  boundaries. The envelope does not relax the upper rim or permit the sphere
  centre through a wall. Contacts beyond the frozen envelope are invalid physics
  and count as failed native attempts.
- No intervening table, floor, robot or unattributed contact for a direct route.
  Rim-assisted captures count; clean capture is a separate quality label.

A completed success is terminal. Later trace events remain diagnostic evidence,
but do not revoke the completed shot. Every native state and contact interval is
retained locally for independent rescoring.

## Setup admission

One new canonical initial park is checked against historical coordinates and an
independent geometry grid. It defines a new simulator initialization, not an
asserted path from the old colliding park. Real fixed-base/bucket conflicts stay
ineligible and remain in all requested-trial denominators.

Before either method predicts, the same deterministic resolver checks the17
shared historical geometric recipes, nearest original bucket first, then
bounded declared geometric offsets, with at most64 attempts. It uses no ball
flight labels. Original source and derived anchor provenance are both retained.
Its time and memory are charged. Thus inference includes geometry search;
**frozen policy evaluation performs no ball-flight search**.

The old wrist-sphere reach estimate is diagnostic. Actual IK, independent/native
collision checks, the existing>150 mm board-to-bucket task distance,5 mm
clearance, and forward/reverse serialized paths sampled at≤0.5° remain required.
An unresolved nominal anchor does not veto a potentially valid actual action.
After prediction the exact requested action is checked; it is never projected,
clipped or replaced by a successful fallback. Finite search failures are labeled
unresolved, not proofs of global impossibility.

## Matched learning design

Both methods receive the same current ball,17 source recipes,35 TRAIN bucket
coordinates, paired training schedule, action limits, residual scales and reward.
The five action values are board centreX/Y/height, bucket-facing board tilt, and
release gap above the board. The board faces the bucket; the dropper is aligned
over the bounce point. This is a one-shot contextual decision task, not learned
joint torque control, grasping, transit or compliant impact response.

Three seeds each spend1,024 TRAIN requests per method. Rejected setups consume
budget. Sweep builds a guided residual library; PPO updates an actual continuous
residual actor/critic. Validation at0/256/512/1,024 chooses a frozen artifact.
Every worker completes its full training budget even if the initial artifact
is selected. Twelve validation and24 final-test positions are generated before
training, excluding every coordinate from the earlier71-position benchmark and
each other by at least5 mm. There are three paired nominal/perturbed requests
per evaluation position. Unadmitted layouts are retained, never resampled away.

Capture-rate curves use validation only. Final tests use frozen selected
artifacts without a new flight search or test-driven selection. Training,
validation, final tests, engineering controls and shared historical recipe cost
are accounted separately. Historical acquisition cost is incompletely known.

The benchmark records requested and native attempts, capture/clean/assisted
counts, measured wall and CPU time, geometry/path/native/postprocessing time,
GPU reservation, lifetime RSS and sampled process VRAM, artifact size and query
cost. Native rollout wall time is not GPU kernel-active time. Memory includes
the simulator and evaluation. Disks retain dense traces outside Git; a compact
checksummed publication includes selected policies and independent audit reports.
The internal resource key `admission_wall_s` means the full adapter pipeline wall
time, including native work and evidence I/O. It overlaps other timing fields;
it is not geometry-only cost and these components must not be summed together.

Before a long run, the complete six-worker CPU fixture must pass the real
top-level audit and exporter. It is explicitly synthetic and cannot be published
as native results. A separate TRAIN-only native smoke requires source captures,
a deliberate rejection, a real discarded PPO update, and an independent audit.
The long controller verifies this evidence and frozen source/config identities.

## Reproduction

Use the pinned environment in[requirements.txt](requirements.txt) with native
UF850 assets and a matching corrected reference directory. Run
`python -m sircus.mjlab.capture_calibration --help`, then
`python -m sircus.mjlab.verify_capture_calibration --help` for the isolated controls.
Only after those reports pass, use
`python -m sircus.mjlab.calibrated_benchmark_smoke --help` and independently audit
its controls, then `python -m sircus.mjlab.calibrated_benchmark_study --help`.
Each output directory must be new. The controller supports source-identical
resume and will not overwrite a different study. Select only the authorized4090
by GPU UUID, use the shared read-only-opened GPU lock, and reserve sufficient
root-filesystem space for dense observations. No runtime, hardware or cloud
configuration is changed by source synchronization.

The read-only v2 interface and compact result bundle must be bound to the passed
independent audit. An uncompleted or failed audit is never displayed as a verified
comparison. Existing v1/planner services and policies remain separate.
