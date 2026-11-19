# Fixed-plane, fixed-distance feasibility experiment

This simulation-only development search adds board height as a variable while
holding lateral geometry fixed. It is not a trained policy or a held-out test.

## Geometry

- Transparent conceptual plane: Y = −100 mm, normal toward −Y.
- Board center XY = (670, −100) mm. Ball release has exactly the same XY.
- Bucket center XY = (670, −500) mm: 400 mm perpendicular distance from the plane.
- Board height H: 300, 350, 400, 450, 500 mm above tabletop Z = 0.
- Vertical release gap G: 80, 120, 160, 200, 240 mm.
- Board tilt θ from horizontal: 12, 17, 22, 27, 32, 37, 42 degrees, facing the bucket.
- Release ball-center height is H + G + ball radius (37.5 mm, provisional).
  Dropper flange is another provisional 95 mm above ball center. These are
  contact/release coordinates, not coincident wrist positions.

The ball starts over the board center. Finite sphere size and tilt mean first
contact need not be exactly at the center. The side-view animation projects the
measured 3D ball trajectory; it is not a planner or an arm-motion animation.

Board/bucket/ball dimensions, rigid mount, mass and contact assumptions are
unchanged. The bucket's reported 175 mm height still has unresolved meaning;
175 mm interior depth plus an 8 mm base remain provisional simulation assumptions.
The conceptual plane is not a collision obstacle.

## Execution and scoring

`--plane-sweep` selects a separate finite experiment using the repaired endpoint
IK and measured settling controller from `BALANCED_PILOT.md`. Both arms initialize
at endpoint solutions; grasping and collision-checked travel are not performed.
Gravity compensation is model-based simulation feed-forward. Original release
pose/normal/velocity/drift checks remain mandatory. Unresolved IK and unstable
poses are recorded as rejected candidates, without release; neither is silently
excluded or claimed physically impossible.

The 5 × 5 × 7 Cartesian grid yields 175 throws. Success requires board contact and
rebound, whole-ball entry and a continuous second inside the bucket. Search rows
contain captures, misses, invalid causes and measured settling metrics. Up to three
successful combinations are selected by reward (stable ID tie break), then each
gets five fresh same-condition repeats and one separately reported rendered
replay. Repeats have no added noise; they check repeatability, not robustness.

## Run and inspect

Use the existing isolated Isaac Sim 5.1 / Isaac Lab 2.3.2 environment on the 4090:

```bash
export SIRCUS_ISAAC_SOURCE=/workspace/studies/YOUR_IMMUTABLE_SNAPSHOT
flock -n /workspace/scene.lock "$SIRCUS_ISAAC_SOURCE/python.sh" \
  "$SIRCUS_ISAAC_SOURCE/launch_scene.py" \
  --headless --enable_cameras --device cuda:0 --plane-sweep
```

The executed job has a 40-minute timeout and a 3 GiB free-space guard. It does
not train across distances, use the 4070, connect to hardware or alter Viam.

The run directory contains contract/source hashes, raw grid trials, selected
candidates, replay trajectories and a pre-release camera image. Raw runtime data
stays outside Git. `plane-sweep.json` is published atomically for the read-only
`#plane-sweep` interface, which shows the geometry, per-height heatmaps, repeats
and selectable measured trajectories. Historical experiment panels remain intact.
The setup image is for selected candidate 1 and shows the ball parked before
release; the side view supplies the measured flight separately.

Before a distance-conditioned policy, review the successful regions and invalid
boundaries at this distance. Do not interpret the grid capture fraction as policy
success or a guarantee for arbitrary tabletop bucket positions.

Completed study: [fixed-plane results](../../baselines/isaac-plane-sweep-20261031/README.md).
