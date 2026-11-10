# Farther shot: fixed bucket-facing gripper, tilt and gap search

This is a finite, simulation-only development experiment. It tunes the farther
`corrected-imitation-test-r0-c6-n1` layout after the yaw experiment removed robot
contact but produced no capture in three repeats. It does not retrain or promote
a model, change historical results, or establish performance on unseen positions.

## Fixed scene and intervention

- Board center: **(653.661200, -203.412474, 449.818340) mm**.
- Bucket center X/Y: **(1107.026526, -536.265843) mm**.
- Horizontal board-to-bucket distance: **562.434 mm**.
- Dropper world yaw: **-36.285503 degrees**, using
  `Rz(atan2(bucket_y-board_y,bucket_x-board_x)) @ diag(1,-1,-1)`.
- Physics: existing native **64 substeps**, unchanged contact model and v3 score.

The gripper passage stays aimed toward the bucket. Only board tilt and nominal
drop gap vary. Board center, bucket, geometry, ball initial velocity and all
physics/scoring settings stay fixed. New board joints are necessary to realize
different tilts; new dropper joints realize the different release heights.
The original strict endpoint, full-pose and sampled setup-path guards remain.
Failed finite IK searches are recorded as setup rejections, not impossible poses.

Nominal drop gap means ball-bottom Z minus board-center Z. The ball radius is
37.5 mm, so ball-center release height is board-center height + gap + 37.5 mm.
It is not the gripper flange height or clearance measured normal to the tilted
board surface. Table-contact labels alone do not establish a causal table
rebound; the saved trajectory and contact timings supply that evidence.

## Frozen search and confirmation

The 49 candidates are the Cartesian product of:

| Parameter | Values |
| --- | --- |
| Board tilt from horizontal | 24, 26, 28, 30.33068546648235, 32, 34, 36 degrees |
| Gap offset from 75.64961424222355 mm | 0, 20, 40, 60, 90, 120, 160 mm |

Each candidate gets one discovery request. Every valid discovery capture gets
three fresh nominal repeats, using the same admitted joints. The rule for
selecting repeats freezes before discovery, and every repeat remains in the
report. A capture is not declared reliable merely because one search shot scored.
No additional adaptive grid, noise sweep, 128-substep test or PPO follows this run.

There are nine exact reference requests: three farther rotated baseline misses,
three farther successful current-policy recipes, and three nearer successful
bucket-facing recipes. Eighteen original direct/table-assisted/robot-contact
controls bracket execution. Thus the maximum is **223 requests**: 49 discovery,
147 possible repeats, nine references and 18 formal controls. Rejected requests
and actual releases are reported separately. Successful parent recipes are
preserved byte-for-byte; their different shot coordinates are clearly labelled.

Capture requires a board bounce, whole-ball entry and one continuous second
inside. Table and rim assistance count. Robot, floor and unattributed contacts
invalidate the route. Clearing the gripper is not itself a capture. The UI
separates entry, retention, contacts, setup refusals and accepted capture.

## Execution and evidence

Only the RTX 4090 is eligible, through the existing cooperative GPU lock. The
finite limits are 900 seconds queued, 3,600 acquired worker seconds and 6,300
pipeline seconds, including a 1,800-second CPU-audit reserve. Maintain at least
8 GiB free storage. Never stop another workload to acquire the GPU. Preserve the
4070, Isaac configuration, historical outputs, model weights and running viewer.

The independent audit reconstructs candidate and repeat selection, checks new
full-orientation proofs, verifies board center/normal and dropper pose from
native reset joints, and recomputes capture from saved trajectories. Native
flights hold admitted arm endpoints; neither moving-arm transit nor real
grasp/release is validated. Movies replay saved native states with this limit
explicitly shown. Source is committed; raw trajectories, runtime settings,
machine paths, logs and movie files stay outside Git.

The other Isaac conversation's applicable lesson is retained: geometry admission
precedes useful flight learning, with separate request, release and compute
accounting. This experiment changes no Isaac benchmark, controller or checkpoint.
