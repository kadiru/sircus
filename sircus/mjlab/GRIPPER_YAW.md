# Bucket-facing gripper rotation: two matched collision cases

This is a finite simulation experiment, not policy training or deployment. The
corrected imitation policy lost two previously successful shots to ball–robot
contact. We test the proposed gripper orientation without changing the shot
coordinates, board arm, scoring or physics.

The contexts are `corrected-imitation-test-r0-c6-n1` and
`corrected-imitation-test-r1-c2-n0`. Historical native force evidence establishes
robot contact; a particular finger/link is only named if reconstruction supports
it. Both cases are inspected development examples, not fresh policy TEST data.

## Fixed intervention

For each corrected recipe, let `yaw = atan2(bucket_y-board_y, bucket_x-board_x)`.
The requested world rotation of the downward dropper flange is:

```
Rz(yaw) @ diag(1, -1, -1)
```

This points the gripper's local +X finger passage toward the bucket. The chosen
direction is exact: no substitution by yaw + 180 degrees. The requested yaws are
approximately −36.286 degrees and −150.579 degrees. All five shot coordinates
(board X/Y/height, tilt and drop gap), the bucket, initial ball state and the
original corrected board-arm joint vector remain unchanged. Only the dropper's
orientation and the dropper joints needed to realize it change.

A separate opt-in resolver solves only dropper IK using the preserved source
joints as a warm start, 16 deterministic starts, 180 iterations and the full
rotation target. It ranks finite solutions by joint distance from the source,
then solver residual and branch ID; no flight outcome selects the branch. It
retains the original strict endpoint pose, 5 mm clearance and sampled setup-path
checks. It never searches a new board branch or weakens a guard to force release.
Finite setup refusals remain failed requests with no fabricated native flight.

The full rotation, source-row hash, original board joints and unchanged release
position are bound into a new `yaw-connections` cache key and proof. Original
five-coordinate proofs cannot silently authorize the new orientation. Native
reset joints are recorded; full orientation is checked by model FK of those
observed float32 joints. This is labelled FK of reset joint evidence, not an
independently observed camera pose or a newly sampled GPU quaternion. Both arms
remain held at admitted endpoints during flight; dynamic arm transit is not
being tested.

## Frozen plan and controls

Each case receives three nominal repeats of each variant:

- `current_exact`: successful historical current-policy recipe and original joints.
- `corrected_exact`: colliding corrected-policy recipe and original joints.
- `bucket_facing`: corrected recipe with only the explicit yaw intervention.

That gives 18 diagnostic slots, bracketed by 18 existing direct-capture,
table-assisted-capture and expected-miss controls: **36 slots total**. Successful
current recipes stay intact. Controls retain the original 17-source proof
namespace; original diagnostics retain the 19-source namespace; changed yaw
uses its own namespace and new proofs. All requests freeze before new outcomes.

Physics stays at **64 substeps** and the existing v3 scoring. Capture requires
board contact, whole-ball entry and one continuous second retained. Rim and table
assistance are allowed; robot, floor and unattributed contacts disallow capture.
The report separates setup refusal, robot contact, entry, retention, invalid
physics and accepted capture. No noise sweep, timestep change, PPO update or
policy promotion is included. Three nominal repeats do not establish robustness.

## Evidence, UI and finite execution

The report records exact target/measured yaw, original and observed board joints,
setup proofs, per-repeat outcomes and contact counts. An independent CPU audit
reconstructs the full requested rotation and native reset FK, verifies unchanged
coordinates/board branch and rescores saved native traces. UI/videos must label
saved-state held-endpoint replay accurately, preserve failed/refused variants,
and never infer capture just because contact disappeared.

Run `sircus.mjlab.gripper_yaw_pipeline` with authenticated `--parent-study`,
`--parent-audit`, `--diagnosis`, the committed `--rule`, `--reference`, `--assets`,
a fresh `--output` and existing `--gpu-lock`. The parent is
`corrected-imitation-v1`; the diagnosis is `imitation-regression-diagnosis-v1`.
The earlier midpoint correction was queued but did not run and is not a result.

Only the 4090 UUID is eligible. Caps are 900 seconds queued, 3,600 acquired
worker seconds and 6,300 total pipeline seconds, reserving 1,800 seconds for CPU
audit and 8 GiB of free storage. A deadline stops this task's own child only.
The 4070 workload, Isaac setup, services, hardware, historical evidence and model
weights remain unchanged. Raw traces, logs and host configuration stay out of Git.
