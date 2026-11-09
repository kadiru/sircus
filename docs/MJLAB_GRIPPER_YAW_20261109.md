# Bucket-facing gripper rotation — matched native64 experiment

November 9, 2026. Simulation only. This is a release-orientation experiment on
two inspected development cases, not model training or a new policy score.

## Hypothesis and frozen scope

The simulated open fingers lie on opposite sides of local Y. Their open passage
runs along local X. The existing dropper orientation is fixed downward with
rotation `diag(1,-1,-1)` for every bucket. Rotate it about world Z by
`atan2(bucket_y - board_y, bucket_x - board_x)` so local X points toward the
bucket. Do this before release; no wrist motion or withdrawal during flight is
introduced.

The two cases are `corrected-imitation-test-r0-c6-n1` and
`corrected-imitation-test-r1-c2-n0`. Their original corrected-policy flights
contacted a robot. The aggregate contact channel does not itself identify the
struck link; rendered geometry motivates the finger-clearance hypothesis.

| Case suffix | Bucket X/Y mm | Board X/Y/Z mm | Tilt | Nominal gap | New world yaw |
| --- | --- | --- | ---: | ---: | ---: |
| r0-c6-n1 | 1107.027 / -536.266 | 653.661 / -203.412 / 449.818 | 30.3307° | 75.650 mm | -36.2855° |
| r1-c2-n0 | 496.391 / -414.691 | 646.740 / -329.902 / 291.275 | 27.2919° | 75.836 mm | -150.5791° |

All five shot values, bucket position, original board-arm joints, opening,
geometry, initial ball state, timestep, scoring and physics stay fixed for
`bucket_facing` versus `corrected_exact`. Dropper IK must satisfy the new full
orientation without changing the release center. Its branch may differ; the
actual joint change is reported. Fresh strict endpoint and sampled setup proofs
are mandatory. No tolerance relaxation or fallback shot is allowed.

## Accounting and interpretation

The plan has 36 requested slots: each of the two cases receives three nominal
repeats of `current_exact` (successful earlier reference), `corrected_exact`
(colliding source), and `bucket_facing`, plus nine original controls before and
nine after. Each control block contains three direct captures, three
table-assisted captures and three expected robot-contact misses.

The earlier successful reference has different shot values; only the original
corrected versus rotated pair isolates commanded orientation. Refused setups
count as requests and have no fabricated flight. Three deterministic repeats
do not establish noise robustness, other-timestep consistency or all-table
generalization. Native arm endpoints remain held; grasp/release mechanics and
moving-arm transit are not learned or validated here.

The prior 72-slot midpoint experiment remains not run. This new user-authorized
rotation experiment uses a fresh output and contract. All successful recipes,
old scores and checkpoints remain unchanged. The 4070, Isaac setup, services
and physical hardware are outside this task's mutation scope.

## Results

Pending source freeze, actual-input preflight and native execution. No new
capture or clearance improvement is claimed by this protocol document.
