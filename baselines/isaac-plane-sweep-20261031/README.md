# Fixed-plane feasibility: bucket 400 mm away

Completed on RTX 4090, October 31, 2026 (UTC run `plane-sweep-20261101T020705Z`).
The finite experiment took 350.9 seconds and exited. No cross-distance training,
hardware, Viam, or 4070 changes occurred.

[Method](../../sircus/isaac/PLANE_SWEEP.md) · [Verified aggregates and hashes](results.json)

Board/release XY stays at (670, −100) mm; the bucket stays at (670, −500) mm.
Only board-center height H, vertical gap G, and bucket-facing tilt vary. The ball
center starts at H + G + 37.5 mm; the dropper flange is another 95 mm higher.

## Grid results

175 combinations: **25 captures, 115 misses, 35 dropper endpoint IK rejections**.
Every released grid shot passed the measured settling gate. Rejected endpoints
remain in the denominator; they are unresolved under the solver/pose constraints,
not proofs of global physical impossibility.

| Board height | Captures / searched | Rejected |
|---|---:|---:|
| 300 mm | 2/35 | 0 |
| 350 mm | 6/35 | 0 |
| 400 mm | 10/35 | 0 |
| 450 mm | 5/35 | 14 |
| 500 mm | 2/35 | 21 |

This is a search-coverage table, not a learned-policy success rate. Height helps
some shots but higher releases also encounter the dropper's endpoint constraints.
No single parameter is monotonically better across the tested grid.

## Selected combinations and fresh repeatability

| Board H | Gap G | Tilt from horizontal | Fresh repeats | Separate rendered replay |
|---|---|---|---|---|
| 450 mm | 80 mm | 32° | 5/5 captures | capture |
| 350 mm | 120 mm | 22° | 5/5 captures | capture |
| 400 mm | 240 mm | 27° | 5/5 captures | capture |

Selection used successful grid candidates ranked by reward with deterministic ID
ordering for ties. These are same-condition repetitions without injected noise,
not independent layout generalization tests. The result demonstrates multiple
working H/G/tilt combinations in this simulated fixed-distance setup.

Success requires measured board contact/rebound, whole-ball entry and one
continuous second inside. Rim-assisted captures are allowed. The 140 released
grid shots had maximum board-center error 0.0932 mm and normal error 0.00786°;
max joint speed was below 1.22e-8 rad/s. These are ideal simulated readbacks with
model-based gravity compensation, not physical calibration accuracy.

## Interface and evidence

`#plane-sweep` displays an actual Isaac image for selected setup 1, a side-view
projection of each fresh replay's measured trajectory, per-height heatmaps and
repeat counts. The image shows the ball parked before release; the animated
trajectory shows release/flight separately. The arm endpoints are simulated,
but travel/grasp trajectories have not been learned or collision-validated.

Raw trials, joint state, camera/depth observations and trajectories remain outside
Git in `~/sircus-isaac/output/plane-sweep-20261101T020705Z/` on Ubuntu, with a local
ignored evidence copy. Immutable runtime source is in
`~/sircus-isaac/studies/plane-sweep-v1/`. Source and artifact hashes are recorded.
The summary verifier checked complete grid coverage, capture prerequisites,
accepted settling before release, selected-repeat completeness and artifact hashes.

The provisional rigid mount, mass/contact properties and unresolved meaning of
the bucket's reported 175 mm height are unchanged. No learned checkpoint was
created, and prior experiment baselines were preserved.

Next: review these feasible regions, then explicitly authorize a distance sweep
using H/G/tilt, with feasibility boundaries and held-out distances defined before
training. This study does not claim the bucket is beyond the dropper's reach.

Validation: 58 Isaac tests, five diagnostic/pilot/plane DOM tests, the full project
mock/numerical suite, Ruff and repository scan passed. Initial full-suite invocation
used system Python without Viam; rerunning with the established project venv passed.
Completed interface, selectable replay controls and setup image were inspected.
