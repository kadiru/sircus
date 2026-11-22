# Check the candidate path before the reference path

This experiment changes **the order of arm-path checks**, not the learned
model or ball physics. It compares the current optimized checks with a
candidate-first version. Results remain pending until native drops and the
independent evidence audit finish. The live planner is unchanged.

## Why change the order?

A proposed shot specifies where the board and ball release should be.
Inverse kinematics finds the arms' joint angles for those endpoints. A
path check then asks whether the arms can move there from their home pose
without violating the 5 mm clearance requirement.

The earlier planner first finds a path to a previously known reference
shot, then connects that reference pose to the proposed shot. At the
previously timed-out bucket position, **(971.201, −285.388) mm**, the original
completed plan spent **85.73 seconds** searching for the reference path.
Its eventual successful shot used a different, direct home-to-shot path.
The previous speed improvements still could not return a checked shot at
this target within the 30-second decision limit.

The new order checks the actual candidate's direct path first. If that
path fails, the reference-path search remains available as a fallback.
“Direct” here describes the **arms' path**; it does not restrict the ball
to a direct board-to-bucket flight. Table-assisted captures still count.

## Controlled comparison

| Item | Protocol |
|---|---|
| Baseline | Optimized geometry checks; reference path searched first |
| Change | Same checks; candidate's direct home path checked first |
| Decision limit | 30 seconds per method and bucket position |
| Fresh evaluation | 20 new bucket positions; at most one drop per method at each |
| Development target | Old timeout tested separately, outside the fresh-position denominator |
| Frozen inputs | Same model, shot ordering, geometry and physics settings |
| Accepted paths | Same 5 mm clearance, detailed collisions, 0.5° motion samples in both directions |
| Native simulation | MuJoCo at 64 substeps, arms held at selected endpoints |
| Controls | Same saved controls before and after the test drops |

All selections are saved before native outcomes are observed. No-setup
positions stay in each method's denominator. Compare both capture counts
and full planning time; finding a feasible arm pose alone does not prove
a capture. The old target is development evidence, not a fresh test.

Capture requires a board rebound followed by one continuous second with
the whole ball inside the bucket. Direct and table-assisted routes count;
robot-assisted outcomes remain separate. This experiment does not train a
new model, use camera perception, or validate moving-arm execution.

## Evidence and interface

The driver is `python -m sircus.mjlab.direct_path_first_study`. It freezes
source hashes, input hashes, test coordinates and selections; records
separate reference-first and candidate-first evidence; then independently
audits the native results. The output directory and exact launch arguments
are recorded with the run. Raw observations, generated runtime files and
logs stay outside Git.

The additive UI section is **`#directPathFirst`**, sourced from
`direct_path_first_study`. It shows planning/drop progress while running.
Capture counts appear only when the report is complete, its independent
audit passed, and per-position outcomes reconcile with the totals. The
previously timed-out target has its own table. A missing native outcome
is shown as “Not run,” never a predicted capture.

The study uses only the RTX 4090 under the existing shared GPU lease.
The 4070 workload and Isaac setup are not modified. Publishing the results
panel does not switch the live planner or restart a service.
