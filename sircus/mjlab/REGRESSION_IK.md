# Two IK regressions: frozen feasible-pose comparison

This is a bounded, simulation-only **66-request development experiment**, not
another policy-training run or a change to the IK algorithm. The two inspected
contexts are `recovery-test-r1-c1-n1` and `recovery-test-r2-c4-n0`. Their original
recovery-imitation actions failed finite IK admission while their full parent
recipes captured. Successful recipes, historical scores and model weights
remain intact; the recovery policy's original **37/70** is unchanged.

## Prespecified actions

For each case, start from its original recovery vector
`[board_x_m, board_y_m, board_height_m, tilt_deg, drop_gap_m]` and its matched
successful parent vector. Bucket X/Y stays fixed. The following eight variants
are frozen before any new outcomes:

| Variant | Exact action definition |
| --- | --- |
| `parent_exact` | All five parent coordinates and the original admitted branch |
| `recovery_exact` | All five recovery coordinates and its original cached refusal |
| `parent_xy` | Restore only board X and Y to their parent values |
| `parent_height_gap` | Restore board height and drop gap to their parent values |
| `parent_tilt` | Restore only tilt to its parent value |
| `blend25` | `recovery + 0.25 * (parent - recovery)` for all five coordinates |
| `blend50` | `recovery + 0.50 * (parent - recovery)` for all five coordinates |
| `blend75` | `recovery + 0.75 * (parent - recovery)` for all five coordinates |

Restoring height and gap jointly restores the parent's absolute requested ball
release-center height. XY restoration can change reach and target-facing yaw;
interpolation changes several pose coordinates together. These are explicit
local feasibility proposals, not isolated causal estimates of individual
parameters. Every exact novel action receives strict admission with unchanged
solver seeds, tolerances, endpoint checks and sampled setup-path checks.
No placeholder zero joints are ever released after a refusal.

| Context | Bucket X/Y mm | Recipe | Board X/Y/height mm | Tilt ° | Gap mm |
| --- | --- | --- | --- | ---: | ---: |
| `r1-c1-n1` | 378.083 / −438.148 | Parent | 635.421 / −392.848 / 282.162 | 31.622 | 211.240 |
| `r1-c1-n1` | same | Recovery | 641.313 / −416.373 / 257.834 | 32.196 | 236.932 |
| `r2-c4-n0` | 898.306 / −244.167 | Parent | 579.343 / −59.661 / 460.706 | 24.096 | 212.200 |
| `r2-c4-n0` | same | Recovery | 565.650 / −62.266 / 478.303 | 25.631 | 182.232 |

Rounded values are for reading. Full-precision actions are derived from
hash-authenticated original rows. The [literal rule](regression-ik-correction.json)
binds the original CPU diagnosis bytes; no failure or successful recipe is
relabelled or overwritten.

## Requests, controls and repetition

1. Nine original formal controls: three direct captures, three table-assisted
   captures and three expected misses.
2. Two contexts × eight variants × three nominal requests: **48 diagnostic
   requests**, comprising six parent controls, six historical refusal slots and
   36 fresh-action requests across twelve distinct changed configurations.
3. Repeat the same nine formal controls after the diagnostic phase.

All **66 slots** are frozen before outcomes. Every variant receives all three
requests; there is no best-flight selection, replacement, adaptive retry or
extra budget triggered by a result. Original refusals remain cached decisions,
not three independent IK searches. Each novel exact action has one strict proof
reused across its three requests. If admitted, those requests are fresh native
releases; if rejected, all remain unflown requests in the denominator. Nominal
repeats do not qualify robustness to noise or changes in timestep.

Parent captures are positive diagnostic comparisons, not a hard gate hiding an
unstable recipe. The two original formal control blocks are the gates. Formal
controls retain their original 17-source proof namespace; diagnostics retain
the original 19-source geometry bank. Both share one lazily allocated native
64-substep environment.

## Evidence and acceptance

The unchanged version 3 objective requires a board bounce, whole-ball entry and
one continuous second inside. Table and rim assistance count; robot, floor or
unattributed contacts and invalid physics do not. Setup refusals, ordinary
flight misses, and invalid physics remain distinct.

Every row records its exact requested action, proof hashes, refusal reason or
admitted branch, actual native reset, and setup diagnostics including valid IK
branch counts and tested clearance. Because the recovery originals never
released, their zero joint placeholders are not a physical posture reference.
Every admitted diagnostic is compared explicitly with its **matched successful
parent**: actual q10/q14 differences, board-center displacement and normal angle.
Controls use their exact original admitted source as the reference.

The independent CPU auditor reconstructs the full plan and coordinate algebra,
checks both namespaces and original branches, verifies strict changed-action
proofs and native resets, and rescores dense native trajectories with the
independent v3 oracle. Final UI publication requires its hash-bound pass.
Finite solver rejection is not proof a pose is unreachable. Admitted held-pose
flight is not validation of dynamic grasping or executed arm transit.

## Runtime boundary

- 4090 only, selected by the existing GPU UUID and cooperative lease.
- At most 3,600 acquired-slot seconds, 900 queue seconds, and 6,300 pipeline
  seconds, with 1,800 seconds reserved for the independent CPU audit.
- At least 8 GiB free storage; fresh isolated source/evidence output namespaces.
- No 4070 workload, Isaac setup, physical robot, Viam, running service, model
  weights, scoring version, or interactive selector is changed.
- Raw native caches, proofs, receipts, logs and runtime configuration stay outside
  Git. Compact audited results and source code are published separately.

Launch through `sircus.mjlab.regression_ik_pipeline` with authenticated
`--parent-study`, `--parent-audit`, `--diagnosis`, `--rule`, `--reference`,
`--assets`, fresh `--output` and the existing `--gpu-lock`. Freeze and test the
producer plus independent auditor before launch. Do not silently resume a
partial run or extend the original deadline.
