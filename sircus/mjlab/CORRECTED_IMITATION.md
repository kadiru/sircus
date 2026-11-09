# Corrected imitation: frozen comparison on 70 new positions

This simulation-only study tests whether the five verified recipe corrections
help the current imitation policy generalize. It starts from the exact
`coverage-recovery-v1` recovery BC checkpoint, preserves its **92 teachers**, and
adds **one corrected recipe for each of five inspected contexts**. Existing
nominal confirmations already scored 3/3 for each recipe. Their repeats are
verification evidence, not fifteen additional training examples.

| Correction | Original development context | Added recipe |
| --- | --- | --- |
| Height | `recovery-test-r3-c1-n0` | Height-corrected first repeat |
| Drop gap | `recovery-test-r0-c5-n1` | Gap-corrected first repeat |
| Drop gap | `recovery-test-r0-c6-n1` | Gap-corrected first repeat |
| IK admission | `recovery-test-r1-c1-n1` | Parent board X/Y restored; other recovery values preserved |
| IK admission | `recovery-test-r2-c4-n0` | Parent board X/Y restored; other recovery values preserved |

These five previously inspected TEST contexts are now development data. Their
historical labels and the current policy's historical **37/70** are unchanged.
The new result comes only from the new test set below.

## Training and frozen evaluation

The candidate starts from the current actor and receives **800 full-batch Adam
steps**, learning rate 0.001, on the 97 unique teacher contexts. It minimizes
mean squared error between its five bounded residual outputs and the teacher
residuals. The critic, exploration variance, network architecture, observation
features, action bounds and 19-source geometry anchor bank are unchanged.
This is imitation learning; there are no PPO updates in this study.

Each corrected action is reconstructed against its **original parent anchor**.
Its nine-feature observation and five normalized residual labels are rebuilt
and checked, rather than trusting residual labels carried by an earlier
correction row. Old teachers remain identical records. Source report, audit,
recipe, setup-branch proof and trajectory hashes preserve the lineage.

The fresh seed is `2026110941`: a 7-column × 5-row table grid, two new bucket
positions per cell. All 70 positions are at least 5 mm from the inventoried
historical bucket positions, all training/correction positions, and each other.
Sampling is based on coordinates only; difficult or rejected contexts are not
removed after observing their outcomes.

Both models and all **140 predictions** are frozen before the first TEST drop.
Each policy receives exactly one request per position, including requests
rejected by setup checks. Geometry-only admission may select an anchor; it
cannot test ball flights or optimize a shot after seeing its result. No TEST
updates, retries, checkpoint selection, or replacement positions are allowed.
This tests the combined effect of the five new examples **and 800 additional
optimization steps**; it does not isolate those two effects independently.

## Physics, scoring and controls

Native MuJoCo-Warp simulation remains at **64 substeps**, with the same scene,
calibration and version 3 evaluator. A capture requires a measured board
bounce, whole-ball entry and one continuous second retained. Table and rim
assistance count. Robot, floor and unattributed contacts invalidate the route;
invalid physics and setup rejections remain separately reported.

Nine original controls run before fitting and nine after TEST: three direct
captures, three table-assisted captures and three expected misses in each
block. These retain their original 17-source proof namespace and admitted
branches. Both policy methods share the unchanged 19-source geometry bank.
The full budget is **18 controls + 140 TEST requests = 158 requests**.

The report shows capture counts and rates, clean captures, setup rejections,
flight misses, invalid physics, per-position recipes and paired gains versus
regressions. The independent CPU audit replays the actor-only optimization,
checks model and teacher identity, reconstructs the held-out split and all
predictions, verifies setup and native reset proofs, and rescores dense saved
trajectories. The UI publishes final results only after that audit passes.

Nominal held-pose simulation does not establish noise robustness, timestep
convergence, real hardware performance, dynamic grasping or arm transit.
No model is automatically promoted or activated by this comparison.

## Bounded execution

Launch `sircus.mjlab.corrected_imitation_pipeline` with:

- `--parent-study`, `--parent-audit`: audited `coverage-recovery-v1`.
- `--height-study`, `--height-audit`: audited `recovery-regression-v1`.
- `--gap-study`, `--gap-audit`: audited `regression-drop-gap-v1`.
- `--ik-study`, `--ik-audit`: audited `regression-ik-v1`.
- `--history-root`, `--reference`, `--assets`, a fresh `--output`, and the
  existing cooperative `--gpu-lock`.

The runner uses the 4090's fixed UUID only, with maximum 900 seconds queued,
3,600 acquired-slot seconds and 6,300 total pipeline seconds. It reserves
1,800 seconds for the independent CPU audit and requires 8 GiB free storage.
It preserves the original absolute deadline and stops only its owned worker
on timeout. Partial evidence remains available but is not a passed result.

The 4070 workload, Isaac installation, running dashboard services and physical
hardware are untouched. Generated checkpoints, native traces, proofs, logs and
host configuration stay outside Git; only reviewed source and compact audited
summaries are published.
