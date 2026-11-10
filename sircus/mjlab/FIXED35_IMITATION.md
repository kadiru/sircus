# Fixed 35-position imitation study

The first evaluation uses exactly **35 fixed bucket locations**, arranged in
seven columns and five rows. There is one request per location per frozen
model. Bucket placement perturbations, noise and 128-substep checks are outside
this study.

The canonical coordinates, in millimeters, are:

| Axis | Fixed values |
| --- | --- |
| X: seven columns | 100, 290, 480, 670, 860, 1050, 1240 |
| Y: five rows | −650, −425, −200, 25, 250 |

Every X/Y combination appears once. These locations were used in historical
development, so this is **fixed-grid coverage**, not unseen-position or
all-table generalization. No position is replaced after an unfavorable outcome.

## Fixed behavior and imitation refit

Both models use the same bucket-facing dropper-gripper orientation. It is a
geometric drop behavior, not an extra parameter the policy must discover. The
full-pose IK and collision checks remain enabled. This is not a separate
orientation A/B experiment.

The candidate retains 97 teacher contexts and adds one verified bucket-facing
recipe from each of two inspected layouts. One teacher per distinct layout
prevents the ten successful farther-shot alternatives or repeated confirmations
from overwhelming the other examples. Original teachers retain their recorded
orientation and evidence: their success labels do not establish that the same
recipes work with a different orientation.

The planned fit is 800 full-batch, actor-only Adam updates at learning rate
0.001. Both checkpoints and all 70 evaluation commands freeze before evaluation
outcomes. The current checkpoint remains unchanged. The model predicts five
continuous shot adjustments around the existing geometry anchors; inference
does not simulate candidate ball flights and choose the best observed outcome.
There are no PPO updates, evaluation retries or outcome-based model selection.

## Evaluation and reporting

The 64-substep simulation and version 3 evaluator stay unchanged. A capture
requires a board bounce, whole-ball entry and one continuous second retained.
Rim and table assistance count; robot, floor and unattributed contacts invalidate
the route. Setup declines remain in the 35-request denominator and are distinct
from released flight misses. A finite IK refusal is not proof that a position
is impossible.

The nominal budget is **70 evaluation requests plus 18 original controls**:
three direct captures, three table-assisted captures and three expected misses,
both before and after evaluation. Control outcomes do not contribute to a
policy's 35-position capture count.

The saved `#fixed35Imitation` panel shows the 7×5 matrix, exact bucket
coordinates, both predicted shot settings, captures, flight misses and setup
declines. Final outcomes are gated on the complete independent native audit;
pending or failed execution cannot appear as a completed policy comparison.
Historical panels and the runtime report are preserved through additive
publication. No viewer restart or automatic policy activation is required.

The 4090 is the only permitted GPU for the finite worker. The shared GPU lock,
queue and execution deadlines preserve other work. The 4070 workload and Isaac
installation remain outside this task. Generated checkpoints, complete native
traces, environment configuration and logs stay outside Git. Simulation arms
hold admitted endpoints during the ball flight; this does not validate learned
transit, perception, grasping or physical hardware.
