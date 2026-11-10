# Continuous seed-22 shot geometry: offline preservation check

The original seed-22 policy directly produces five continuous shot settings.
This separate layer checks that exact proposal, preserves it if admitted, and
otherwise attempts one bounded nearby continuous geometric adjustment. It does
not use a target-to-candidate library. The saved seed-22 weights, completed
sweep/PPO benchmark, candidate-library experiments and V4 evidence remain
preserved historical evidence, unchanged.

The fixed candidate-library continuation changed the original policy into a
network that chooses from finite options. V4 is evidence about that separate
hybrid, not a promoted successor to the original direct-output policy. This
task returns to the continuous-output direction. It does not connect the new
layer to training, a simulator, a reader or hardware.

The completed [independently audited preservation check](../../baselines/isaac-continuous-geometry-20261110/README.md)
found **1,534 unchanged / 0 adjusted / 0 rejected**, including all 485 clean
historical actions. Geometry-only analysis took 77.84 s wall and independent
audit 55.59 s. All 38 focused checks passed without skips. No training or new
shot ran.

## What the layer does

`continuous_geometry_layer.filter_shot(scene, cell, action, links, starts)`
accepts a bucket position, exact five-setting proposal, the two robot chains and
actual measured six-joint starts. The original settings are bearing offset,
radius, board height, release gap and tilt. Admission delegates to the existing
coupled-arm `joint_feasibility.preflight` and independent validation. Existing
action bounds, endpoint residuals, FK/joint limits, enclosing-volume checks,
endpoint capsule checks and 16-sample path proxies stay unchanged.

The minimum gap remains 80 mm, defined as **ball-bottom release Z minus board
center Z**. Board tilt/normal clearance is checked separately. It is not a new
flange-to-board distance rule.

An admitted proposal is returned exactly unchanged and invokes no repair.
Otherwise one deterministic local least-squares solve varies the five
continuous settings and both six-joint endpoint fits. The default bounds are
24 optimizer function evaluations, 512 actual residual evaluations, and at
most 10% of the original action span in each setting. Distance penalties prefer
small adjustments; they do not establish a globally nearest solution. There
are no per-target choices, target lookup, flight simulations or reward/capture
queries in this solve. Its answer must then pass every original admission gate.
No verified answer means an explicit decline, not proof that the target is
impossible. Malformed or stale inputs fail closed.

Each result records `unchanged`, `adjusted` or `rejected`, the original cause,
any proposed repair, physical/normalized deltas, solver budgets, source and
state bindings, and independently checkable geometry evidence. It always has
`release_authorized=false`. Later execution would still need fresh measured
state, motion, settling and release checks.

## What this preservation test can establish

The primary input is every successful **paid TRAIN** action from the original
seed-22 run, including assisted captures under its unchanged scorer. These
actions were collected throughout the 10,000-request training history; they
are not all outputs of the final seed-22 checkpoint. The analysis verifies the
original passed audit, all 289 paid TRAIN packets and the final checkpoint's
raw bytes. It does not load a model or read held-out shot packets.

Fresh IK starts from each request's recorded measured arm positions. The
primary check does not supply that success's saved IK answer, consult a
candidate library, or sample a policy. A separate independent reader checks
one-to-one coverage, classifications, changes and geometric evidence.

Preserving a recorded action means the new layer does not alter that setting
and finds an admitted geometric solution under the declared finite solver.
It does not reproduce its old ball flight. If an action is adjusted, its
previous capture label cannot be attached to the adjusted setting. This test
does not measure a new capture rate, generalization or learning improvement.
Numerical runtime versions are recorded because solver branches can depend on
the environment.

## Limits and next boundary

The inherited clearance and sampled-path checks are conservative proxies, not
a full mesh/self-collision certificate or obstacle-avoiding planner. Numerical
IK can fail to find an existing solution. The local repair can also fail, and
has no guarantee of finding every nearby feasible setting.

No training or new task shot is authorized here. A later continuous-policy
training design must explicitly account for raw proposals, adjusted executed
actions and declines; the old continuous PPO probability cannot silently be
relabelled as a probability of a repaired action. That design and any native
evaluation remain separate work.
