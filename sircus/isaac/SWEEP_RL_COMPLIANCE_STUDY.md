# Sweep, direct RL inference and impact response

Design checkpoint, November 2, 2026. **Not an executed experiment or a new
controller.** Existing studies, checkpoints, scene fingerprints and scorers stay
unchanged. The first comparison keeps the ball/material properties fixed.

## Research questions

1. Which combinations of board height, tilt and release gap capture a ball at
   each target distance, within the two arms' reachable workspace?
2. Can a continuous RL policy predict a successful combination at an unseen
   distance without native shot search, candidate ranking or retuning there?
3. How much does the actual articulated board move during impact, and does
   changing arm posture alter a shot when relative shot geometry is held fixed?
4. Later: can force/displacement observations from squeezing a ball improve shot
   selection for balls with different dynamics?

Physics equations are candidate generators and comparison baselines. They are
not capture labels. The shaded explanatory movie is not experimental evidence.

## Existing work and execution boundary

The earlier Isaac study found searched recipes at 19/35 grid sites; its PPO
selects from a finite candidate bank, with no unseen-bucket test. Preserve those
results under their original definitions and fingerprints.

At the read-only check on November 2, a separate `window-sac-v2` source snapshot
was running against output `window-sac-v1` on the shared 4090. Its published
branch is `codex/macbook-mjlab-window-sac`, commit
`349f6a9`. It compares adaptive search with terminal SAC over **release gap** at
one fixed bucket/board XY and disjoint height/tilt contexts. Its results do not
establish unseen-bucket-distance performance. Read its `sircus/mjlab/WINDOW_STUDY.md`
before commissioning another comparison. Do not interrupt or duplicate it.

Both GPUs were busy and the Ubuntu root filesystem had about 3.8 GiB available
at that check. These are historical observations, not current scheduling
authority. Recheck processes, the shared GPU lock and storage before any run.
This design checkpoint does not queue or launch another job.

## First benchmark: fixed materials, unseen distances

Use the existing articulated Isaac runtime and an immutable new source snapshot.
Keep current measured/provisional assets, bucket dimensions and capture rules.
The 175 mm reported bucket height remains unresolved; retain its original
provisional interpretation and record it in the new contract.

- Board/release XY = (670, -100) mm in the dropper-base frame; bucket along -Y.
- Bucket distance domain: 320-640 mm. This first benchmark is scalar-distance
  interpolation along a fixed direction, not coverage of arbitrary tabletop XY.
- Three continuous actions: board-center H 140-540 mm, ball-bottom gap G 80-560
  mm and board tilt 12-52 degrees from horizontal, facing the bucket.
- Ball-center release Z = H + G + radius; record actual first-contact fall and
  velocity separately. Do not interpret G as the exact free-fall distance.
- Heading, release XY and downward dropper orientation are fixed. Endpoint IK
  and measured settling still reject infeasible combinations. Apply a common
  actual-FK geometry gate for the rotated board, table, bucket and release/tool
  clearance before execution. `ParallelShots.prepare()` alone does not provide
  the expanded sweep's full board-clearance filter. Retain failures in all
  attempt denominators; do not silently project onto a successful menu.

Before outcomes, create a hashed manifest of 24 training, six validation and
50 test distances with no exact overlap. Stratify all splits across the domain;
label this as interpolation, and report near/middle/far bands separately.
Keep test distances/outcomes inaccessible to proposal generation, optimization,
normalization fitting and checkpoint selection. Reserve a different experiment
for extrapolation or arbitrary bucket XY.

### Methods and bounded pilot budget

Each development method gets 4,096 native training/development shot requests;
invalid requests consume budget. Validation adds 72 requests per development
method, and primary testing adds 50 each for library/RL/analytic methods. The
planned comparison therefore uses 8,192 development + 144 validation + 150
primary-test = 8,486 requests, excluding separately logged preflight controls or
optional oracle searches. Use one prespecified pilot seed and publish its
limitations. Multi-seed confirmation follows only after the pipeline is verified.

**Sweep/search:** start with 24 distances x 3 heights x 9 gaps x 5 tilts = 3,240
requests. Use H = 180/340/500 mm, G = 80:60:560 mm, tilt = 12:10:52 degrees.
Spend the remaining 856 requests on deterministic transition refinement and
unexplored regions, using training outcomes only. Publish the exact refinement
schedule before execution. Store disconnected successful runs rather than one
Cartesian min/max box.

**Continuous PPO:** reuse `farm_policy.Policy` and `ParallelShots`, with a new
versioned distance observation and local action transform. Output real-valued
H/G/tilt, not a candidate index. Use 32 updates x 128 terminal shots = 4,096;
balance distance assignments over the full request stream. Freeze algorithm,
warm-start information, action bounds, seed and reward definition in the run
manifest. Do not train on the sweep's outcome labels and describe it as an
independent pure-RL comparison. If labels later warm-start the actor, name that
hybrid method separately and charge their collection cost.

Use the same native capture evaluator for every method. A capture requires a
measured board rebound, whole-ball entry and one continuous second inside, with
no external-contact assistance. Rim-assisted captures count. Clean captures,
retention, misses and endpoint rejects remain separate fields. Freeze any
shaped training reward under its own version; do not rewrite old rewards.

### Frozen evaluation

Validate at request milestones 0/1,024/2,048/4,096, with three paired episodes per
validation distance. Choose the checkpoint by validation captures, then mean
reward, then earliest milestone. Freeze before revealing the test set.

Primary test: one predicted action per each of the 50 unseen distances.

- RL: one deterministic actor evaluation, followed by geometric IK/settling and
  one shot. No retry, simulator-outcome query, per-target optimizer or menu
  substitution. A geometric rejection remains a failed test request.
- Library baseline: retrieve a frozen training-derived recipe by a prespecified
  rule. Do not average disconnected successful regions or tune against test
  outcomes. State clearly whether the rule interpolates or uses nearest distance.
- Analytic baseline: one fixed equation-derived proposal with geometric checks.
- Optional per-test search is an **oracle/search reference**, not policy inference.
  Report its extra native calls and latency separately, never as a learned score.

Report capture counts, clean captures, invalid requests, near/middle/far results,
query/IK/shot latency, native request cost and exact predicted/executed values.
Do not promote a weak policy simply because training reward rose. Follow-up
repeats or placement perturbations need a separate frozen protocol; identical
deterministic repeats are not independent robustness evidence.

## Arm posture and impact response

The existing arms have finite implicit PD gains plus gravity compensation. The
board is a rigid fixed link; board bending and suction-mount flexibility are not
modeled. The current runtime can move under impact but does not save the
telemetry needed to quantify that effect.

Add compact per-shot telemetry before claiming compliance-aware learning:

- Settled pre-release joint configuration and board pose.
- First board-contact timestamp, force peak and step-resolved impulse estimate.
- Board position/orientation change, linear/angular velocity and joint response
  around impact, with an explicitly bounded optional trace.
- Incoming/outgoing ball velocity relative to the estimated contact point on the
  moving board, including its linear and angular velocity contributions. Record
  signal/finite-difference provenance; contact timestamps and impulse estimates
  are resolved at the observation/physics step, not exact solver-substep events.
- Provenance for each effort signal: commanded gravity feedforward and an
  implicit-drive estimate are not measured total motor torque.

For a matched posture experiment, translate board, release and bucket together
across reachable workspace locations while preserving D/H/G/tilt. Keep arm bases,
gains and ball properties fixed. Verify actual settled geometry and measured
incoming ball velocity, not only requested commands; flag unmatched trials rather
than attributing posture-dependent sag to a different impact response. Compare
drop trials with paired no-ball holding controls so ordinary drift is not labeled
impact deflection. Log reach failures rather than forcing an extended pose.

Greater extension does not necessarily mean greater yielding: joint geometry,
effective inertia, gravity compensation, drive gains and effort saturation all
matter. Measure the relationship. Relative target distance alone cannot describe
it; a later policy needs the settled joint configuration and board dynamics, or
features derived from them.

`prepare()` currently initializes endpoints directly. It discards arbitrary
starting poses; feeding those poses to the policy would not teach motion cost.
Choosing motions from an arbitrary start requires a separate driven-transition
task and collision/path validation.

The historical rebound check projects ball velocity onto the settled board
normal. Keep that historical field unchanged. Add instantaneous contact-relative
velocity as a diagnostic first; any revised rebound/scoring rule needs a new
version with results reported separately.

## Stretch goal: squeeze-informed ball identification

Do not assume static stiffness determines bounce restitution. A squeeze can
estimate a force/displacement curve; loading/unloading hysteresis and rate can
add information about energy loss, but dynamic drop/bounce measurements remain
necessary. Estimate mass separately when relevant.

First establish available gripper position and force sensing, or calibrate a
motor-current proxy. Current geometric gripping controls do not establish this
measurement capability. There is no physical-arm access in this study.

A later policy may condition on a measured ball descriptor plus bucket/arm
state. If that descriptor is unavailable, use an explicit identification phase
or recent bounce history; do not give the actor hidden simulator material truth
while describing it as squeeze-based perception. Distinguish oracle-property,
identified-property and unseen-ball tests. Changing a rigid sphere's mass and
restitution is an approximation, not a deformable-ball implementation.

## Next implementation gate

Before a new native run: finish/review the current narrower study; implement the
new split/action contract and compact impact telemetry; verify CPU leakage and
action-transform checks; run a bounded native control and continuous-action smoke
under the shared GPU lock. Freeze source/assets/contracts and resource limits
before collecting benchmark outcomes. This document does not launch that run.
