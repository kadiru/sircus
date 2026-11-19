# Fixed-64 success bootstrap → PPO

Completed results, exact checkpoints and nominal settings are published in the
[November 5 baseline](../../baselines/mjlab-policy64-20261105/README.md).
PPO captured 18/50 new test layouts versus imitation's 20/50; it did not improve
capture rate. The protocol below was frozen before collecting those outcomes.

This study deliberately fixes native physics at **64 substeps**. Earlier
64/128 disagreements remain published and are not reclassified as reliable
physics. No 128 convergence or noise requirement is imposed on this new policy.
All outcomes still require board bounce, whole-ball entry and one continuous
second retained; rim-assisted captures count separately from clean captures.

## Historical data and learning

The source is the independently audited calibrated benchmark v2. Its six
complete TRAIN logs contain 6,144 requested decisions, 4,380 native releases,
2,439 captures and 842 clean captures at 35 bucket layouts. Capture teachers
exist for 22 layouts. Historical validation/test labels never enter training.
The prior acquisition cost is disclosed, not treated as free cold-start data.

Choose one captured TRAIN teacher per context: clean first, then highest
calibrated reward, then stable source ID. This avoids averaging incompatible
successful recipes. Train the actor by full-batch behavioral cloning from these
22 examples (800 Adam steps at .001); fit the critic from all 6,144 TRAIN
outcomes (400 full-batch Adam steps at .001), retaining misses and rejections.

The actor observes nine normalized geometry values: query bucket XY, original
geometry-source bucket XY, and five effective source parameters. It outputs a
continuous residual in board X/Y/height, tilt and release gap, scaled by
40/40/50 mm, 8° and 120 mm. Board yaw faces the bucket by the existing geometric
rule. Dropper XY is above board center; its release center height is board
height + nominal center-plane gap + ball radius. No point-cloud perception,
grasp or continuous arm-transit policy is trained here.

Refine the bootstrapped model with **1,024 fresh requested TRAIN shots**, in
32 terminal-shot PPO updates. Reuse the existing 9→64→64→5 actor and critic,
Adam .0003, six epochs/update, clip .2, value weight .5, entropy .001,
gradient norm 1 and log-standard-deviation bounds [-3,-.7], initialized at -3.
Every rejected requested action consumes its training slot; no resampling.

Reward is unchanged calibrated-shot-reward-v2: rejected actions -1; valid shots
receive .15 for board bounce plus .5 exp(-opening distance/.12 m). External
contact subtracts .5 and ends additional bonuses. Otherwise add up to .5 for
one-second retention, +3 for capture, +.3 for clean entry and +.1 for a single
board-contact episode. This reward remains separate from the binary capture
criterion and does not promote a near miss to success.

## Frozen evaluation

TRAIN retains the 35 original grid coordinates. Sample 12 new validation and
50 new test bucket coordinates uniformly within X100–1240 mm, Y-650–250 mm,
at least 5 mm from each other and all old benchmark coordinates. Seed the
validation/test pools with 20261503/20261504; training seed is 20261511.
No admission-based replacement of hard positions is permitted.

Validate deterministic actor means at TRAIN prefixes 0/256/512/1024. Select by
validation captures, mean reward, then earliest prefix. Freeze that checkpoint
and all 200 test predictions before observing any test outcomes. Test each of:

1. Zero residual from the shared geometry prior.
2. Nearest immutable successful TRAIN teacher's normalized residual.
3. Frozen behavioral-cloning policy before fresh PPO.
4. Validation-selected frozen BC→PPO checkpoint.

Each method gets exactly one nominal requested shot at each of the same 50
positions. Rejects stay in the denominator. No test-time learning, ball-flight
search, retries or noise. The nearest library remains the original 22 bootstrap
teachers; fresh PPO successes do not silently expand it. This is a warm-start
ablation with unequal training exposure, not an equal-budget algorithm contest.

Geometry-only source resolution precedes prediction and is explicitly reported.
IK branches and finite sampled setup paths may be searched, but a policy's
requested action is never replaced after seeing its flight outcome. This does
not validate driven actuator motion or grasping. All four methods share the
same fixed per-bucket geometric anchors.

Fresh source-command positive/native-miss controls run three times each before
and after learning, using newly resolved branches. They are not frozen historical
joint replays. Total maximum new requests: **1,284** = 1,024 TRAIN +48 validation
+200 test +12 controls. The worker has a two-hour acquired-slot deadline and
uses only the 4090 UUID/shared lock; foreign workloads, 4070 and Isaac stay alone.

## Runtime and evidence boundary

The historical source guard is preserved. Native reset/clock code changed after
the source benchmark, so its labels are bootstrap information, not current
rollout guarantees. A new explicit runtime-reference bridge binds the original
scene/assets to the newer independently audited impact-study source. Current
source, exact allocated native/Warp options and SciPy1.15.3 are frozen separately;
fresh controls and the unchanged calibrated oracle score every new outcome.

Save TRAIN source row hashes, teachers, fit losses, initial/BC/milestone states,
all 32 behavior/optimizer/update buffers, pre-outcome predictions, setup proofs,
native arrays and a complete manifest. A separate CPU-only auditor checks source
exclusion, reconstructs BC/PPO/frozen inference and rescores all dense native
traces. The UI key `policy64_study` / panel `#policy64Study` requires that full
byte-bound audit before presenting final results as verified.

Raw source rows, joint states, observed poses/contact arrays, host configuration
and native traces remain outside Git. Publish only scalar summaries, nominal
actions/coverage, checksums and trained model weights. The existing interactive
solver and historical panels remain separate; a result panel is not automatic
activation of the new checkpoint for arbitrary bucket queries.
