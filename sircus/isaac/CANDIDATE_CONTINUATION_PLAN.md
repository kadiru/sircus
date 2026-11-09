# Seed 22 continuation: proposed experiment, training off

Implementation status, November 9: the [candidate adapter and pure losses / ledger](CANDIDATE_ADAPTER.md)
now pass [53 CPU no-update tests and actual-parent initialization checks](../../baselines/isaac-candidate-adapter-20261109/README.md).
The checklist below records the proposal's original gates. The optimizer/update
loop, checkpoint writer, fresh split/support preparation, native integration,
operational supervisor and independent training auditor remain deferred.
No training or new native shot is authorized or launched by this implementation.

Prepared November 8, 2026 (America/Los_Angeles). **This is a reviewable proposal,
not a launch authorization or an implemented trainer. No imitation, PPO update,
new native request, or new held-out evaluation belongs to this planning task.**

## Plain-language proposal

Keep seed 22's existing neural networks. First practice copying successful
TRAIN examples from the checked bank; then let PPO improve its choices among
that bank's candidates. Keep reachability, spacing, motion and release checks
outside the learned model. Test the final result against both the original
model and the current bank selector on fresh positions.

Proposed limits, subject to review before any training:

| Phase | Task ball-release limit | Requested execution slots | Other limit |
|---|---:|---:|---|
| Existing-data imitation | 0 | 0 | 20 passes through 1,534 demonstrations |
| PPO continuation | 1,400 | 1,750 maximum | Stop at the first limit reached; 2 hours native wall |
| Final assessment: three methods | At most 420 | Exactly 420 if completed | 35 TRAIN + 35 new validation + 70 new TEST positions per method; 30 minutes native wall |
| New held-out candidate preparation | 0 | 0 | Exactly 8,505 raw proposals; 4 CPU workers; 30 minutes wall and 2 aggregate CPU-hours |
| Inherited startup diagnostics, once | 0 task releases | 0 task slots | At most 14 diagnostic cohorts / 490 initialized probe-ball trajectories; details below |

Thus the maximum is **2,170 newly requested task slots and 1,820 task ball
releases**, not 2,170 guaranteed releases. Existing data, the completed 70-request
comparison and explicitly recorded startup control probes are separate entries.
These figures are proposed workload caps, not measured cost or a power analysis.
The current task consumes zero of these proposed budgets.

## What is inherited and what changes

The [audited native comparison](../../baselines/isaac-candidate-native-20261109/README.md)
used the same 35 TRAIN positions per method: original 13 captures / 4 clean /
7 rejections; bank selector 16 / 5 / 0. It changed no weights. All 28 preserved
actions retained their observed outcomes; seven actions changed. Its results
motivate this proposal and do not establish held-out improvement.

Inherit the exact final checkpoint from seed `20261122`:

- SHA256: `57d7658e9ed2f72ad09cf4e8523ab497ccbe0708364f1b8ec4a4c5c1aef1fe9f`.
- Original provenance: 10,000 requested TRAIN shots and 73 optimizer updates.
- Actor: `5 -> 64 -> 64 -> 5`; critic: `5 -> 64 -> 64 -> 1`; tanh hidden
  activations; five learned Gaussian log scales. The saved tensor shapes match.
- Preserve the five bucket-geometry observations, physical action domain,
  radius prior, scene/assets/camera, physics, scorer, reward and all guards.
  In particular, the original gap variable remains ball-bottom release Z minus
  board-center Z, with its original 80 mm lower bound; it is not silently
  redefined as tool-to-tool clearance.
- Reuse the [1,814-candidate TRAIN bank](CANDIDATE_BANK.md) and its exact action,
  qualification, witness/start and audit identities. No new TRAIN positions,
  bank replenishment, changed geometry, perception or operator board-goal inputs.

The actor, critic and log-scale **weights** initialize from the parent. Fresh
optimizer states are deliberate: old Adam moments describe a different action
law/objective. Keep inherited and new counters separate. Load the parent with
its original architecture/study/domain/fingerprint guards, then create a new
continuation checkpoint schema that records the parent hash and new protocol.
Do not relabel the new policy as the old continuous policy or alter its loader
to bypass those original checks. Proposed continuation RNG seed: `20261125`.

## A trainable choice rule, not action replacement after PPO sampling

For each TRAIN position, freeze an ordered list of qualified exact candidates
and one immutable latent vector `z_i` per candidate. Use its original saved
parent latent for the exact parent action; use the original physical affine
inverse and `atanh` for other candidates. Saturated boundary values fail closed;
there is no clipping, averaging, projection or invented candidate.

At each update, obtain `mu_theta(s)` and `sigma_theta(s)` through
`ContinuousPolicy.distribution(observations)`, preserving the state-dependent
radius prior. Define the new finite distribution by:

```text
score_i = sum_d log Normal(z_i[d]; mu_theta(s)[d], sigma_theta(s)[d])
log_probability_i = score_i - logsumexp(all eligible scores)
```

This reuses the current bank selector's latent-density preference convention.
It is a new categorical action distribution, not the original continuous
Gaussian conditioned on feasibility. No tanh Jacobian is added to these finite
scores. Sampling and training use this same law; deterministic assessment takes
its argmax. An exact frozen-parent candidate wins a numerical tie under the
original `1e-12` score tolerance; other ties use stable candidate-ID order. This
is an immutable tie priority, not a rule that keeps the parent maximal as weights
change. Training may choose a different qualified candidate even when the
original parent action is feasible.

Important source-review findings:

1. The frozen helper in [candidate_selector.py](candidate_selector.py) substitutes
   the supplied mean as the exact parent's latent. That was correct for the
   frozen comparison. Reusing it with a changing mean would incorrectly move
   that fixed candidate in latent space and keep it maximally scored. The new
   implementation must cache the **original** latent and remove forced parent
   preservation from the trainable rule. The frozen helper stays unchanged.
2. [continuous_policy.py](continuous_policy.py) computes continuous log densities
   and squashed-Gaussian entropy. Its old updater/rollout buffers are unsuitable.
   New PPO ratios must use categorical old/new log probabilities on the same
   support, and exact categorical entropy. No old Gaussian sample or log density
   is imported into a new PPO buffer.
3. Current supports range from 1 to 176 choices, and existing log probabilities
   reach about -109. Use float64 candidate latents/scores and log-space
   normalization while retaining the parent weight dtype. Test finite gradients
   and rare probabilities explicitly. A singleton support legitimately supplies
   zero action-selection gradient; it still supplies a real critic target.

This is consistent with the need to use the actual allowed-action distribution
for both sampling and policy gradients discussed in
[Huang and Ontanon, invalid-action masking](https://arxiv.org/html/2006.14171).
That paper's discrete-game results are motivation, not evidence that this
specific robot implementation or latent score family will work.

## Stage A: learn from existing successful TRAIN examples

Use only the existing hash-bound export of **1,534 paid TRAIN captures**, including
485 clean captures. Each exact action must map to its audited bank candidate and
original TRAIN context. No validation/TEST outcome, failed proposal labelled as
successful, or new ball flight is used. Do not add the 70-request diagnostic's
outcomes as extra supervised labels in this version.

The historical successes cover **21 of 35 TRAIN positions**. Give each of those
21 contexts equal total success-loss weight and each unique successful candidate
within a context equal weight. Assisted captures remain valid examples under
the original scorer; there is no new clean-example weighting. The other 14
contexts receive no invented success target.

Proposed fixed schedule:

- 20 epochs, each visiting all 1,534 demonstrations once: 30,680 demonstration
  presentations; deterministic shuffling; batches of at most 128 examples;
  12 batches per epoch and at most 240 optimizer steps.
- Train actor layers only with fresh Adam, learning rate `1e-4`; keep critic and
  log scales frozen. Neither imitation nor its verification is run in this task.
- Minimize negative log probability of the separate exact successful choices,
  with fixed global per-example weight `1534 / (21 * successes_in_context)`.
  Take the weighted NLL mean over the actual minibatch size; do not renormalize
  the weights separately inside a minibatch. Do not regress to an average board
  pose: the average of two good shots need not be a good shot.
- Add a fixed `0.05 * KL(parent_bank_distribution || current_distribution)`
  anchor: exact forward categorical KL over the identical fixed support,
  equally averaged over all 35 contexts and applied once per optimizer step.
  This limits forgetting of the original finite preferences, not a claim that
  those choices were successful. It provides the only supervised preference term
  at the 14 no-success contexts.
- Record per-context loss, successful-candidate probability mass, entropy and
  choice changes before/after imitation. Stop on nonfinite values; do not choose
  an epoch from a validation capture score or extend epochs after looking at it.

The diagonal-Gaussian score family cannot express arbitrary probabilities over
many separated successful actions. Imitation loss decreasing is not proof of
better ball capture; the final native comparison must decide that.

## Stage B: bounded PPO on the fixed TRAIN bank

Freeze the bank throughout this stage. Draw a candidate categorically, execute
its exact action once and retain its full terminal outcome. For an exact parent
action use the same native IK seeding as the frozen parent; for other candidates
use their prepared fits as declared warm starts. All original preflight,
sampled-path, driven-motion, settling and release checks still run. Warm starts
do not authorize movement or release. The bank's measured-start binding is not
a substitute for checking the actual start.

Use one deterministic round-robin schedule of the 35 TRAIN contexts, with a
seeded permutation per logical round. A rejection is followed by the next
scheduled context, not a redraw to get a free attempt at the same target.
Every selected action that the native guards reject stays in the PPO record
with the unchanged invalid reward. Kinematically rejected raw preparation
proposals never become sampled PPO actions or consume a task release.

Stop when **1,400 task releases or 1,750 requested execution slots** is reached,
whichever comes first. Limit the next active native batch to the minimum of
35, remaining request allowance and remaining release allowance, so simultaneous
releases cannot overshoot. Record both counters and per-context exposure; a
completed request-cap run below 1,400 releases reports that target shortfall.

Proposed PPO settings are the inherited reward and defaults: fresh Adam at
`3e-4`, clip `0.2`, 4 epochs per update, value coefficient `0.5`, entropy
coefficient `0.005`, gradient norm `1.0`. Unfreeze actor, critic and log scales
(with the original `[-3, 1]` log-scale clamp). Collect fresh buffers of 140
requested slots and update, flushing the final partial buffer: at most 13
updates and 52 optimizer steps. Normalize advantages for buffers of size two
or more; a singleton final buffer uses its raw advantage. The critic trains on
all fresh terminal outcomes, never on success-only imitation labels.
The terminal advantage is `return - rollout_old_value`, detached before all four
epochs; there is no bootstrap or GAE. No intervening imitation update changes
the policy behind an existing PPO buffer. Inactive padded native slots are
neither requested slots nor optimizer samples.

Store context/support/mask hashes, immutable candidate ID/latent/exact action,
old categorical log probability, selected probability, critic value, terminal
return, rejection/release status and source bytes for every sampled action.
Keep support fixed across each rollout and its update. No changing temperature,
exploration mixture, reward, candidate domain or hidden online support growth.

The original capture reward with its clean bonus is retained. Primary capture
and secondary clean reporting do not make that reward mathematically
lexicographic. Do not retune it to make this pilot look better.

## Fresh assessment and what would count as improvement

The old validation/TEST outcomes have already been inspected. They remain
historical evidence and cannot be reused as an untouched final exam.

Before training, a new outcome-blind manifest would draw **35 new validation
and 70 new TEST bucket positions** with proposed split seed `20261126`. Retain
the original bounds, 7-by-5 stratification, minimum 25 mm separation from the
union of the frozen discovered historical pool **and all 140 original benchmark
TRAIN/validation/TEST coordinates**, and between new positions, with at most
1,000,000 coordinate draws. The old benchmark's pre-experiment historical pool
alone is insufficient: it predates those already inspected 140 positions.
Apply no reachability, capture or reward filter to
select the positions. Failure to fill the grid stops preparation without
shrinking the distance or dropping difficult tiles. Hash the history-discovery
receipt; novelty is relative to discovered recorded coordinates, not all
unpublished experiments.

For each of these 105 positions, prepare exactly 81 raw choices: the frozen
parent's deterministic action, 16 parent-Gaussian proposals and 64 Sobol
proposals. Proposed proposal RNG seed: `20261127`. No historical-success
lookup, learned-model refresh, nearest-success transfer, flight ranking or
outcome query is allowed. One bounded coupled-arm preflight per exact proposal
uses declared canonical starts, the original finite robust-IK protocol and
unchanged checks. Internal multistarts and their counts remain explicit.

The 8,505 raw choices and qualified survivors are frozen and independently
audited before learning. The new policy and frozen bank selector share these
same exact held-out supports and start/seed rules. Empty supports remain
failures in the requested-position denominator; they are not refilled or
removed. Raw fresh coordinates remain outcome-held-out, but using their
geometry to prepare support is **geometry-assisted evaluation**, not a claim
that the pipeline never saw the test geometry. The TRAIN policy never receives
these contexts, supports or outcomes during its gradient updates.

After the prespecified TRAIN stopping rule, hash the final checkpoint **before
any new validation or TEST ball outcome**. Use that artifact only, not the best
intermediate checkpoint. Evaluate each of these three frozen methods once at
all 35 known TRAIN, 35 new validation and 70 new TEST positions:

1. Original seed 22, continuous deterministic inference and original guards.
2. Frozen seed 22 finite bank selector. At fresh positions this means the same
   frozen scoring rule on the new geometry-only bank, not exporting the
   outcome-informed TRAIN bank to test positions.
3. The continued actor's deterministic categorical choice on the same bank.

Run order within each 35-position cohort rotates to limit a fixed order effect:
TRAIN: `1,2,3`; validation: `2,3,1`; first TEST half: `3,1,2`; second TEST half:
`1,2,3`. Reset to canonical home before each method/cohort. There are exactly
420 assessment requests if complete, at most 420 task releases, no repeat,
replacement or weight update. Validation is a final diagnostic, not a model
selection or hyperparameter-tuning loop. TEST is opened only after freezing.

Primary comparison: new policy versus frozen bank selector, **TEST captures /
70 requested positions**. Also compare with original continuous seed 22.
Report clean captures, all rejection reasons, releases, capture conditional on
release, paired gains/losses and per-position outcomes. Keep rejected/empty-bank
positions in the primary denominator. Known-TRAIN retention separately lists
any loss of the original 13 capture positions and the selector's 16 captures;
the historical counts are references, while the new paired controls determine
the fresh assessment result.

Report paired 95% intervals from position-wise paired bootstrap resampling
(10,000 replicates, proposed reporting seed `20261128`) and exact gain/loss
counts. These describe this fixed grid and one continuation, not uncertainty
over independent training seeds. An interval crossing zero is inconclusive;
even a positive observed difference is not an automatic model promotion.
Disclose any clean-entry or rejection tradeoff rather than declaring a winner
from training reward. No automatic rerun, extra seed, larger budget or follow-on
experiment follows an unfavorable or inconclusive result.

## Cost, operational limits and implementation gates

Report inherited experience and new work separately. The parent already paid
10,000 requests; this is an extra-budget continuation, not equal-cost superiority
over its frozen baseline or a new three-seed sweep/PPO benchmark. Show actual
release count, request count, geometry proposals/solver work, demo presentations,
optimizer steps and wall/aggregate CPU/device measurements by phase. Keep bank
preparation, imitation, PPO, shared initialization, assessment and independent
audit costs separate, with overlapping timers identified. RSS is lifetime high
water; Torch memory and whole-device occupancy have different scopes.

Use **one native application start** for TRAIN and final assessment together,
not one startup per method. The inherited diagnostic schedule is exactly
`FilmShots.exposed_rim_probes()` followed by `ParallelShots.probes()`, as present
in producer `5edcc89ed9c5f6f7b645fe9f070cd35722a92bf3`:

- Up to eight rim-ray cohorts at azimuths `0,90,180,270,45,135,225,315`, with
  the original early stop, 35 ball placements and 48 physics steps each.
- Five contact cohorts (`bucket,rim,table,floor,robot`), 35 placements and
  240 physics steps each.
- One foreign-cell isolation cohort, 240 steps: env0's ball is the probe and
  the other 34 balls are parked. All 35 initialized ball states are logged.

This is at most 14 diagnostic cohorts, 490 initialized non-task probe-ball
trajectories and 1,824 simulated steps. Parking resets are logged separately,
not presented as learned shots. No additional or repeated startup probe, smoke
release or automatic native restart is permitted. Report actual diagnostic
cohorts, active probes, placements, steps, CPU and wall cost separately from
task releases. Failure stops the run; it does not buy a second diagnostic pass.
The reference file identities, also bound by the completed native audit, are:

```text
film_cylinder_farm.py 849b1af800a0f13a73af8845d39ec30117f8f01648effeb9698b5891af4efe8d
parallel_runtime.py  33261c6eadbc7d0f4f823caaedb80f8e95f08229da5d545c688043d945b1b25d
```

The future entry point will have its own producer hash; its use of these helper
bodies and this finite schedule must be independently compared with that exact
reference. A changed startup procedure requires a new reviewed protocol.

Proposed limits: at most 4 CPU qualification workers; 30 minutes new support
preparation wall and 2 aggregate CPU-hours; 10 minutes imitation wall; 2 hours
native TRAIN wall; 30 minutes native assessment wall; 10 minutes CPU audit wall;
one exclusive 4090 job, waiting at most 2 hours for an idle GPU and scene lock.
Shared application initialization/diagnostics have a 10-minute wall cap, and
the sole native application has a 3-hour absolute wall cap including all phases.
Keep the original 6 GiB storage floor. Do not delete evidence, interrupt foreign
workloads, touch the 4070, restart containers/readers, or access physical arms,
Viam/cloud/audio to make progress. Ordinary per-request admission/settling/release
declines are outcomes to count and continue, not experiment-integrity failures.
A wall/storage/source/guard-integrity or nonfinite-data failure stops the pilot
and preserves all evidence. Before TEST begins, such a failure leaves it unopened;
after TEST begins, it preserves the partial TEST ledger and stops without
retrying or replacing any consumed request. No incomplete pilot is promoted.

Before any future execution, the following still require implementation and
verification; source inspection alone has not completed them:

- A differentiable fixed-candidate categorical adapter, context-balanced
  imitation objective, categorical PPO updater and new checkpoint/ledger schema.
- No-update tests for exact checkpoint loading, immutable candidate encodings,
  inclusion of the radius prior, original selector parity at initialization,
  valid finite probabilities/gradients/entropy, empty/singleton support,
  categorical ratios, frozen rollout support, new RNG/counter provenance and
  rejection/release cap handling including a partially enabled last batch.
  Initialization checks bind exact model bytes but use a declared CPU/GPU
  numerical tolerance for means and log probabilities; they must still verify
  the candidate/action ranking under the fixed tie rule. The source GPU packet
  supplies each exact original parent's immutable latent.
- CPU-only split/support preparation using a separately versioned generator;
  the original fixed benchmark split validator must not be weakened or relabelled.
- Independent audit of demo-to-bank lineage, unchanged original guards and
  scorer, every selected request and terminal reward, new probability/update
  receipts, 420-request final ledger and phase cost scopes.
- Separate source snapshot, untouched output paths, actual runtime/dependency,
  GPU-ownership/storage checks and bounded launch/termination ownership. The
  metadata export used Torch 2.7.0+cu126; generic documentation is not proof of
  the future producer environment. No package/environment changes are implied.

The existing updater and two-cohort native runner do not implement this plan.
Do not invoke either as a shortcut, or treat a new checkpoint with old schema
labels as a valid continuation. The completed benchmark, bank, raw evidence,
saved checkpoints and read-only interfaces remain unchanged.

HALO's [expert/distillation/refinement pipeline](https://arxiv.org/html/2611.06908v1)
motivates learning from demonstrations while separating execution from planning.
Its humanoid results are not evidence that this Sircus proposal improves capture.
Board-goal conditioning, perception, material development and physical deployment
remain outside this proposed pilot. Training remains off until separately
authorized after this plan and its implementation are reviewable.
