# Cooperative shot learning: privileged state first, cropped clouds second

**Completed:** both runs finished. See [frozen results and next-stage decision](../../baselines/isaac-privileged-study-20261031/README.md). Quasi-perception training is deferred because privileged coverage is not yet reliable. The active-run notes below are historical.

This is a new Isaac/PhysX study. Historical numerical models, fixed-scene RL,
and perception validation retain their original definitions and results.

## What exists and what is being tested

The original 256-throw fixed-scene experiment used REINFORCE to fit three
state-independent action means. Its frozen 8/8 result repeated one deterministic
layout; it did not demonstrate generalization, and its heuristic also succeeded.
The separate frozen RGB-D detector completed 144 observations (24 layouts × six
conditions). Clean performance was good, but board detection failed under changed
lighting or 2mm depth noise. Neither experiment establishes camera-driven control.

The new learner is a **centralized cooperative, one-step contextual PPO policy**.
Both arms share a reward and a single action. It is not adversarial learning,
independent multi-agent PPO, or a learned pick-and-place controller. Isaac Lab
2.3.2 / Isaac Sim5.1 and actual PhysX ball contacts execute each throw.

## Observations and actions

The 11 normalized privileged inputs contain board center XYZ, bucket offset XY,
bucket rim height and inner diameters, board XY size, and ball radius. Object poses
come from the episode configuration, not a camera. Reward, capture status and future
trajectory never enter observations.

The actor chooses board tilt12–45° from horizontal, heading offset−15–15°, and
release gap60–250mm above the board (plus ball radius). A geometric prior already
points the board toward the bucket; learning refines it. Release XY is above the
board center. U/V exploration and full joint trajectories are not part of this stage.
Board center+normal IK leaves in-plane spin free; dropper IK constrains its full pose.
Unreachable endpoints count as failures, not excluded layouts.

Arms are initialized at solved endpoints, settle0.5s, and the ball is spawned at
release. Simulation advances at240Hz for up to4s. This does **not** validate travel
collisions, physical grasp/release, or fast board interception. The board is rigid;
its real suction compliance is not modeled. Ball properties remain provisional.

## Reward and success

The unscaled return is:

`board_bounced + 2 exp(-closest_bucket_distance / 0.12m) + 2 min(longest_retention_seconds,1) + 10 success`

An invalid endpoint returns−2. Ordinary misses can earn partial positive reward;
there is no fabricated negative reward for every miss. Dense proximity and retention
help exploration but are not reported as success. The critic trains on return/15.
Success requires filtered **board contact and rebound**, whole-ball entry, then one
continuous second contained. Rim-assisted capture counts; a partial exit resets
retention. Scoring uses simulator truth, separately from policy observations.
Bucket reported height175mm remains unconfirmed; the current simulation assumes
175mm interior depth and8mm floor thickness. Previous fingerprints are unchanged.

## Training and evaluation protocol

- Prespecified seeded pools:1,000 training layouts,128 validation,256 test. Board
  centers vary X640–700,Y−140–−80,Z360–420mm; bucket distance200–400mm with
  ±0.25rad lateral direction. Near/medium/far bins split at267/333mm.
- First comparison:100 versus1,000 training layouts, each4,096 exploratory throws,
  initialization seed42. Each run has an independent simulator/GPU. Equal throw
  budgets isolate scene diversity, not compute-free extra training. One seed is
  insufficient to claim statistical superiority; seed replication is the next gate.
- Training visits shuffled layouts before repeating. Actor and critic each use
  two64-unit tanh layers. Actor has three outputs and learned diagonal Gaussian
  exploration; tanh maps latent actions into bounds.
- PPO: Adam0.0007, clip0.2, batch64 throws,4 epochs/full batch, value coefficient0.5,
  latent Gaussian entropy coefficient0.001, gradient norm≤1, gamma1 for terminal
  one-step returns. Initial log standard deviation−1, clamped[−2.8,−0.2].
- Warm start uses the previous fixed-scene action(21.359°,−0.941°,167.692mm), not a
  per-scene optimizer. Initial actor output is constant; subsequent weights can
  depend on scene. A constant warm-start policy and midpoint policy are baselines.
- Every512 throws, evaluate deterministic actions on32 validation layouts twice.
  Select by captures, then mean return. Repeats expose contact sensitivity; they
  are not independent layouts. Independent-sample confidence intervals are omitted for repeated validation rows.
- Freeze the selected checkpoint, then evaluate128 validation and256 unseen test
  layouts, plus both fixed baselines on the same test pool. No reward search or
  gradient update occurs at inference. Invalid IK remains in the denominator.
- Report capture rate, Wilson95% interval on distinct test layouts, distance bins,
  invalid endpoints, actual parameter changes, and validation learning curve.
  Training capture rate is exploratory and cannot be compared directly to frozen
  deterministic test capture rate. Contact-sensitive repeats may differ.
- Development smoke pools use separate seeds and never consume final test layouts.
  Physical floor probes verify bucket collision geometry follows randomized poses.

Why not start at10,000 scenes? Scene count is not attempt count. First establish
that the policy learns and beats useful baselines on unseen layouts. If1000 layouts
improve over100 and learning curves still rise, expand data and seeds. If they do
not, diagnose reachability, reward exploitation, action expressivity and model
capacity before spending more compute. Also test restitution, mounting compliance,
ball sizes, lateral offsets, bucket height/size and observation occlusion; distance
alone need not determine difficulty.

PPO reference: [Schulman et al.](https://arxiv.org/abs/1707.06347).

## Quasi-perception gate

Only begin cloud-based training after reviewing the privileged study. Crop **actual
rendered depth points**, using privileged object pose/geometry plus padding, into
ball/board/bucket clouds. Preserve occlusion, missing depth, nearby background and
measurement noise. Do not synthesize unseen ideal surfaces. Cropping is oracle
segmentation and must remain labeled as such, not real detection.

Capture inputs before the chosen action so board orientation cannot leak the answer.
Keep truth/labels in separate artifacts. Test crop-padding and pose-error sensitivity;
missing clouds must remain missing, without ground-truth fallback. A practical next
step is a point-cloud encoder with supervised teacher distillation, followed by
closed-loop frozen evaluation and optionally PPO fine-tuning. Distillation itself
is not RL. Both the policy **and its pose/heading controller** must consume estimated
geometry; leaving ground-truth bucket heading in the controller would hide leakage.

## Reproduction

Inside the isolated container (exclusive scene lock):

```bash
flock -n /workspace/scene.lock /workspace/source/sircus/isaac/python.sh \
 /workspace/source/sircus/isaac/launch_scene.py --headless --enable_cameras \
 --device cuda:0 --general-episodes 4096 --train-layouts 1000 --rl-seed 42
```

Use `--general-episodes 16 --train-layouts 16 --study-smoke` for development checks.
Runs save contract/splits, episode JSONL, initial/latest/selected checkpoints,
validation curves and separate frozen evaluations beneath the runtime output.
These are local synthetic evidence, excluded from Git. No automatic hardware/cloud
deployment. Checkpoints do not yet support exact optimizer/RNG resume.

## Development checks and known confounds

The first generalized v1 smoke is **invalid as a randomized physics experiment**:
editing a static bucket USD transform did not move its PhysX colliders. The v2 floor
probe caught this before large studies. Contractv3 uses a kinematic compound body;
three physical floor probes at different XY positions passed. Eight repeated
identical development inputs had zero capture disagreements on each tested GPU.
This is a reset check, not a guarantee of cross-device deterministic contacts.

The corrected4090 and4070 smokes completed16 training throws and frozen8-layout development
checks (6/8 validation,5/8 development test,4/8 warm fixed,2/8 midpoint). These tiny
checks verify execution/checkpoint loading; they do not establish RL superiority.
All final train/validation/test pools remained unused by the smoke.

Two different GPU models and one initialization seed are confounds in the initial
100-vs1000 comparison. Treat the result as exploratory; reproduce selected policies
on the same device and repeat training seeds before attributing differences to data
volume. Do not tune hyperparameters on final test results.

Long jobs use `study_job.sh` with an immutable source copy, an exclusive scene lock,
a6-hour timeout and a3GiB free-space floor. They save selected checkpoints and final
frozen evaluations automatically. The live loopback preview can read both outputs
with `--secondary-output`; it has no robot controls. Cloud-based training is not
started automatically merely because a privileged job exits successfully: its
results and reachability/failure breakdown must first support the next stage.

## Active run references (20:50 UTC, October31)

4090: `general-rl-20261031T205020Z-n1000-seed42` under
`~/sircus-isaac/output/`.4070: `general-rl-20261031T205024Z-n100-seed42`
under `~/sircus-isaac4070/output/`. Runtime containers are `sircus_isaac_4090`
and `sircus_learning_4070`. Both run the copied `studies/privileged-v3-20261031`
source tree; `SIRCUS_ISAAC_SOURCE` selects it without changing other runtimes.
Read `progress.json`, `summary.json`, and `contract.json` in each run directory.
Do not infer process liveness from this historical note. Final study results were
pending when this checkpoint was published. No cloud-trained checkpoint exists yet.
