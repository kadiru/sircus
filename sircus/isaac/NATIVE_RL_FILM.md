# Native RL training film from the selected farm camera

This is a separate, finite reinforcement-learning study in Isaac Sim 5.1 / Isaac
Lab 2.3.2. One shared policy selects shot configurations for 35 tables and 70
articulated arms. Native PhysX outcomes provide the rewards, and the filmed
on-policy shots enter the optimizer. The [previous one-minute film](PRESENTATION_FILM.md)
remains a saved-configuration demonstration with no new optimizer updates.

The recording uses the exact saved perspective camera and the existing
silver-gray stage. It does not follow the balls. The foreground table stays at
source cell `r1c2`, in the same whole-cell display position chosen by the user.
Camera imagery does not feed the policy: the policy uses privileged simulator
geometry. Grasping, perception and full joint-path control are separate tasks.

## Policy and the available physical choices

Each of the 35 distinct, fixed bucket sites has 24 geometry-only,
native-qualified configurations. Each option specifies five linked quantities:

1. Board bearing around the bucket.
2. Horizontal board-to-bucket radius.
3. Board-center height above the tabletop.
4. Vertical ball-bottom gap above board center.
5. Board tilt from horizontal, with heading prescribed toward the bucket.

The actor scores the physical geometry of each candidate, then samples one of
24 categories. It does not receive candidate IDs, saved capture outcomes or
old policy weights. Its final layer starts at zero, giving exactly uniform
initial probabilities of `1/24` at every table.

The state has the existing five bucket-geometry features: normalized X/Y,
distance to the common reference, and sine/cosine of reference bearing. Each
candidate has ten features: its five globally normalized physical parameters,
then the same five parameters normalized within that table's qualified range.
The local coordinates resolve small changes that would otherwise be tiny
compared with the global limits. Constant dimensions receive local value zero.

One shared candidate actor consumes these 15 features through two 64-unit
`tanh` layers and produces one logit per candidate. A separate shared critic
consumes only the five state features. Candidate permutation reorders logits
without introducing an action-index shortcut. The checkpoint policy version is
`sircus-categorical-shot-policy-v1`.

The available choices are **nearby qualified operating regions**, not a broad
continuous sweep. Across the existing menus, per-table spans are approximately
20–40 mm of board height, 20–40 mm of drop gap, 1–4 degrees of tilt and
3.3–20 mm of horizontal radius. The selected foreground menu has board height
360–400 mm, gap 245–285 mm, tilt 25–27 degrees and radius about 396–406 mm.
Other tables occupy different absolute regions. Broad original action limits do
not mean this policy explores every point inside those limits. Report actual
selected-category counts, entropy and executed geometry ranges separately from
menu availability.

## Learning budget and reward

| Quantity | Prespecified value |
|---|---:|
| Training batches | 128 |
| Attempted shots per batch | 35 |
| Training shots entering PPO | 4,480 |
| Batches per update | 2 |
| PPO updates | 64 |
| Attempted slots per update | 70 |
| PPO epochs per update | 4 |
| Adam learning rate | 0.003 |
| Clipping range | 0.2 |
| Value / categorical entropy coefficients | 0.5 / 0.01 |
| Gradient-norm limit | 1.0 |
| Training seed | 20261110 |

Each shot is one terminal decision. PPO uses the observed scalar reward, a
state-value baseline and normalized advantages, with no next-state bootstrap or
GAE. The categorical action log probability is stored with its sampling
checkpoint. Every attempted slot enters its two-batch update, including a
release withheld by an admission guard. Actor and full actor/critic hashes are
tracked separately so critic-only changes cannot establish actor learning.

The existing reward and capture evaluator are unchanged. An invalid attempt
receives `-1`; a detected external route receives `-0.5`; a shot without a board
bounce receives `-0.25`. Other shots receive the existing opening-distance and
retention shaping, capture reward, and clean/single-bounce bonuses. A qualified
capture still requires a board bounce, whole-ball entry and one continuous
second inside the bucket, without a detected external intermediate contact.
Rim-assisted captures count; partial exits reset retention. Audio, manual labels
and the editor never determine the reward.

The numerical policy runs in float32 with TF32 disabled and highest matmul
precision. Checkpoints bind architecture, study/geometry/menu hashes, update and
training-batch counters, optimizer state, and separate actor/full weight hashes.
The CPU audit loads them with `weights_only=True` and independently reconstructs
all sampled categories' likelihoods and entropy. It verifies the logged update
membership and hash chain; it does not independently reproduce optimizer arithmetic.

## Frozen checks on the same known layouts

Assessments occur after **0, 32, 64 and 128 completed training batches**, when the
policy has completed 0, 16, 32 and 64 updates. Each stage uses eight independently
seeded stochastic batches of 35 shots: **280 attempts per stage and 1,120 total
assessment attempts**. The same eight assessment seeds are reused across stages
for comparison; they are separate from the training sampling stream.

Assessment weights stay frozen, no optimizer is run, and none of these attempts
enters the training buffer. No search selects a better shot for the assessment.
The policy samples from its learned categorical distribution using the fixed
qualified geometry features. Checkpoint reloads also preserve training RNG state.

These are checks on the **same 35 known bucket layouts**, not unseen-layout
validation or a disjoint spatial test. The curves include batch, cumulative and
last-eight-completed-batch capture rates. Their denominator includes all
attempts, including rejected releases. Assessment rates include nominal Wilson
95% intervals and remain distinct from on-policy training rates.

Actual rates may rise, stay flat or fall. The foreground table already has a
strong qualified operating region and may show a ceiling effect while other
sites remain difficult. A changed actor hash or higher shaped reward alone does
not prove better capture performance. Preserve the initial, intermediate and
final results even if they show no improvement.

## Common setup and actual training footage

Every scored episode uses the same canonical endpoint initialization followed
by the unchanged measured settling/release checks. Filmed episodes first show
the existing articulation drives moving toward the selected configuration. That
optional visual transit is followed by the same canonical initialization used
for unfilmed training and assessment. The board and ball then evolve under
fresh PhysX simulation. The setup protocol is explicitly versioned as
`scripted-visual-transit-then-common-canonical-endpoint-init-v1`.

This preserves a common release-setup protocol rather than comparing scores
from different initialization methods. It does not establish learned pickup,
continuous learned joint trajectories or collision-free paths. The filmed
36-frame setup compresses scripted travel; the video is not a real-time
recording of the controller's full execution.

The 24 filmed training batches are selected **before outcomes are known**:

| Film chapter | Zero-based training batches | Sampling updates |
|---|---|---|
| Early | 0–7 | 0–3 |
| Middle | 60–67 | 30–33 |
| Late | 120–127 | 60–63 |

These actions are collected on-policy and subsequently consumed by PPO. They
are not replays of successful actions. The unfilmed training batches still have
full outcome and native-state evidence. Explicit chapter labels identify the
iteration jumps; the film does not suggest all 128 batches were recorded.

Each selected chapter retains arm setup, release, board bounce and the immediate
outcome. Quiet tails may be cut while full one-second retention proof remains
in telemetry. Misses and rejected attempts remain in the scheduled sequence;
a rim touch alone cannot terminate a chapter. All retained imagery comes from
the native camera. There is no ball tracking, looping, padding, generated
replacement imagery or outcome-dependent chapter selection.

The overlay is a declared data-derived layer: training phase, sample/update
counts, measured capture-rate curves and separately labeled known-layout
assessments. During setup and flight it shows only completed prior training
batches. The current completed-batch summary is revealed at the chapter end,
using its full saved rollout; no later training batch is used. Assessments are
shown only once their corresponding training stage is complete. This is an
edited training summary, not a live controller display.

The export retains all 24 scheduled chapters at 30 fps. It preserves 1× flight
sampling when the action edit is at most 90 seconds; a longer edit can be
uniformly shortened by at most 1.10×. It never stretches a short film to claim
one minute. Actual duration and every source-frame mapping are recorded.

## Geometry, fingerprints and evidence

The original geometry contract remains byte-for-byte intact, with fingerprint
`63362ef37a93881f769b39d1ee7456335f1cc7bf5db04ffa7e548ce9697455bf`.
The original camera-source fingerprint remains
`98ebe47b48a01b9028747ab03bc913a033d90611e44f03b912e58f41c87ff79a`.
The separate `study.json` fingerprints this new policy, budget, setup protocol,
source hashes and filming schedule; it does not relabel the old ten-batch PPO
study or the saved-configuration film. Its actual fingerprint is reported by
the completed run rather than regenerated from changed sources.

All 35 bucket identities, arm bases, native qualification proofs and camera
bindings stay fixed. The board remains 200 × 300 × 5 mm. The bucket's 175 mm
measurement still means reported height, with overall-height versus interior-depth
interpretation unresolved. No new calibration or material claim follows.

The immutable producer source is commit
`207811f28364a2c09fedb221e8553fde0f7562b2`. Reproduction requires that pinned snapshot,
the qualified `output/expanded-training-20261102-v4` evidence, the exact selected
camera JSON and existing generated assets. All original physics, scoring and
control source hashes are checked; this study adds its own versioned helpers.

Raw evidence remains outside Git: native JPEGs, 240 Hz ball/contact/arm/board
NPZ packets, per-batch actions/rewards, checkpoint files, logs and rendered video.
The audit and exporter verify all 128 training and 32 assessment batches,
checkpoint bindings, native state identities, full-rate outcomes, fixed camera,
original/fresh probes and complete video decode before final publication.

## Configured Ubuntu launch and sequential CPU export

Use the existing dedicated-4090 container and its pinned environment from
[the Isaac foundation](README.md). Do not restart services, start another copy
of a shared cloud part, change the 4070 workload or access physical hardware.
First verify the container is restricted to the actual 4090 UUID. Acquire the
host's shared scene lock and check for a foreign 4090 process **before** launching
Isaac. Keep that lock for the entire native process lifetime. The configured
capture supervisor additionally binds the Git archive/source bytes, bounded
timeout and owned process receipts.

This reference command runs from Ubuntu with a fresh snapshot and output path.
The existing lock is opened read-only; no shared chmod or chown is needed. The
GPU UUID below is the configured host's 4090, not a host-index assumption:

```bash
film_source=/workspace/studies/REPLACE_WITH_PINNED_207811F_SNAPSHOT/sircus/isaac
film_output=/workspace/output/REPLACE_WITH_NEW_RL_RUN
qualified_source=/workspace/output/expanded-training-20261102-v4
(
  flock --exclusive --nonblock 9 || exit 1
  film_gpu_processes=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader) || exit 1
  if printf '%s\n' "$film_gpu_processes" |
      grep -Fq GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4; then
    echo "4090 is occupied; leave the existing workload running." >&2
    exit 1
  fi
  docker exec -u 0 -e SIRCUS_ISAAC_SOURCE="$film_source" \
    -e CUDA_VISIBLE_DEVICES=0 -e PYTHONDONTWRITEBYTECODE=1 sircus_isaac_4090 \
    /usr/bin/timeout --signal=TERM --kill-after=30 7200 \
    /bin/bash "$film_source/python.sh" "$film_source/film_cylinder_farm.py" \
    --headless --enable_cameras --device cuda:0 --assets /workspace/assets \
    --appearance silver --rl-study-source "$qualified_source" \
    --rl-study-camera-json "$qualified_source/selected-camera.json" \
    --output "$film_output"
) 9< "$HOME/sircus-isaac/scene.lock"
```

The output must be new. Source synchronization is not runtime deployment, and
this command must not be run alongside an existing scene. Inspect actual startup
receipts and the first native assessment/filmed clip; unit tests alone do not
establish native camera, admission or checkpoint compatibility.

After the native run completes successfully, run the **checkpoint audit first**
and the **exporter second**, on CPU with the matching frozen helpers. Both the
qualified source and output paths must be accessible on that host:

```bash
film_helpers=/home/kadir/sircus-isaac/studies/REPLACE_WITH_PINNED_207811F_SNAPSHOT/sircus/isaac
film_run=/home/kadir/sircus-isaac/output/REPLACE_WITH_COMPLETED_RL_RUN
qualified_run=/home/kadir/sircus-isaac/output/expanded-training-20261102-v4
film_cpu=/home/kadir/sircus-mjlab/venv/bin/python

CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$film_cpu" \
  "$film_helpers/rl_film_audit.py" "$film_run" --source "$qualified_run"

CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$film_cpu" \
  "$film_helpers/rl_film_export.py" "$film_run" "${film_run}-export" \
  --source "$qualified_run"
```

The audit creates a new `checkpoint-audit.json`; do not overwrite an old audit
to conceal mismatched evidence. Export requires a new directory and a passing
bound audit. It writes `rl-training.mp4`, `poster.png` and `film.json`, then checks
H.264 dimensions, cadence, chapter/frame counts, duration and a full decode.

Before launch, 48 new policy/study/checkpoint-audit/export checks were exercised.
They cover geometry inputs, genuine PPO changes, RNG/checkpoint isolation,
invalid-attempt accounting, future-data restrictions and evidence tampering.
The learnable bandit used by a policy unit test is synthetic unit evidence only;
it is never a native training result. Relevant tests can be rerun without Isaac:

```bash
python -m unittest discover -s sircus/isaac/tests -p 'test_rl_film_*.py' -v
python scripts/check-repository.py --staged
git diff --check
```

## Final native study and film result

Pending. The producer was launched from the pinned source above, but this section
makes no completion or improvement claim. After the full run, record the actual
study fingerprint, initial/intermediate/final known-layout capture rates and
intervals, training/assessment attempts and rejection counts, actor/entropy and
executed-geometry changes, checkpoint/native audits, selected film duration and
hash, visual review, transfer checks, final storage/GPU/service state and the
published task commit. Keep all older baselines identified separately.
