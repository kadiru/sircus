# Geometry-guided continuous PPO pilot

This separate follow-up addresses the broad pilot's 1,214 admission failures in
1,600 training attempts. The original study evidence remains frozen; its former
localhost:18794 listener was unavailable and was not restarted. The [completed
study and portable checkpoints](../../baselines/isaac-feasibility-guided-pilot-20261105/README.md)
show fewer reach/IK failures but worse capture performance.

## Proposal method

The existing full robust IK geometry map supplies 155 endpoint candidates at
33 of the 35 inherited bucket coordinates. At each coordinate, a geometry-only
rule selects one anchor with a sampled canonical-home path proxy: closeness to
the declared H350 mm / gap240 mm / tilt29° / workspace-reference prior plus a
small joint-travel cost. No contact, capture, reward or validation outcome enters
that choice.

The actor outputs five continuous residuals. A fixed per-context affine decoder
maps them into a local box around the geometry anchor. Initial half widths are
5° bearing, 25 mm radius, 25 mm board height, 60 mm gap and 6° tilt. Sixteen
prespecified Sobol probes check each candidate box; widths shrink through
1, 1/2, 1/4 and 1/8 until at least 75% pass endpoint/path proxies or the budget
is exhausted. All failed probes remain recorded. The smallest fallback can
remain below the target, explicitly marked.

Preparation retained 1,040 probes with 693 qualifications overall. The selected
boxes had 452/528 qualifications (85.6%); five boxes fell below the 75% target.
Two sites remain unsupported, not proven impossible. Preparation cost was
21.58 seconds wall and 116.31 user CPU seconds using eight workers; maximum RSS
223,128 KiB is not an aggregate worker-memory peak. This is additional geometry
preprocessing, not free RL data or a successful-shot search.

Every native action still receives a fresh full robust IK solve seeded by the
anchor, independent FK, sampled path proxies and unchanged measured settlement
checks. Seed joints are never substituted for a requested endpoint. Empty
support is an invalid attempted action and stays in the optimizer/denominator.
There is no nearest-action projection, capture lookup or rejection resampling.
Finite probes do not certify every box point, mesh collisions or self-collision.

## Learning and comparison

A distinct checkpoint contract uses a fresh 2×64 actor/critic, zero normalized
local-residual mean and the same seed, Gaussian log standard deviations, PPO
hyperparameters and reward as the broad pilot. The geometry-conditioned decoder
changes physical action support and exploration scale. It is not a fine-tune of
the old global-domain checkpoint or an experiment isolating only PPO.

The completed matched budget was 64 × 25 = 1,600 attempted training shots, 16 updates,
25 training/five validation/five test coordinates, and eight nominal frozen
initial/final repeats per site. Invalid requests remain in the reward buffer.
Report guided initial→final and broad final→guided final separately so the
geometry prior's benefit is not attributed to reinforcement learning.

Held-out geometry assists the decoder at those known coordinates. Held-out
outcomes never update the actor or support. This is a geometry-assisted,
transductive comparison on inherited coordinates, not globally unseen layouts
or robustness to noise. Identical repeats are numerical repeatability tests.

PPO likelihoods and entropy are in normalized residual coordinates. The local
physical affine Jacobian is constant per context and cancels in PPO ratios;
the independent auditor reconstructs every physical action and checks the
support, domain, geometry, optimizer membership and fresh initialization.

## Native execution and recording

The physics, strict board-bounce/whole-ball-entry/one-second-retention scorer,
canonical episode home, native measured release gates, saved camera and
silver-gray scene remain unchanged. Four geometry-coverage and sixteen
prescheduled early/middle/late PPO batches are filmed at native 30 fps. Episode
resets precede filming; visible setup uses articulation drives, without scored
endpoint teleports. Arm travel and zero-speed initialized ball loading remain
scripted. No camera follows the ball and failures are not edited out.

The board remains 200×300×5 mm in landscape orientation. Bucket diameters
remain inside 177/150 mm and outside 200/158 mm. Reported 175 mm height still has
unresolved overall-height versus interior-depth meaning; interior 175 mm plus
8 mm bottom remains the provisional simulation interpretation.

## Commands and isolation

Use the pinned environment in [README](README.md), the existing shared scene
lock and dedicated 4090 supervisor. Do not start a competing cloud part, change
hardware, use the 4070 workload or restart prior services.

```bash
CUDA_VISIBLE_DEVICES='' python feasibility_guided_support.py \
  --map MAP/feasibility.json --source-spec SOURCE/contract.json \
  --assets ASSETS --output NEW_SUPPORT --workers 8

# Through the bounded 4090 launcher, append to continuous pilot arguments:
# --feasibility-guided-support NEW_SUPPORT/support.json
# Use --continuous-batches 4 for a separate smoke, then 64 for the pilot.

CUDA_VISIBLE_DEVICES='' python feasibility_guided_audit.py \
  --run RAW --executed-source PRODUCER_SNAPSHOT --output NEW_CHECKPOINT_AUDIT
CUDA_VISIBLE_DEVICES='' python continuous_native_audit.py \
  --run RAW --output NEW_NATIVE_AUDIT
python feasibility_guided_report.py RAW NEW_EXPORT \
  --baseline-report FROZEN_BROAD_REPORT --media --action-edit
python feasibility_guided_server.py --output NEW_EXPORT --port 18795
```

Keep raw observations, generated assets, media and runtime configuration outside
Git. Compact models and independently audited evidence are in the linked baseline.
The original broad study and scoring fingerprints remain separate. No physical, Viam or audio deployment.
