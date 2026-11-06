# PPO versus imitation diagnosis — November 6, 2026

**The main observed loss was setup feasibility.** Of seven imitation captures lost
by PPO, five became setup rejections and two were native flight misses. The
original scores remain BC 20/50 and PPO 18/50; no policy or WAM was retrained.

The [frozen protocol](../../sircus/mjlab/POLICY64_DIAGNOSIS.md) completed 96 requested
slots / 64 actual native releases in 406.8 reported worker-seconds. All 164,270
independent checks passed; positive/native-miss controls passed before and after.
No physics-invalid releases occurred. The 12 control slots are separate from 84
diagnostic slots; three retained rejection slots do not mean three native throws
or new solver attempts. Aggregate captures are not a new policy score.

## Exact repeats and controlled interventions

| Case | Original PPO failure | BC exact native captures | PPO exact native captures | Single-coordinate replacements that captured once |
|---|---|---:|---:|---|
| 04 | strict_endpoint_clearance_or_pose_rejected | 3/3 | no release (3 retained rejected slots) | height |
| 07 | native flight miss | 3/3 | 0/3 | gap |
| 13 | finite_ik_unresolved | 3/3 | no release (3 retained rejected slots) | height, tilt |
| 17 | native flight miss | 3/3 | 0/3 | X, height, gap |
| 21 | strict_endpoint_clearance_or_pose_rejected | 3/3 | no release (3 retained rejected slots) | height |
| 26 | finite_ik_unresolved | 3/3 | no release (3 retained rejected slots) | height, tilt |
| 43 | strict_endpoint_clearance_or_pose_rejected | 3/3 | no release (3 retained rejected slots) | height, tilt |

Every original admitted BC branch captured 3/3 at all seven selected positions
(21/21 actual native releases). The two original admitted PPO misses reproduced
0/3 each (0/6). Five originally rejected PPO actions remain rejected in 15 slots;
no placeholder joint vector was executed. Original branches/geometry were frozen.

The 42 fresh modified requests replaced one PPO coordinate with BC's value for
X/Y/height/tilt/gap, or height+gap together. Every modified request passed the same
finite admission procedure or remained rejected. These are post-hoc interventions,
not original PPO inference. Each replacement was tested once: a 1/1 result does
not establish a robust window or a general correction rule.

## Concrete examples

- **04, wrist/table clearance:** PPO requested board height 250.800mm versus BC's
  258.898mm. Its best checked wrist/table clearance was 3.832mm against the unchanged
  5mm requirement; BC's was 7.065mm. Restoring height alone (+8.098mm), keeping all
  other PPO shot coordinates, admitted the setup and captured cleanly 1/1. X,Y,
  tilt or gap alone remained rejected. Height+gap also captured cleanly once.
- **13/26, finite IK:** a height-only replacement (+3.813/+3.726mm) or a tilt-only
  replacement recovered one capture each. The original fixed-budget solver's
  failure is not proof that no physically reachable joint configuration exists.
- **07, bucket contact/flight:** BC's gap was 234.511mm, PPO's 247.360mm. BC captured
  3/3; PPO hit the bucket without whole-ball entry and captured 0/3. Restoring only
  the 12.849mm shorter BC gap captured once. Other single swaps and height+gap missed.
- **17, parameter coupling:** BC 3/3 versus PPO 0/3. Restoring X, height or gap alone
  captured once, but restoring height+gap together missed. Successful recipes depend
  on interacting pose, release and contact parameters.

The UI compares actual recorded BC/PPO flights in one bucket-origin projection.
Missing/rejected flights are not fabricated. Displayed prefixes stop at capture
or the scoring fall cutoff, with shown/raw sample counts; full recordings stay
unchanged. All positions/heights/settings in the CSV are requested nominal scene
geometry, not hardware calibration. The 175 mm bucket-height interpretation remains
provisional, and all previous 128 failures retain their original definitions.

## What the optimizer evidence does and does not show

PPO lowered board height at 50/50 original TEST layouts and raised gap at 44/50.
Mean analytical post-update KL was 0.3022 nats. After each six-epoch update, an
average 81.64% of sampled likelihood ratios were outside [0.8,1.2]. That is a
policy-change diagnostic, not an invalid-action rate. PPO has no explicit penalty
for departing from the original BC actor. These facts support testing feasibility
constraints and more conservative updates; this experiment does not establish
which hyperparameter caused the drift. TEST shaped reward also worsened.

## A physics model / world-action model next

The current actor is already a neural network, but predicts actions rather than
physics. We propose comparing feasibility-aware shot selection with a small
uncertainty-aware forward-model ensemble predicting rebound motion, aperture
error and capture/retention. Keep exact IK/clearance checks separate. This is a
proposed next experiment; no dynamics model, WAM or new policy was trained here.

References: [PETS](https://arxiv.org/abs/1805.12114),
[MBPO](https://arxiv.org/abs/1906.08253),
[NVIDIA's WAM overview](https://developer.nvidia.com/blog/beyond-vlas-how-world-action-models-reshape-robot-manipulation).
They motivate a comparison; they do not establish Sircus performance. The original
50 TEST coordinates are now diagnosis/development data. Future tuned models need
fresh, frozen, layout/family-separated holdouts excluding all earlier split and
diagnostic positions. A model-predicted success must be verified in native physics.

## Artifacts and verification

- [Sanitized summary, optimizer scalars and proposal](summary.json).
- [All 50 paired nominal configurations/outcomes/deltas](pairs.csv).
- [All 56 variant groups representing 84 diagnostic slots](native-variants.csv).
- [Audited-copy checksums](checksums.json) and [publication checksums](SHA256SUMS).
- [Original study and exact model weights](../mjlab-policy64-20261105/README.md).
- [Read-only UI](http://127.0.0.1:18781/#policy64Diagnosis).

Frozen producer commit: `5d025073cbb6105f8404d1ed6f6ae95046354c32`.
Frozen archive SHA256: `55adfe6dab498bd518ea723ab9175cd473ac79033d23e21865068290a8730f94`.
Original native source: `f00a29a42cac69221b0896173f9be785f113a8a6`.
Diagnostic contract: `bae99c785e5bec2d51f303bdce25e498434e5b8e4ebec6b836a4ace0cd1e8dcc`.
Independent proof: `b40d30748b978b0804c8b67f1ae07b8bb32212984510223fe15741ae1276564a`.

Raw joints, poses, dense trajectories, contacts, logs and host configuration are
excluded. The full mock/headless suite, 21 producer/verifier checks, five publisher
checks, five curator checks and eight Node UI checks passed. No new 128/noise,
perception, driven transit or physical-validation claim is made. Only the 4090 ran
these native diagnostics; its worker exited. The 4070 Isaac workload was left
untouched and the existing viewer was not restarted.
