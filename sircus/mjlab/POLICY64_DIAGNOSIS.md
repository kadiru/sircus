# Why did PPO underperform imitation?

Completed [results, per-layout deltas and native variant counts](../../baselines/mjlab-policy64-diagnosis-20261106/README.md):
all 96 slots, 64 native releases, zero physics-invalid releases and 164,270 passed
independent checks. This protocol and producer were frozen before new outcomes.

This is a post-hoc diagnostic of the completed [fixed64 study](POLICY64.md),
not another training run. Its original 50 TEST positions now serve as diagnosis
and development data. Any later tuned model requires newly frozen holdouts.
Original scores, checkpoints, fingerprints, scoring rules and earlier 128 results
remain unchanged. Noise and 128 checks are outside this task.

## Original paired evidence

Imitation captured 20/50; validation-selected PPO captured 18/50. Both captured 13,
PPO gained 5, and PPO lost 7 imitation captures. Five losses became setup rejects
(04,13,21,26,43); two remained native flight misses(07,17). On 37 jointly admitted
layouts, BC captured 15 and PPO 18, a conditional description that does not remove
any rejection from the primary 50 denominator or establish a policy advantage.

PPO lowered the board on all 50 TEST positions, by 1.5–10.2mm, and increased the
release gap on 44. There is no BC-retention penalty. These are plausible mechanisms,
not proof that a particular hyperparameter caused the result. TEST shaped reward
also fell, so the observed comparison does not support a reward-gaming explanation.

The CPU analyzer reads authenticated source records, admission proofs, saved
optimizer/checkpoint states and original dense traces. It reports five-dimensional
action deltas, geometry evidence, common-reference paired flights and first-flight
measurements without extrapolating past bucket/other contact. A missing rim-plane
crossing is unknown, not a fabricated landing point. BC/PPO likelihood ratios and
KL are computed after each six-epoch update; likelihood clipping is distinct from
physical-action rejection.

## Frozen bounded native plan

Use only the 4090 UUID and the existing shared scheduling lock, at 64 substeps.
At most 96 requested decisions or 3,600 acquired-slot seconds:

- The seven historical BC-only capture layouts listed above.
- For each, replay BC and PPO requested geometry three times each. Preserve exact
  authenticated q10/q14 for original native rows. Retain original PPO rejects as
  rejected slots; their placeholder joints are never executed. This is repeated
  evidence bookkeeping for rejected slots, not three fresh solver attempts.
- For each, start from PPO and replace X,Y,height,tilt,gap individually with BC's
  value, then replace height+gap together. These 42 modified requests require fresh
  unchanged finite IK, strict 5 mm clearance and sampled setup connection proofs.
  They are post-hoc interventions, not original PPO inference or new TEST scores.
- Three exact positive-control and three native-miss-control replays before and
  after: 12 controls. Total 42 original replay/rejection slots + 42 swaps + 12 controls.

Freeze the complete plan, source/checkpoint/analysis bindings and original parent
proof before outcomes. Same geometry, materials, ball, calibrated scorer and
retention requirement. Preserve effective native/Warp options, actual initial
ball and joint states, dense contact traces and scorer/oracle checks. Nominal
repeats test reproducibility, not noise robustness. Admission is not driven
actuator-transit validation. No training, physical access, 4070 changes, Isaac
changes or service restart accompany this diagnostic.

Write a new native output namespace and independently verify every request,
source row, copied proof, cache association, dense score and manifest. A separate
derived combined report binds the frozen CPU analysis to the native report. The
UI's additive `policy64_diagnosis` / `#policy64Diagnosis` panel requires the matching
complete new audit before marking the diagnosis verified. The old policy panel
and interactive solver are preserved. Raw observations and host state stay out
of Git; publish scalar/nominal results and checksums only.

## A learned physics model or WAM

The current BC/PPO actor is already a neural network. It maps geometry to shot
settings; it does not explicitly predict the ball's next state or trajectory.
Our proposed next comparison is a small ensemble forward model: geometry and
shot settings predict post-impact velocity, aperture error, whole entry and
retention. Use uncertainty to prioritize native verification. Preserve exact
IK/clearance checks separately; a dynamics predictor cannot make an infeasible
robot posture feasible.

This proposal follows the uncertainty-aware dynamics/planning idea in
[PETS](https://arxiv.org/abs/1805.12114). [MBPO](https://arxiv.org/abs/1906.08253)
also illustrates why predictions must be checked for model bias and long-rollout
error. Neither paper establishes success on Sircus. Compare with BC, frozen PPO
and bounded native search on new layout/family-separated holdouts; predictions
must never be counted as actual simulator captures.

A video world-action model jointly predicting observations and actions is a
larger alternative; [NVIDIA's WAM overview](https://developer.nvidia.com/blog/beyond-vlas-how-world-action-models-reshape-robot-manipulation)
describes that approach. Our recommendation to start with a compact state-based
physics ensemble is an inference from this task's low-dimensional geometry and
available numerical traces, not a measured WAM comparison. No WAM, dynamics model,
or new policy is trained during this diagnosis.
