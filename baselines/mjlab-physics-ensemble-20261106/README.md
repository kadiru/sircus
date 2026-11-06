# Learned physics ensemble — trained; native comparison pending

The actual five-member state-based forward model is trained and saved here.
**There is no fresh held-out capture-rate result or method winner yet.**
The [read-only UI](http://127.0.0.1:18781/#physicsEnsembleStudy) separates the
completed CPU fit, preparation/queue and actual native evaluation.

## What was trained

- Five independently initialized 7→64→64→9 SiLU networks, each with a seeded
  bootstrap and 600 full-batch Adam updates at 0.001 on two CPU threads.
- The 1,024 authenticated current parent TRAIN requests at 35 layouts: 887 native
  releases, 615 captures and 137 rejected setups. One known physics-invalid
  native release has all physical targets masked; execution is still counted.
- Inputs: board XY, height, tilt, gap and bucket XY. Targets: native execution,
  bounce, whole entry, capture, retention, first-rebound opening distance and
  forward/lateral/vertical rebound velocity. Physical target support is 886.
- No validation, TEST, control or diagnosis outcomes fit the model. The bank
  includes saved BC/PPO proposals and trajectories collected by warm-started
  PPO, so this inherits historical expertise; 1,024 is direct fit exposure,
  not a cold-start total-acquisition claim.

Full-TRAIN loss changed as follows. These mixed classifier/regression losses
are not native capture rates or held-out prediction errors.

| Member | Initial loss | Final loss |
|---|---:|---:|
| 1 | 5.289431 | 0.388326 |
| 2 | 5.942647 | 0.414016 |
| 3 | 5.785951 | 0.387335 |
| 4 | 5.625964 | 0.431017 |
| 5 | 6.141310 | 0.410712 |

Independent CPU reconstruction of the real bound TRAIN targets/masks and all
five 600-step fits passed **126,235 checks**, without initializing CUDA.
That proof is explicitly training-only: it does not verify fresh predictions
or any new native outcome, and cannot satisfy the full-study UI audit gate.

## Reuse this checkpoint

[ensemble.pt](ensemble.pt) contains plain model tensors, initial states, Adam
states, bootstrap indices, normalization and fit losses. It contains no raw
observations, trajectories, host paths or credentials. Use the pinned Ubuntu
Torch environment and the [model implementation](../../sircus/mjlab/physics_ensemble_model.py):

```python
from sircus.mjlab.physics_ensemble_model import load, inputs
model = load("baselines/mjlab-physics-ensemble-20261106/ensemble.pt")
# All distances in metres; tilt in degrees. These numbers are an input example,
# not an evaluated or guaranteed successful shot.
bucket_xy = [0.6, -0.4]
shot = [0.5, -0.2, 0.3, 30.0, 0.3]  # board X/Y/height, tilt, gap
prediction = model.predict_inputs([inputs(bucket_xy, shot)])[0]
```

Capture scores and ensemble disagreement are uncalibrated model outputs.
Predictions must not be counted as simulator captures or verified feasibility.
This model is not activated in the legacy interactive solver.

## Fresh comparison in progress

The [frozen protocol](../../sircus/mjlab/PHYSICS_ENSEMBLE.md) specifies 24 diagnostic
validation and 60 TEST bucket positions at least 10 mm from 169 declared prior
benchmark/diagnosis coordinates and each other. All 252 predictions must precede
any evaluation outcome. The fixed model is not selected or tuned on validation.

Compare unchanged saved BC/PPO with ensemble ranking of 16 fixed candidates and
at most four geometry-only checks. Each method executes one shot per position;
all setup rejects stay in the denominator. Extra model/geometry planning makes
this a system comparison, not an equal-inference-budget architecture ablation.

At most 264 fresh native slots: 72 validation +180 TEST +12 pre/post controls.
Native execution waits for the 4090 shared slot and no foreign GPU process,
then has a two-hour acquired limit. The queue expires after 24 hours. Existing
Isaac work, the 4070 and viewer process are not interrupted. The results panel
updates in the background and requires a separate complete native audit.

Only64 substeps, current ball/scorer, privileged geometry and held endpoints.
Earlier128 failures, measured/provisional geometry and old results remain
unchanged. No all-table, perception, video-WAM, grasp/transit, noise or physical
validation claim is made.

## Provenance

- Frozen producer commit: `fa554312b25827315c6eefe56c19c0de42606b0c`.
- Frozen source archive: `22a218528ddb576167385a2f5365db9251e48689629aa22f9920ef750623d898`.
- Study contract: `90a7d5b7e3d1fb1e738a36a7cf30930116516da0d6d6435c06b98eb97be8c376`.
- Exact checkpoint: `0f8d00e59d7b9c07056a74b9f05f8a850ec46603b67a6f96489c90e31ebd40b4`.
- [Sanitized training summary](training-summary.json) binds the source dataset,
  checker and external training-only proof. [SHA256SUMS](SHA256SUMS) binds this
  publication. Full raw source evidence remains outside Git.
