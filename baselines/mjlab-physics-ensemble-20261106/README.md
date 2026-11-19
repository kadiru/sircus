# Learned physics ensemble — completed native comparison

The five-member state-based forward model is trained and saved here. All 264
prespecified fixed64 requests completed; the full independent audit passed
**897,304 checks**. On the same 60 fresh TEST bucket positions, BC captured 23/60,
PPO 26/60 and ensemble-ranked selection 24/60. The ensemble reduced setup
rejections but did not demonstrate a capture improvement over the saved policies.
The [read-only UI](http://127.0.0.1:18781/#physicsEnsembleStudy) keeps model
predictions separate from measured native outcomes.

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

## Completed results

[Completed summary](results-summary.json) and [all 180 TEST settings](test-settings.csv)
retain every requested outcome, including setup rejects. Each method used the
same 60 fresh TEST layouts and one nominal shot per layout:

| Method | Captures | Clean captures | Native releases | Setup rejects | Flight misses |
|---|---:|---:|---:|---:|---:|
| Saved BC | 23/60 (38.3%) | 3 | 44 | 16 | 21 |
| Saved PPO | 26/60 (43.3%) | 5 | 45 | 15 | 19 |
| Ensemble-ranked | 24/60 (40.0%) | 3 | 49 | 11 | 25 |

All 264 slots completed: 72 diagnostic validation, 180 TEST and 12 pre/post
controls. They produced 204 native releases, 60 setup rejects and zero
physics-invalid outcomes. Positive and native-miss controls passed before and
after. The 113 aggregate captures include validation and controls; they are not
a TEST success rate. Acquired native execution took 1,028.1 seconds (17.1
minutes), excluding CPU preparation, queueing and the later audit. Existing
Isaac work, the 4070 and viewer process were not interrupted.

The [frozen protocol](../../sircus/mjlab/PHYSICS_ENSEMBLE.md) used 24 diagnostic
validation and 60 TEST bucket positions at least 10 mm from 169 declared prior
benchmark/diagnosis coordinates and each other. All 252 predictions were frozen
before any evaluation outcome; the model had no validation tuning or selection.
Ensemble ranking used 16 fixed candidates and at most four geometry-only checks,
while the saved actors each proposed one action. Different historical exposure
and extra planning make this a system comparison, not an equal-inference-budget
architecture ablation or a cold-start sample-efficiency claim.

[Descriptive statistics](descriptive-statistics.json) retain Wilson 95% capture
intervals and paired-layout bootstrap differences from these same audited
outcomes. All paired intervals include zero. They are post-hoc descriptions of
one nominal cohort, not independent-seed or noise reliability.

## What remains to improve

More admitted shots did not ensure accurate flights: all 25 ensemble native
misses lacked whole-ball entry. Across 138 native TEST flights from all three
methods, first-flight opening-distance prediction MAE was 65.2 mm. The model was
overconfident: across all 180 TEST requests, 100 capture scores were at least
0.9, but only 53 captured. These scores and model disagreement are uncalibrated.
No convincing method winner is established by one nominal cohort.

Next, inspect bounded high-score misses against measured rebound/aperture traces
before more training. If these 60 TEST positions inform selector changes, they
become development data; future changed models need new frozen held-out layouts.

Only 64 substeps, current ball/scorer, privileged geometry and held endpoints were
tested here. Earlier 128 failures, measured/provisional geometry and old results
remain unchanged. No all-table, perception, video-WAM, grasp/transit, noise or
physical validation claim is made. The legacy interactive solver is not switched
to this model.

## Earlier partial-proof snapshots

The [training summary](training-summary.json) preserves the 126,235-check
training-only proof. The [planning summary](planning-summary.json) preserves the
683,714-check CPU verification of all frozen forecasts, candidate rankings and
ensemble geometry screens. At those publication checkpoints, native outcomes
were unverified; their false native-verification flags remain historical facts.
The completed result binds a separate full native audit rather than relabeling
either partial proof.

## Provenance

- Display/export implementation: `95f746ee46b953c4d6bfb50c55d29d9519ca2142`.
- Frozen producer commit: `fa554312b25827315c6eefe56c19c0de42606b0c`.
- Frozen source archive: `22a218528ddb576167385a2f5365db9251e48689629aa22f9920ef750623d898`.
- Study contract: `90a7d5b7e3d1fb1e738a36a7cf30930116516da0d6d6435c06b98eb97be8c376`.
- Exact checkpoint: `0f8d00e59d7b9c07056a74b9f05f8a850ec46603b67a6f96489c90e31ebd40b4`.
- Full independent audit: `72e5cc9fd4c424896a0c41588b3eec0bd85317ce931cd76e90bd850474f87ebb`.
- [Sanitized completed summary](results-summary.json) binds the raw report,
  manifest and full audit. Earlier training/planning summaries retain their
  source/checker/proof identities. [SHA256SUMS](SHA256SUMS) binds this publication.
  Full raw source evidence remains outside Git.
