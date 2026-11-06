# Offline learned physics ensemble — fixed64 pilot

The [trained checkpoint and independent training-only checks](../../baselines/mjlab-physics-ensemble-20261106/README.md)
are saved separately. Fresh native comparison remains pending; training loss
does not establish a held-out capture rate.

This is a small state-based forward model, not a video world-action model or a
new PPO run. It learns from the authenticated **1,024 current policy64 TRAIN
requests**: 887 native releases, 615 captures, and 137 setup rejections. Known
invalid native dynamics remain counted as exposures but have all physical
targets masked. The older 6,144 bootstrap labels and all previous held-out,
control and post-hoc diagnostic labels are excluded from this fit.

The candidate bank still uses saved BC/PPO proposals, and the source shots were
collected by BC-warm-started PPO. This system therefore inherits historical
expertise indirectly; 1,024 is its direct fit exposure, not the total cost of
creating its priors or a cold-start sample-efficiency claim.

Five independently initialized 7→64→64→9 SiLU networks each receive a fixed
bootstrap sample, 600 full-batch Adam updates at 0.001, with no early stopping or
validation tuning. Inputs are board XY, board height, tilt, drop gap and bucket
XY, normalized by fixed constants. Separate outputs model native execution,
board bounce, whole entry, capture, retention, first-rebound opening distance
and rebound velocity. Missing physical measurements are masked, not zero labels.

Fitting these small networks uses two CPU threads. Native shots use **only the
4090**, after acquiring the existing shared lock and observing no foreign GPU
process. CPU preparation does not allocate CUDA. An active Isaac or other 4090
workload is never stopped, signalled or reconfigured; the queue expires after
24 hours. The 4070 stays untouched. Acquired native execution is capped at two
hours and an eight-GiB free-storage reserve.

## Fresh evaluation

Freeze 24 validation and 60 TEST bucket coordinates in X100–1240 mm,
Y−650–250 mm using seeds 20261603/20261604. Each is at least 10 mm from all
declared ancestral benchmark/exclusion coordinates, all policy64 splits
(including the now-diagnostic 50 TEST positions), and every new coordinate.
These are fresh positions, not proof of coverage over the entire table.

Reuse the exact saved BC and validation-selected PPO checkpoints. The new
ensemble policy ranks a literal 16-candidate bank: BC, PPO, zero residual,
nearest current TRAIN capture, eight fixed axis changes and four seeded joint
changes around BC. The mean modeled requested-capture score is penalized by
0.25×ensemble spread and 0.05×mean first-flight opening distance clipped at 2 m.
Those scores and uncertainty are **uncalibrated model predictions**.

The ensemble checks at most its top four ranked candidates with the unchanged
geometry-only admission procedure, selecting the first admitted. If none is
admitted, its first ranked request remains a rejected slot. There are no native
ball-flight queries during inference, nor replacement after seeing an outcome.
All candidates, rankings, proofs and **all 252 evaluation requests are frozen
before any evaluation outcome**. All methods use the same geometric anchor at
each bucket coordinate.

The fixed model has no validation-selected hyperparameter or checkpoint.
Validation is diagnostic only. Each method gets one nominal shot per layout;
rejections remain in its denominator. The new method has extra model/geometry
inference and different historical exposure, so this is a pipeline comparison,
not an equal-compute architecture ablation.

Maximum **264 fresh requested slots**: 72 validation +180 TEST +12 positive/native
miss controls before/after. Scoring stays board bounce, whole-ball entry and one
continuous second inside; rim-assisted capture and clean entry stay distinct.
All physics is the unchanged calibrated native64 engine. Earlier 128 failures,
materials, fingerprints and provisional bucket-height interpretation remain.

## Evidence and interface

Save TRAIN row/trace identities, masked targets, bootstrap indices, initial/final
weights, optimizer states, losses, model and baseline hashes, pre-outcome plans,
every admission proof, dense native data and a complete manifest. The independent
CPU audit must verify the full experiment before the additive
`#physicsEnsembleStudy` panel can mark results verified. Predicted success never
counts as an actual capture. Raw data, host paths and configuration stay out of
Git; publish sanitized summaries and model weights separately at completion.

The existing interactive solver and all old result panels remain separate.
No policy activation, service restart, physical deployment, point-cloud
perception, grasping, driven arm-transit learning, noise study or 128 convergence
is included.

## Reproduce or inspect

Use an isolated checkout of frozen producer `fa554312b25827315c6eefe56c19c0de42606b0c`
and the pinned mjlab Python. Input directories must contain the original
authenticated policy64 evidence/proof, immutable native source and measured/
provisional assets. A fresh output directory is mandatory. Required local paths
are placeholders below; no credentials are needed for this simulation study.

```bash
CUDA_VISIBLE_DEVICES=GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4 \
  /path/to/pinned-mjlab-python -m sircus.mjlab.physics_ensemble_pipeline \
  --parent-study /path/to/policy64-v1 \
  --parent-audit /path/to/policy64-v1-audit.json \
  --native-source /path/to/frozen-policy64-source \
  --assets /path/to/assets \
  --output /path/to/new-physics-ensemble-study \
  --gpu-lock /path/to/existing-shared-scene.lock \
  --display /path/to/existing-interface-report.json \
  --queue-timeout 86400 --wall-time-limit 7200
```

The supervisor publishes partial/final snapshots without restarting the existing
viewer, waits for the GPU and invokes the independent CPU audit after completion.
The matching static panel must be installed separately. Do not launch a duplicate
worker against the same output or replace/restart an active study to inspect it.
