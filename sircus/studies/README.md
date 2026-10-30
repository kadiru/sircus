# Front-bucket learning study

This module adds a separate, provisional negative-Y scene profile. Historical
`two-arm/scene.json`, the positive-Y studies, and their fingerprints stay unchanged.
The profile is a numerical approximation, not a calibrated deployment package.

Coordinates: dropper base at origin, +X toward arm 14, +Z up. The operator is near
Y=-900 mm; bucket locations are on the negative-Y side. The nominal board center
is [600,-150,350] mm, with RPY [0,-30,90] degrees. Bucket center is [600,-400] mm.
The provisional tabletop is centered at [668.655,-300,-5], size [1537.31,1000,40].
User dimensions for the board (200×300×5 mm) and tapered bucket are preserved.
The reported bucket height 175 mm remains unresolved as overall height versus
interior depth. The current model assumes interior depth 175 mm and floor
thickness 8 mm; those assumptions are not measurements.

The screenshot loading TCP [150,-600,200] and dropping TCP [600,-150,600] mm
are not ball-center measurements. The numerical pickup uses the loading TCP as
an explicitly uncalibrated ball-at-TCP proxy. Arm spacing 1337.31 mm is measured,
but interpreting it as a base-to-base rigid transform and the tool mounts remain
provisional.

## Run without services or hardware

The verified environment is Python 3.12.14, NumPy 2.5.4, and SciPy 1.18.1. Use the pinned dependencies in
`sircus/simulation/requirements-viam.txt`; tests require NumPy and SciPy. Run from
the repository root. Set BLAS thread counts before Python imports.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python -m unittest discover -s sircus/studies/tests -v
python sircus/studies/run_front_study.py \
  --output .runtime/overnight-study/front-initial \
  --seed 20261030 --train-scenes 24 --validation-scenes 6 --test-scenes 8 \
  --teacher-budget 12 --oracle-budget 4 --workers 2
python sircus/studies/run_front_study.py \
  --output .runtime/overnight-study/front-curves \
  --curve --seeds 20261030 20261031 20261032 --sizes 8 24 64 128 \
  --validation-scenes 16 --test-scenes 48 \
  --teacher-budget 16 --oracle-budget 16 --challenge-layouts 24 --workers 2
```

Output directories must be new. These commands import pure numerical modules;
they do not connect to Viam, issue robot commands, or start a web server. Raw
output belongs in ignored `.runtime`, not Git. Retain stdout as a run log.

## Training and frozen inference

A layout contains board XYZ and bucket displacement from the board. Core training
bounds are board X570–630, Y-180–-120, Z330–375 mm, relative bucket X ±120 mm,
relative bucket Y -360–-140 mm. Explicit learned actions are board roll
[-35,35]°, pitch[-55,-8]°, board-local U[-70,70], V[-100,100] mm, and release gap
[60,280] mm. Board yaw is fixed at 90°. Gap is the world-Z clearance defined by
the shared shot contract, not an uncalibrated physical TCP height above the board.

The training teacher searches only training layouts using a predeclared limited
budget. Its proposals include a fixed action, geometric search seeds, and seeded
perturbations; the best valid proposal becomes a supervised label. A scene with
no valid proposal remains a recorded teacher failure and contributes no fabricated
label. Models are linear/quadratic ridge regressions fitted to training labels
only. Feature standardization uses only training data. Three predeclared
model/regularization candidates are compared on validation capture count, then
validation shaping reward. Test data never selects model parameters or checkpoints.

Frozen inference is one feature expansion and matrix multiplication. It does
not invoke the teacher, optimizer, IK solver, or physics. Evaluation checks IK and
rolls out that one prediction. Full approach/release/retreat endpoint infeasibility
counts as failure. This is stricter than release-only IK, but still does not
validate collision-free trajectories or full robot dynamics.

Each run uses the shared versioned one-second capture evaluator: a board bounce,
whole-ball entry, and one continuous second inside are required; rim-assisted
captures count and partial exits reset retention. Checkpoints guard exact asset,
contact/scoring source, dependency, action and coordinate contracts. Editing
`profiles.py`, `front_study.py`, `run_front_study.py`, or inherited fingerprinted
sources invalidates previous checkpoints. Freeze source before long computation.

## Learning curves and challenges

The manifest is saved before training, evaluation, or challenge rollout. Three
training seeds use nested N8/24/64/128 prefixes. All seeds and sizes share the same
16 validation and 48 test layouts and the same release-noise seeds. Splits are
disjoint by layout fingerprint. Only training labels enter model fitting.
The default checkpoint is prespecified N128/seed20261030, never best test.

The fixed baseline and a separately labeled 16-proposal held-out search oracle
are evaluated on the same test layouts. The search oracle is a limited-budget
comparison, not an optimality bound and not frozen learned inference.

Challenges use 24 independent layouts per distance and lateral-offset group.
Distance spans 60–450 mm toward negative Y; lateral offsets span ±200 mm.
Some are explicitly outside the training distribution. Release-noise challenges
use the same 24 layouts and three replicates each, with paired random seeds at
XY/gap amplitudes 0/0, 2/3, 5/8, 10/15, 20/30, and 30/45 mm. Repeated noise trials
are clustered by layout. Per-model core intervals are Wilson intervals over
independent layouts. Challenge and seed-aggregate intervals envelope a layout
bootstrap with a Wilson effective-layout bound; they do not count noise repeats
or training seeds as additional independent layouts. This conservative envelope
avoids certainty claims from finite all-success/all-failure samples. Seed-level
rates/range are reported separately. Paired gains use a paired layout bootstrap.

Artifacts include the complete manifest, candidate teacher rows, training labels,
validation candidates, immutable checkpoints, raw test/challenge rows, summaries,
and actual numerical replay frames with endpoint arm boxes. `report.json` is the
compact dashboard contract; `/study-results.html` displays it when configured by
the dashboard. The report marks all results synthetic. Historical positive-Y
24/6/8 results are a different profile, not a controlled comparison against the
new larger-study splits.

## Limits

No result establishes sim-to-real transfer. Board stiffness/damping, ball mass,
friction/restitution, tool transforms, board pose, table geometry, and bucket
placement require physical identification. Bucket vision calibration, gripper
ball offset, full native collision-path replay and hardware validation remain
separate gates. A successful reduced-order rollout is not permission to actuate
hardware. The v2 native replay guard remains in place.

## Optional declared center/edge interventions

After a completed core curve study, run:

```sh
python sircus/studies/contact_interventions.py \
  --study-dir .runtime/overnight-study/front-curves --workers 2
```

This is a separate experiment, not a better policy score: on the same held-out
layouts and release seeds, it overrides U/V with center (0,0), near-edge
(±65,0)/(0,±95), and corner (±65,±95) mm. Each checkpoint's predicted board
roll/pitch and gap remain unchanged. Requested U/V and simulated actual impact
U/V are recorded separately because compliance and release noise can shift the
contact. Endpoint failures remain failures. The extension records its own source
hash, retains `report-core.json`, and appends explicitly labeled intervention rows
and replays to `report.json`. Re-running over an already extended report is rejected.
Paired-difference percentile intervals are descriptive and can collapse when all
sampled differences are identical; absolute capture intervals retain the
finite-sample layout bounds described above.

Distance groups independently sample small board-position variations, so these
are descriptive sampled challenges rather than a paired causal distance
experiment; no strict monotonic distance effect is claimed. Likewise, forcing
center/edge U/V changes both trajectory and compliant response. Capture-rate
differences cannot be attributed solely to board suspension.

Raw stdout challenge progress uses a generic per-attempt summary. Its
repeated-noise intervals are **not inference-quality** because the 72 attempts
contain only 24 independent layouts. Use the clustered intervals in the saved
`challenge-summary.json` and final `report.json` for conclusions. Original logs
remain preserved; they are not silently rewritten.

## Post-study diagnostic controls

After the contact extension, a separate bounded diagnostic can compare finite
search and the predeclared fixed action on the existing distance220/340/450 mm
and lateral-offset ±200 mm groups (24 layouts each):

```sh
python sircus/studies/diagnostic_controls.py \
  --study-dir .runtime/overnight-study/front-curves --workers 2
```

This is explicitly a **post-study control**, not another trained model. It uses
16 proposals per layout (1,920 total) and120 fixed-action attempts, with an
explicitly allowed out-of-distribution diagnostic evaluator. The original
teacher and frozen-policy guards remain unchanged. No checkpoint is refitted
or selected. Zero finite-search successes would not establish impossibility.
The extension freezes its own manifest, saves raw proposals outside Git, and
preserves `report-pre-diagnostic.json` before adding labeled diagnostic rows.

The predeclared noise cohort differs from the core test cohort: its 24 unique
layouts have mean forward distance 207.1 mm, with 2/24 beyond 280 mm; the shared
48-layout test has mean 241.2 mm, with 16/48 beyond 280 mm. High noise-cohort
capture rates are not overall held-out performance. The original random
manifest stays unchanged; no favorable or unfavorable resampling was performed.

## Offline study video and poster

The reusable renderer consumes the final saved report and its numerical frame
traces. It does not import training code or modify checkpoints. Install its media
dependencies in a separate environment so model dependency fingerprints remain
unchanged:

```sh
python3 -m venv .runtime/media-verification-venv
.runtime/media-verification-venv/bin/python -m pip install \
  Pillow==12.0.0 imageio-ffmpeg==0.6.0
.runtime/media-verification-venv/bin/python sircus/studies/render_video.py \
  --report baselines/front-bucket-20261030/report.json \
  --output .runtime/verification/overnight/sircus-learning-study.mp4
```

`imageio-ffmpeg` supplies the encoder; no system ffmpeg installation or browser
recording is required. The script writes a 1280×720 H.264 MP4 at 15 fps, poster PNG,
per-clip sample PNGs, encoder log, and a JSON provenance record with the report
checksum, selected replay/checkpoint IDs, actual outcomes and scope. Keep these
generated files outside source control unless explicitly publishing a reviewed
media artifact.

The current report produces a 78-second silent film: actual learning curves with
95% intervals, the same predeclared held-out layout at N 8/24/64/128, a held-out
capture, near/far failures, and center/edge contact interventions. Saved frames
are selected by timestamp; no robot motion is interpolated. Arm boxes show static
endpoint geometry. Endpoint-infeasible cases explicitly have no executed ball
trajectory. Out-of-range layouts and forced interventions remain labeled, and the
four paired-layout failures are preserved without inventing visual improvement.

Verification of the delivered 2026-10-30 film: encoded duration 78.00 s,
1280×720, 15 fps; complete ffmpeg decode succeeded with no errors. The poster,
paired-layout sample and an extracted encoded frame were visually inspected.
Native robot paths, real-world success, calibrated tool transforms and identified
contact dynamics remain outside this evidence.
