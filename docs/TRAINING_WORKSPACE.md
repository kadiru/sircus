# Simulator training workspace

The operator dashboard's training workspace runs the existing numerical
`train_scene_study.py` as an isolated, bounded local child process. It uses no
Viam connection, cloud identity, camera, audio service or physical controller.
The training code, physics contracts and published reference checkpoint are
unchanged. Starting training does not deploy its result.

## Experiment

Each run freezes seeded, geometrically disjoint train/validation/test layouts.
Training-only teacher searches supply labels for actual ridge regression.
Validation chooses regularization; the saved checkpoint is then frozen before
held-out evaluation. Inference is one matrix prediction with no scene-specific
optimization. The test summary separates three methods:

- Frozen scene-conditioned policy.
- Predeclared fixed action baseline.
- Per-scene search oracle, explicitly labeled as search rather than inference.

Actions are board roll/pitch, board-local U/V and release gap. Gap is world-Z
ball clearance above nominal contact, **not gripper TCP height**. Board yaw and
bucket upright orientation remain fixed. The existing narrow scene distribution
varies board center and bucket-relative XY; contact compliance, depth, mass and
other uncalibrated assumptions are not randomized by this interface. The
200 × 300 × 5 mm landscape board and tapered bucket geometry are preserved.
The bucket's 175 mm reported height still has unresolved overall/interior meaning.

A larger sample is not automatically a broader distribution. Existing asset,
source, dependency and out-of-distribution guards remain enforced. Endpoint IK
does not establish collision-free paths. Native v2 replay remains blocked until
its matching adapter and path checks exist.

## Controls and artifacts

Use **Quick smoke test** for 4 train / 1 validation / 2 test layouts, teacher budget 4,
and oracle budget 2. The seed defaults to 20261030. The full reference-sized
preset is 24 / 6 / 8 layouts, teacher budget 12, oracle budget 4. These controls
run new training; the published historical 24/6/8 reference is never overwritten.

The manager reports trainer-emitted JSON progress, keeps at most 64 recent rows,
and does not call a completed or canceled job an active training process. **Cancel**
requests termination of only this manager's owned child process group. It does
not stop a simulator server or any external service. Concurrent starts are
rejected until cleanup finishes. Timeout, failure or cancellation preserves
partial local evidence but does not expose a selectable checkpoint.

Successful runs list checksummed JSON artifacts. **Use checkpoint** downloads
that run's completed checkpoint for the guided numerical planner; it neither
updates hardware nor promotes an incompatible policy. Stored artifacts are checked
against their completed manifest on download. Modified or symlinked artifacts,
unlisted names, unsafe run IDs and incomplete runs are rejected.

The standalone manager defaults to ignored `.runtime/training/`; the operator launcher uses `.runtime/operator-lab/training/`. Each run uses
a new random directory; historical runs remain separate. The fixed subprocess
uses the current Python interpreter and only a minimal environment, without
Viam/audio credentials. No user-supplied command, executable, output path or
shell text is accepted.

## HTTP contract

The dashboard server supplies loopback/Origin checks and an ephemeral local
control token. These routes only reach numerical code:

- `GET /api/training/status`: current phase, run ID, config, progress, summary,
  available artifacts and explicit simulation-only limits.
- `POST /api/training/start`: numeric fields listed below; omitted fields use defaults.
- `POST /api/training/cancel`: request cancellation of the current job.
- `GET /api/training/artifact?run_id=...&name=checkpoint.json`: completed,
  checksummed JSON; other names come from the returned artifact list.

| Field | Default | Allowed range |
|---|---:|---:|
| seed | 20261030 | 0–2147483647 |
| train_scenes | 4 | 4–64 |
| validation_scenes | 1 | 1–16 |
| test_scenes | 2 | 1–32 |
| teacher_budget | 4 | 1–32 |
| oracle_budget | 2 | 1–16 |

The combined rollout budget must be at most 1200:
`train_scenes × teacher_budget + 3 × validation_scenes + test_scenes × (2 + oracle_budget)`.
A job has a 30-minute wall-time limit, bounded process output, and fixed artifact
size/name limits. Enabling a second browser does not create a second worker.

## Local checks

From the repository root with its Python 3.12 environment:

```sh
PYTHONDONTWRITEBYTECODE=1 sircus/simulation/.venv/bin/python \
  -m unittest discover -s sircus/operations/tests -p 'test_training.py' -v
```

These manager tests use fake child processes. The real trainer can also be run
headlessly with the documented commands in `SCENE_CONDITIONED_STUDY.md`. Small
smoke success is numerical integration evidence only; it does not prove tomorrow's
robot will capture the ball. Camera-to-robot calibration, board-mount calibration,
real bounce measurements, matching native replay and collision-aware path checks
remain required before physical deployment.

## Implementation smoke — October 30, 2026

The manager completed a real 4/1/2 run with seed 20261030, teacher budget 4 and
oracle budget 2 in 8.74 seconds on the development Mac. All four training layouts
supplied valid labels. The frozen policy, fixed baseline and separately labeled
per-scene oracle each captured 2/2 held-out layouts. This tiny smoke establishes
that the job, training, frozen evaluation and artifact-download paths execute;
it demonstrates no advantage over the baseline or physical readiness.

The generated checkpoint ID was
`94c01a292db6d02fa14f5833afdd724e0066d9bbf8c94283359d97fb2bc6c0be`, with file SHA-256
`5920473fc98932195548b8c1d2ed749e6070dd6db400b71a48d11494eee8324c`.
The trainer verified its checkpoint and split manifest stayed unchanged during
test evaluation. The manager independently verified the checkpoint download hash.
Thirteen manager regressions passed, including cancellation during finalization and
bounded application shutdown and failed local storage. Generated attempts and logs remain local, separate
from the committed historical study.
