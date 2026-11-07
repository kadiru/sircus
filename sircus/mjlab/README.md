# Sircus in mjlab

Latest: [flight-miss diagnosis](FLIGHT_RESIDUAL_DIAGNOSIS.md) of the completed
flight-consistent selector comparison: 17/30 complete captures, nine native
misses and four setup rejections. Eight misses never entered; one was retained
after a disallowed robot contact. All nine failed layouts had negative predicted
opening margins across their 16 proposals. A [paired gap-coverage ablation](GAP_COVERAGE.md)
compared unchanged versus two wider gap proposals at 50 fresh layouts, with no
model fitting or automatic policy switch. Both captured **24/50 (48%)**;
clean captures were 13 original versus 11 wider-gap. The separate v2 audit passed
**1,441,075 checks**. [Results and preserved audit history](../../baselines/mjlab-gap-coverage-20261107/README.md).
UI: `#flightResidualDiagnosis`.

Latest: [PPO versus imitation diagnosis](POLICY64_DIAGNOSIS.md), with
[audited paired examples and 96-slot intervention results](../../baselines/mjlab-policy64-diagnosis-20261106/README.md).
Five of seven lost captures were setup rejections; board-height-only restoration
recovered all five once. The original two native misses repeated at 64. No new
training or policy activation. UI: `#policy64Diagnosis`.

Completed [fixed-64 success-bootstrap/PPO study](POLICY64.md): historical TRAIN-only
successes seeded an actor, followed by 1,024 fresh PPO requests and frozen comparison
at 50 new bucket positions. [Results and checkpoints](../../baselines/mjlab-policy64-20261105/README.md):
prior 19/50 captures, nearest success 18/50, imitation 20/50, selected PPO 18/50.
PPO did not improve capture rate. All 1,284 requests passed the independent audit.
Historical 128 results and compatibility guards remain unchanged; the existing
interactive solver has not switched to the new checkpoint.

Latest diagnostic: the [bounded 650 mm impact-point study](IMPACT650.md) completed
189 requests / 146 native releases in 544.3 worker-seconds. Discovery produced five
rim captures and zero clean; all five selected recipes captured 3/3 at 64 but 0/3 at 128.
No recipe qualified. Both control gates and 54,829 independent checks passed.
[Audited geometry/results](../../baselines/mjlab-impact650-20261105/README.md) and
[read-only UI](http://127.0.0.1:18781/#impact650Study) remain separate from historical
policy benchmarks. No new PPO/noise or physical work was started.

The [matched current-ball sweep/PPO benchmark](SWEEP_RL_BENCHMARK.md) completed
all six workers. Each method/seed spent 1,024 TRAIN requests from the same
17-recipe prior; frozen formal test scores are sweep **67/216 (31.0%)** and PPO
**58/216 (26.9%)**. The strict 1 µm wall predicate is contact-sensitive, so these
results do not establish an ordinary capture or adaptation winner. The
[checksummed evidence and six selected artifacts](../../baselines/mjlab-sweep-rl-benchmark-20261103/README.md)
passed the corrected 902,252-check audit: 7,440 total requests / 3,738 native
releases. Setup + TRAIN wall averaged about 35 minutes per worker for either
method; full reservation averaged about 52 minutes and sampled VRAM 762 MiB.
The new [read-only result view](http://127.0.0.1:18789/#sweepRlBenchmark) is separate
from existing services; no policy was activated. Contact calibration and a
versioned scorer/reward/contract are required before the next comparative study.

Earlier opt-in [conditional gap-window/SAC experiment](WINDOW_STUDY.md): one fixed
development bucket, 24/6/8 disjoint height/tilt contexts and equal1,008-request
training budgets. It compares adaptive search plus a supervised capture predictor
with terminal continuous SAC, preserving rejected requests and frozen evaluation.
This does not replace the existing viewer policy or claim all-table coverage.
Completed: predictor23/184 frozen tests versus SAC0/184; only3/8 test board
contexts resolved. Two separately repeated wider-window centers captured3/3
nominal +20/20 gap-noise trials. [Audited results and exact settings](../../baselines/mjlab-window-sac-20261102/README.md).

The [corrected-contact spatial map](CORRECTED_GRID.md) completed: 35 DEV layouts,
460 native search releases, 87 nominal repeats and 580 fresh requests. Seventeen
settings at 12 positions passed fixed-home paths and the 3/3 plus ≥18/20 gate.
Both CPU audits passed. This supports preparation of a limited corrected-physics
selector pilot; existing trainers still need a matching adapter. No new PPO or
policy switch occurred. [Final evidence and exact settings](../../baselines/mjlab-corrected-grid-20261102/README.md).

The [IK-stability follow-up](../../baselines/mjlab-ik-stability-20261102/README.md)
fixes query-order dependence and preserves all 16 saved-source branches in its
audit. A larger 16-start cold budget also recovers 16/16. Another 10,240 native
throws at the ten remaining positions found direct captures at five, but none
passed independent final reliability qualification. No additional PPO was run.
[Stable solver protocol](IK_STABILITY.md).

The [targeted bank expansion](../../baselines/mjlab-expanded-bank-20261102/README.md)
adds one 9/10-confirmed template after 15,360 search throws, followed by 4,096 PPO
throws. On 100 fresh positions × 3: learned 188/300 versus uniform 182/300; the small
capture gain is uncertain, while clean captures improve 145 versus 76. Shared
original candidates retain identical cached IK branches. [Protocol](EXPANSION.md).

The [interactive learning interface](web/README.md) and [two completed learning
studies](../../baselines/mjlab-learning-20261102/README.md) extend this contact
comparison. Bucket dragging, positional locks, frozen prediction, explicit search
and native trajectory verification are available. [Method and limits](LEARNING.md).

This comparison runs the two-arm bounce task with mjlab's MuJoCo Warp backend and scores the recorded ball/contact traces using the existing Isaac-independent Python scoring code. The starting scene and selected shots come from the completed Isaac cylinder-farm study. This is a contact-model and endpoint replay comparison. The existing Isaac setup provides the reference implementation.

## Runtime

Pin `mjlab==1.6.0` in a separate Ubuntu environment. The tagged release accepts Python 3.10–3.13 and PyTorch 2.7 or newer, and declares MuJoCo and MuJoCo Warp 3.11 dependencies. It differs from the moving `main` branch. The experiment uses only the RTX 4090 selected by its GPU UUID. Runtime packages, generated models, logs and full observations belong outside Git. [Tagged dependencies](https://github.com/mujocolab/mjlab/blob/v1.6.0/pyproject.toml)

The GPU simulation is created with `mjlab.sim.Simulation` and `SimulationCfg`, using a compiled MuJoCo model. The checked run uses1920 Hz physics and240 Hz shared scoring. Reset state by writing the existing `sim.data.qpos[:]` and `qvel[:]` buffers and calling `sim.forward()`. Keep buffer identity stable because CUDA graphs retain their addresses. [Simulation source](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/sim/sim.py)

Bucket placement uses a world-child mocap body and per-world `sim.data.mocap_pos`/`mocap_quat` values. MuJoCo Warp caches the collision transforms of static geoms when data is created. Changing an expanded `model.body_pos` field can therefore move the reported body pose while leaving its static collision geometry at the template location. Check the bucket base's `geom_xpos` against its requested position and run a direct-drop contact probe before comparing shots. [Static-geometry update implementation](https://github.com/google-deepmind/mujoco_warp/blob/main/mujoco_warp/_src/smooth.py)

## Matched task

Both engines use the measured 300 × 200 × 5 mm board, the same provisional ball properties, and the same straight-sided tapered bucket. The bucket retains the reported 175 mm height and its unresolved interpretation. The current model assumes 175 mm interior depth plus an 8 mm bottom, putting the rim at 183 mm above the table. Bucket walls remain 64 separate convex wedges surrounding an open cavity.

The fresh paired comparison preserves selected bucket XY coordinates, frozen paired arm configurations, board geometry, release position, initial zero ball velocity and physics timestep. Source joint states were measured at the end of the original throws, so the original outcome labels only stratify the diagnostic sample. Fresh paired engine results use the same prescribed initialization in both engines. It reuses `ShotScore`, `EntryQuality` and `BoardRoute` from `sircus/isaac`. A successful route requires a measured board rebound, whole-ball entry and one continuous second inside. Rim-assisted captures count. A partial exit resets retention, and intervening table, floor, robot or unidentified contact disqualifies a direct route.

MuJoCo and PhysX have different contact solvers. MuJoCo's `solref`/`solimp` contact response is not numerically interchangeable with PhysX restitution and contact-offset settings. The report therefore needs the exact contact settings, contact-probe results and trajectories alongside capture counts. Compare first impact time, rebound velocity, landing/entry position and retention. Any tuned contact setting must be labeled with its calibration trials; evaluate it on separate shots.

Timing has two useful measures. Physics-only throughput excludes installation, compilation, CUDA graph warmup, rendering and Python scoring. End-to-end trial time includes reset, stepping and the shared scorer. Report both, with environment count, device, package versions and repeated measurements. GPU parallelism is a potential benefit; speed remains an experimental result for this task.

## Perception and the interactive showcase

mjlab provides RGB-D cameras and pinhole raycasting. Depth images can be back-projected into point clouds using their camera intrinsics and extrinsics. Segmentation can supply simulator object IDs for the planned quasi-perception stage, while the actual perception pipeline learns to estimate the ball, board and bucket from sensor data. Ground-truth measurements remain available for evaluation. [Camera documentation](https://mujocolab.github.io/mjlab/v1.6.0/source/sensors/rgbd_camera.html)

A future interactive task should accept bucket XY and optional constraints on either arm. The state contains the bucket pose, arm joint states, board geometry and any fixed constraints. An action proposes a board center/orientation and release position/height. A feasibility stage masks unreachable or conflicting actions and passes valid ones through IK and measured settling before a simulated throw. The result should show the predicted configuration, observed trajectory and score. If no feasible configuration is found, show that outcome and allow the user to relax a constraint.

Start by varying bucket positions with both arms free. Then introduce constraints during training, including fixing board center, restricting board height or tilt, or fixing the dropper's XY/height. Supply the constraint mask and values to the policy so it can condition its answer. Use disjoint training, validation and test layouts and frozen-policy evaluation. Finite training grids establish performance at those layouts; success across most table positions needs held-out continuous-position tests and a coverage map.

The current goal is entirely simulated. ExPloy's export pipeline is optional if portable inference artifacts become useful. The earlier UF850/Viam adapter comment concerned executing policies on physical robots; that adapter is outside this showcase's requirements.

## Reproduce the checked comparison

Run from the repository root on the Ubuntu machine. The native assets in `$HOME/sircus-isaac/assets` and the referenced `vendor/xarm_ros2` meshes are required. Install into an isolated Python 3.10 environment using `sircus/mjlab/requirements.txt`. GPU index 0 was verified as the RTX 4090 on this host. The runner refuses a visible set containing multiple GPUs or a device other than the 4090.

```bash
python3 -m venv "$HOME/sircus-mjlab/venv"
"$HOME/sircus-mjlab/venv/bin/python" -m pip install -r sircus/mjlab/requirements.txt
CUDA_VISIBLE_DEVICES=0 "$HOME/sircus-mjlab/venv/bin/python" -m sircus.mjlab.run \
  --manifest baselines/mjlab-comparison-20261102/benchmark.json \
  --assets "$HOME/sircus-isaac/assets" \
  --output "$HOME/sircus-mjlab/output/a-new-run" \
  --substeps 8 --solref .004 .15
```

The checked run uses 1920 Hz physics with eight substeps per 240 Hz scoring interval. Forces are aggregated by category over the eight substeps. Both engines apply the same recorded joint positions and zero joint velocities before every inner step. Rendering is excluded from timing. Full scoring/state reads are included; physics-kernel-only speed was not measured.

`calibrate.py --fine` reproduces the independent normal-drop calibration scan. Its fixed 5 mm board, train drops at 100/300/600 mm and held-out drops at 200/400 mm do not use bucket success outcomes. The selected `solref=(.004,.15)` meets the 2 mm penetration limit at 1920 Hz. Restitution still varies materially with impact speed/phase. [Calibration and checked results](../../baselines/mjlab-comparison-20261102/README.md)

The fresh Isaac counterpart is `isaac_pair.py --headless --device cuda:0 --substeps 8 --benchmark BENCHMARK --output NEW_RESULT --assets /workspace/assets`. Copy it and `contract.py` beside the existing pure Isaac source in a new snapshot, then launch with the existing `python.sh` inside the isolated 4090 container. The checked counterpart is `sircus-isaac/output/mjlab-paired-fine-v1.json`; its source snapshot is `studies/mjlab-paired-v1`. Original source, runtime configuration and study outputs retain their paths.

```bash
"$HOME/sircus-mjlab/venv/bin/python" -m sircus.mjlab.compare \
  --manifest baselines/mjlab-comparison-20261102/benchmark.json \
  --first ISAAC_RESULT --second MJLAB_RESULT \
  --output comparison.json --markdown comparison.md
```

`render_replay.py` creates a common native MuJoCo rendering of both engines' recorded ball trajectories at the fixed arm configurations. The video is a physics-trajectory replay; it does not show an Isaac application viewport.
