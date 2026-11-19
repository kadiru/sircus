# Sircus in mjlab

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
