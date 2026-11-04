# Sircus Isaac foundation

A separate Isaac Lab / PhysX backend for two articulated UFACTORY UF850 arms,
a wrist-attached board, an open tapered bucket, a ball, and an ideal RGB-D camera.
This checkpoint builds and validates a scene. It does **not** train an RL policy,
validate grasping, or inherit numerical scores from the earlier simulator.

The separate opt-in [fixed-scene RL task](RL_FIXED_SCENE.md) now trains shot
parameters on this scene. `run.sh start` remains the stationary foundation;
`run.sh train` explicitly replaces it with a bounded training run.

## Parallel workcells

The separate [32-table world](PARALLEL_WORLD.md) uses64 articulated arms in one
isolated GPU physics scene, with distinct fixed bucket coordinates and batched
height/gap/tilt experiments. Its [first verified results](../../baselines/isaac-parallel-world-20261101/README.md)
and video appear in `/parallel-world.html`. It does not switch the stationary
foundation or existing PPO task automatically.

The separate [one-minute native farm film](PRESENTATION_FILM.md) records fresh
35-table trials from the saved fixed perspective camera, using qualified
geometry variation, driven arm setup and complete outcome evidence. It makes no
new optimizer updates and preserves the original training/scoring contracts.

## Pinned baseline

- Isaac Lab tag `v2.3.2`, commit `37ddf626871758333d6ed89cf64ad702aef127d0`.
- Mounted Isaac Sim binary: `5.1.0-rc.19+release.26219.9c81211b.gl`.
- Python 3.11.13, bundled PyTorch 2.7.0+cu128.
- Manufacturer `xArm-Developer/xarm_ros2` commit `ac736c92b02de7ca7e0be6f737d77d46707464d8`, sparse `xarm_description`.
- The reused container **image tag** is `nvcr.io/nvidia/isaac-sim:6.0.1`; its `/isaac-sim` is replaced by the existing **5.1 binary read-only mount**. Do not infer runtime version from the image tag.
- `constraints.txt` records added venv package versions. It does not replace the Isaac binary's bundled dependencies.

The official [Lab 2.3.2 binary installation guide](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/binaries_installation.html) supports this generation. This deliberately avoids a second full simulator download and changing the existing other-GPU workload. Migrating to Lab 3.x requires a separately pinned compatible Sim release and API/quaternion migration.

## Ubuntu host layout and launch

Default workspace: `~/sircus-isaac`, container `sircus_isaac_4090`.
No Viam credentials, hardware endpoints, shared cloud part, or robot services are used.

For a **new** workspace only, set the dedicated GPU UUID from `nvidia-smi` and run:

```bash
export SIRCUS_GPU_UUID=GPU_REPLACE_WITH_DEDICATED_DEVICE_UUID
bash sircus/isaac/bootstrap.sh
bash sircus/isaac/run.sh build-assets
bash sircus/isaac/run.sh smoke
bash sircus/isaac/run.sh start
```

On the configured host, use the existing workspace's source scripts:

```bash
cd ~/sircus-isaac/source
bash sircus/isaac/run.sh status
bash sircus/isaac/run.sh start   # refuses a duplicate scene process
bash sircus/isaac/run.sh stop    # only the Sircus scene, not other containers
```

`smoke` requires the live Sircus scene to be stopped first. It renders 180 frames,
steps 6 seconds of simulated time, saves one observation, tests a ball falling
through the bucket opening to its floor, and commands a small wrist-joint change
on both **simulated** arms. These tests are not full-shot success measurements.

For a read-only preview, in another host terminal:

```bash
python3 ~/sircus-isaac/source/sircus/isaac/preview.py --output ~/sircus-isaac/output
```

The configured host already has that preview process running. Do not launch a
second copy on the same port. Its host PID is recorded in `~/sircus-isaac/preview.pid`.

Open `http://127.0.0.1:18860` on Ubuntu. From a separate computer, SSH-forward that
loopback port. The preview serves only latest RGB, depth preview, and status;
it provides no physical or simulation command endpoint. Stale data is marked.

From the Mac, recreate the preview tunnel if needed (while port18780 is unused):

```bash
ssh -N -L 127.0.0.1:18780:127.0.0.1:18860 kadir@kfuHomeUbuntu.local
```

Inspect a saved capture without starting another simulator:

```bash
docker exec -u 0 sircus_isaac_4090 /workspace/source/sircus/isaac/python.sh \
  /workspace/source/sircus/isaac/inspect_capture.py /workspace/output/UTC_CAPTURE_DIRECTORY
```

See [STATUS.md](STATUS.md) for the actual validation results and bundled optional
dependency warnings. Core runtime tests passed; `pip check` has documented caveats.

The container is restricted to the explicitly selected GPU, 12 CPU cores' worth
of quota, 28 GiB RAM, and 4 GiB shared memory. No host networking, auto-start on
boot, existing cache writes, or changes to other containers are configured.
Latest preview files are replaced in place. Timestamped captures are opt-in.
The scene stops if its output filesystem falls below 3 GiB free; it deletes nothing.

## Asset and sensor contract

All new scene units are SI. Historical `two-arm/scene.json` is unchanged.

- Arms: manufacturer URDF geometry, link inertias and joints. Base spacing
  1.33731 m (52.65 in measured; base reference interpretation assumed); arm 14
  faces arm 10 with a 180° base yaw. Starting poses use offline endpoint IK.
- Board: measured 200 × 300 × 5 mm, 90° landscape mounting. A fixed joint attaches
  it to arm 14; 100 mm tool offset and 0.4 kg mass are provisional. This first
  backend uses a **rigid mount**; suction compliance is a later dynamics task.
- Gripper: two controllable prismatic geometric finger proxies, explicitly not
  a calibrated commercial gripper model. Ball pickup is not implemented yet.
- Bucket: inner top/bottom 177/150 mm; outer top/bottom 200/158 mm. A 64-segment
  straight taper with convex wall wedges and a separate bottom preserves the
  opening. Inner wall surfaces handle ball contact; outer wall surfaces handle
  robot collision. No convex hull caps the bucket. Handles are omitted.
- Reported bucket height 175 mm remains **unresolved overall vs interior**.
  Interior 175 mm + bottom 8 mm are provisional modeling assumptions.
- Bucket lies on the operator / negative-Y side, not behind the board.
  Placement and table size are provisional.
- Ball: 75 mm diameter, 0.2 kg, friction/restitution are provisional.
- Camera: fixed overview, 640 × 480 RGB and axial depth in metres. Point clouds
  are unprojected from rendered depth and intrinsics; they are not full mesh
  ground truth or physical sensor data. Ideal depth currently has no added noise.
- Camera optical frame: +X right, +Y down, +Z forward. Metadata includes intrinsics,
  simulator world transform (quaternion WXYZ), joints, body poses and scene config.
  Same-step simulator synchronization is known; no physical calibration is implied.

`output/<UTC>/` contains RGB PNG, float depth NPY, colored camera-frame PLY and
metadata JSON. Generated USD/URDF, observations, caches, logs and venv remain
host-local and outside Git. Manufacturer assets are fetched from the pinned
upstream rather than redistributed; upstream `package.xml` leaves its license
as TODO, so do not assume the older Viam model's Apache notice applies to them.

## Next implementation boundary

1. Characterize timestep/contact convergence and implement compliant mount dynamics.
2. Implement a resettable Lab RL environment, joint/task-space actions, release,
   board-bounce event tracking and whole-ball + continuous-one-second scoring.
3. Validate grasp/release, arm–arm collisions and workspace constraints. Current
   self-collision is disabled for the initial imported articulations.
4. Train a privileged-state cooperative baseline on seeded disjoint scene splits.
5. Build table/ball/bucket/board perception from depth, then train/evaluate a policy
   that uses only estimated observations. Keep ground-truth labels/critic separate.

No historical policy can be replayed into this physics contract without an explicit
adapter and new validation. Existing results and scene fingerprints remain intact.

## RGB-D perception

See [PERCEPTION.md](PERCEPTION.md) for the independent sensor-only estimator,
preview overlays, separate truth evaluator, tests and current limitations.

## Randomized perception study

[Validation contract and reproduction](PERCEPTION_VALIDATION.md) describe the
frozen detector, paired lighting/depth-noise sweep, separate evaluation truth and
results panel. This is static perception validation, not policy training.

## IK and distance diagnostic

[Method and reproduction](IK_DISTANCE_AUDIT.md) and [completed results](../../baselines/isaac-ik-distance-20261031/README.md) cover the offline IK audit and bounded215-shot PhysX sweep. The read-only preview shows these at `#ik-distance`. This does not upgrade the training/physical controller or start another training run.

## Spatial bucket grid

[Grid protocol and RL compute examples](BUCKET_GRID.md) describe the nine-position
development search, instrumented contact routes, fresh perturbed confirmations
and next policy-training boundary. New results appear in `#bucket-grid`; prior
scoring and numerical studies remain separate.
