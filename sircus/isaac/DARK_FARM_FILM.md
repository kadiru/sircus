# Compact farm recording

The next recording defaults to [a fixed camera and twelve foreground drops](STATION_RECORDING.md). The appearance is awaiting selection from the gray/blue-gray previews. The one-minute tour below describes the historical film.

This replay uses the recorded PPO actions from the 35-table cylinder study. The native Isaac camera follows a 7 × 5 farm with 70 articulated UF850 arms. Each table retains its own bucket placement. Four consecutive attempts at the same front-row table use different board bearings, board heights, release gaps and tilts from the original training run.

The floor is one continuous matte black visual surface with 9 mm charcoal seams. Physical floor contact continues to use the isolated per-cell colliders. Table spacing is 2.50 m across and 2.10 m between rows, giving 700 mm and 850 mm aisles around the 1800 × 1250 mm tabletops. The camera descends to follow misses beyond the table edge.

The film contains visible joint movement. During transit, position drives use the joint effort limits from the generated UF850 URDF, with stiffness 6000 and damping 180. The original hold gains and effort limits are restored before the existing settling and release checks. Motion frames record measured joint positions. The recording skips waiting for the endpoints to settle. This controller is a simulation visualization of previously sampled actions. Collision-free paths, learned pickup and physical actuator calibration require their own validation.

## Capture and export

Use the existing pinned Isaac Sim 5.1.0 / Isaac Lab 2.3.2 environment and generated assets from [the Isaac setup](README.md). Set `SIRCUS_ISAAC_SOURCE` to an immutable snapshot of this source directory. Run the following inside its isolated 4090 container, with a fresh output directory and the original completed cylinder study available.

```bash
python.sh film_cylinder_farm.py --headless --enable_cameras --device cuda:0 \
  --dark-replay /workspace/output/cylinder-film-35-v5 \
  --recording-plan dark-tour-v1 \
  --output /workspace/output/dark-farm-new
```

On the Ubuntu host, use Python with NumPy and Pillow, and FFmpeg/ffprobe with H.264 support.

```bash
python3 dark_export.py output/dark-farm-new output/dark-farm-new-export \
  --source output/cylinder-film-35-v5
```

The exporter requires fresh rim, bucket, table, floor, robot and foreign-cell checks. It verifies camera readback, source action lineage, measured joint travel, visible board impacts, a qualified bucket capture and a real floor contact. Captures use the existing whole-ball entry and continuous one-second retention contract. Ground contact needs measured force at the modeled floor height. Failed shots keep their observed outcomes.

`dark-farm.mp4` is the 60-second camera tour. `dark-drops.mp4` removes most motion setup and stationary tails around the impacts. `station.mp4` and `station-10x.mp4` show the repeated front table using the same trimmed native frames, at normal and tenfold playback speed. `film.json` records file digests, source intervals and observed outcomes. `contact-sheet.jpg` shows the release, board contact and landing at each featured table.

If a later camera setup fails, `--resume-filming PREVIOUS_RUN` restores the last completed shot's measured joint positions at a camera cut and continues in a new output directory. The resume record preserves and hashes the original completed RGB frames and shot records. The exporter verifies these against the previous output, including its contact checks. A failed or partially captured shot is excluded from the resumed prefix.

## Training and deployment framework

Sircus currently runs Isaac Lab on Isaac Sim. This is the installed environment for the farm, PhysX contact checks and rendered depth. A separate mjlab port could reuse the task's definitions and compare the same shots using MuJoCo Warp. Its current sensor API includes RGB-D cameras and pinhole raycasting, which can supply segmented point clouds after back-projection. Asset conversion, contact parameter calibration, solver checks and paired held-out evaluation would precede any comparison of policy performance. [mjlab](https://github.com/mujocolab/mjlab), [sensor documentation](https://mujocolab.github.io/mjlab/v1.6.0/source/sensors/index.html)

ExPloy supports Isaac Lab and mjlab adapters and exports PyTorch observation processing, policies and action processing to ONNX with a C++ execution library. The current Sircus policy is a high-level shot selector. Its NumPy geometry menu, SciPy IK and Viam controller need explicit integration and numerical parity checks around an exported model. Compatibility with our pinned versions is unverified. [ExPloy](https://github.com/rai-opensource/exploy), [RAI explanation](https://rai-inst.com/resources/blog/introducing-exploy-simplifying-rl-policy-deployment-in-autonomous-robotics/)
