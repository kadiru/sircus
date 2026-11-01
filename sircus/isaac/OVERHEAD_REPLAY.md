# Orthographic replay video

A close, synthetic replay of four selected targets from the completed 32-table study. It shows eight articulated arms, the releases, board impacts and bucket captures from synchronized cameras. The arms initialize at solved and verified stationary endpoints; this does **not** demonstrate pickup, gripper actuation, arm transit, a new learned policy or hardware performance.

The delivered movie is 1920 × 1280, 30 fps, 18.6 seconds. Its first section shows four tables at 0.5× speed. Its second section follows cell-00 at 0.25× with overhead and low-front insets. White trails are simulator ball-center samples, not predictions or depth-camera estimates. One-second initial and two-second final holds are editorial pauses; timestamps remain visible. No motion interpolation or generated imagery is used.

## Reproduce

Use the existing isolated Isaac Lab 2.3.2 / Isaac Sim 5.1 environment and generated UF850 assets described in [README.md](README.md). These scripts never import Viam or connect to hardware. Do not start another job while the simulation scene lock is occupied. Select the authorized GPU through the container configuration (`cuda:0` inside the 4090-only container).

Inside that container, with the repository at `/workspace/source`:

```bash
export SIRCUS_ISAAC_SOURCE=/workspace/source/sircus/isaac
timeout 600 flock -n /workspace/scene.lock \
  "$SIRCUS_ISAAC_SOURCE/python.sh" "$SIRCUS_ISAAC_SOURCE/overhead_replay.py" \
  --source-summary /workspace/source/baselines/isaac-parallel-world-20261101/summary.json \
  --assets /workspace/assets --output /workspace/output \
  --headless --enable_cameras --device cuda:0
```

`--preview-only` verifies the endpoints and saves camera-framing stills without recording a scored shot. The full run verifies bucket/rim/table/floor/robot contact attribution and cross-cell collision isolation, then replays one frozen action in each cell. It does not search for a better setting or rewrite any study pointer/baseline.

After the run exits, on the Ubuntu host with NumPy, Pillow >= 9.0, ffmpeg and DejaVu Sans fonts:

```bash
python3 sircus/isaac/overhead_video.py /path/to/output/overhead-replay-YYYYMMDDTHHMMSSZ
```

The encoder rejects incomplete runs, failed endpoints/contact probes, modified raw evidence, unmatched cameras and uneven frame timestamps. It produces `sircus-overhead.mp4`, posters and `video.json` with timing, outcomes and the movie SHA-256. It preserves all actual outcomes; a miss would be labelled as a miss.

## Contract and provenance

- Selected cells: 00, 02, 03, 06. Distinct bucket coordinates and frozen settings are read from the verified parent report; selected cells previously passed 10/10 narrow perturbations each. A changed scene or failed source verification is rejected.
- Same physics, geometry, scoring and endpoint/settling gates as [PARALLEL_WORLD.md](PARALLEL_WORLD.md). Bucket height interpretation, board mount/compliance and ball properties remain provisional as recorded in `scene.json`.
- Rendering uses actual [OpenUSD orthographic projection](https://openusd.org/release/api/class_usd_geom_camera.html) through the supported [Omniverse camera projection](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html). Apertures are converted from metres to USD's tenths of a scene unit. Isaac's pinhole intrinsics are **not** used as orthographic calibration. Clip planes isolate each table in the close views; only camera settings change.
- Physics/contact scoring runs at 240 Hz, six matched RGB cameras capture at 120 Hz. Frame sampling yields exact 0.5× and 0.25× playback at 30 fps, with no interpolated motion. Overlay projection uses the recorded ROS camera poses and orthographic width.
- Raw frames/contact samples and the movie remain outside Git. Compact evidence: [November 1 replay](../../baselines/isaac-overhead-replay-20261101/README.md).

This movie makes a tested synthetic shot easy to inspect; four selected successes do not establish generalization or a new reliability rate. The original 32-cell results and previous movie remain unchanged.
