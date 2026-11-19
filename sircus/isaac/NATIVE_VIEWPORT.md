# Direct native Isaac viewport recording

This is a screen recording of the actual Isaac Sim 5.1 application window on the RTX 4090. One live PhysX world contains 32 tables and 64 articulated UF850 arms. Isaac menus and viewport controls remain visible. The orthographic view also retains the scene tree; the perspective view hides the side panels to give the farm more screen space. Frames come from X11 window capture; there are no composited panels, added trails, generated replacements, or frame interpolation. H.264 encoding may pad the right/bottom by at most one black pixel to obtain even dimensions.

The earlier `farm_video.py` movie used actual Isaac camera renders with presentation labels and synchronized insets. It was a composite, not a native application recording. The new recording is kept separate.

## Camera views

`--view perspective` selects a true pinhole camera looking diagonally across the farm from an elevated three-quarter angle. It uses focal length 24, horizontal/vertical apertures 36/20.25, and an eye offset of (18, −23, 20) m from the farm center. Camera metadata is stored in `ready.json` and `result.json`. The existing orthographic camera remains the default. Both options use the same task and endpoint initialization.

To reproduce the perspective recording, add `--view perspective` to the launch below and choose a new output directory. `--enable_cameras` is required and now checked before startup. The [perspective verification](../../baselines/isaac-native-viewport-20261101/perspective-verification.json) is separate from the original orthographic capture.

### Low close-up

`--view close-up --focus-cell 2 --playback-rate 0.5` frames a single workcell with a 35 mm perspective camera. Its eye is at table-local (2.05, −3.10, 1.25) m, looking at (0.67, −0.20, 0.45) m. The selected cell is explicit, and the other 31 cells still simulate. Cell 2 is the centered near target, chosen for framing rather than by recorded outcomes.

`--playback-rate` bounds simulation-time progression per wall second during the filmed rollouts. The close-up uses at most 0.5× speed without modifying the 240 Hz physics step or interpolating video frames; rendering can make it slower. The camera follows no object and adds no trails or result overlays. Arm endpoints are reset between attempts, and gripper pickup/transit remains outside the task. This is a camera demonstration, not a fresh generalization study.

See [close-up verification](../../baselines/isaac-native-viewport-20261101/closeup-verification.json) for the recorded camera, playback cap, source hashes and native movie checksum.

## What runs on screen

`native_viewport.py` reuses the existing parallel PhysX scene, IK/settling checks, contact probes, shot evaluator and PPO implementation. A fresh seed (20261104) runs four simultaneous 32-shot batches and two PPO updates. All attempted shots, including misses, remain in the footage. This short demonstration is not a new policy comparison or a replacement for the [2,048-throw pilot](FARM_PPO.md).

Actions are board-center height, drop gap and tilt. Board/release XY stays fixed and facing the bucket is prescribed. Arms initialize at solved endpoints; the clip does not show learned pickup, gripper actions, collision-free transit or full joint control. Ball trajectories and contacts are computed by PhysX. The native viewport runs as fast as rendering permits, with pacing that prevents the simulated rollout from exceeding real time; it is not a real-time performance benchmark.

The current pilot covers 32 training bucket positions on one tabletop, with 32 nearby validation and 32 nearby test positions. It does not cover arbitrary 3D positions or the whole reachable table. Selected-policy test success was 6/32 versus 9/32 for a fixed action; all policies scored 0/14 on the far subset. No general drop-bounce policy is claimed. A workspace feasibility map and revised action/reward study are the next research steps.

## Reproduce

Use the existing isolated Isaac Lab 2.3.2 / Isaac Sim 5.1 installation and generated assets described in [README](README.md). Create a separate immutable source snapshot, a new output directory, and a temporary GUI-enabled container restricted to the 4090. Share only the authorized local X11 socket and its existing Xauthority file read-only. Do not publish Xauthority, widen X-server access with `xhost +`, or restart an existing workload.

Inside that container, with `DISPLAY` and `XAUTHORITY` configured:

```bash
export SIRCUS_ISAAC_SOURCE=/workspace/studies/native-view-v2
# --enable_cameras is required for the native viewport/render configuration.
timeout 600 flock -w 15 /workspace/scene.lock \
  "$SIRCUS_ISAAC_SOURCE/python.sh" "$SIRCUS_ISAAC_SOURCE/native_viewport.py" \
  --enable_cameras --device cuda:0 \
  --output /workspace/output/NEW_NATIVE_RUN
```

Once `ready.json` appears, on the same Ubuntu graphical session, run:

```bash
# Host dependencies: Python 3, ffmpeg with x11grab/libx264, xwininfo.
# Set DISPLAY/XAUTHORITY to your existing authorized local graphical session.
python3 /path/to/record_native_viewport.py /host/path/to/NEW_NATIVE_RUN
```

The host user must be able to write that output directory. Fit the Isaac window on the display and keep it visible. The recorder requires exactly one window titled `Isaac Sim 5.1.0`; it refuses to overwrite a previous recording. It starts recording before creating the `record` marker. The simulator waits at most five minutes for that marker, then writes four batch files, `policy.pt` and `result.json`. The recorder ends on completion (maximum three minutes) and writes `recording.json`. A failed capture is diagnostic only and must not be presented as a completed run.

Inspect samples from every batch, decode the full MP4 and check its checksum. Verify 128 rows, two changed policy updates and passed contact probes. Archive raw frames/video, logs, generated USD and demonstration policy outside Git. Publish only source and compact verification metadata. Close only the temporary GUI container after capture; leave the existing preview, simulation container and other GPU workloads unchanged.

See [capture verification](../../baselines/isaac-native-viewport-20261101/verification.json) for the completed run and media hash. Geometry and provisional measurements are unchanged.
