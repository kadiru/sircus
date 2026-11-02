# Interactive camera scout

The Mac camera scout is a static 3D view built from the original UF850 visual meshes and measured transforms exported from a frozen native Isaac scene. It contains 35 tables, 70 arms, their tools/boards, balls and tapered buckets. It changes only the browser camera; final appearance and motion must be rendered in Isaac. Browser lighting and the gray board display material are approximations of the selected silver look.

Open `http://127.0.0.1:18784/` on the configured Mac while the local server is running.

- Choose **Orbit** and drag to rotate around the aim point.
- Choose **Pan** and drag to translate the camera and aim together. Right-drag or Shift-drag also pans.
- Scroll to zoom. Use **Fit farm**, **Focus front table**, presets, or numeric eye/aim coordinates.
- **Reset view 06** restores the selected camera at Z=2 m.
- **Save camera JSON** downloads full-precision world eye/aim, Z-up, the scene fingerprint, 1920 × 1080 resolution and 23/36/20.25 mm intrinsics. Browser autosave restores the last camera after refresh.

The renderer stays at 16:9 even in fullscreen. The exported camera provides the pose for a subsequent native Isaac render; this tool does not launch that render automatically. Source workcell `r1c2` remains at display slot `r1c7` using the documented whole-cell permutation.

## Build from a native snapshot

In the pinned 4090 container, use a fresh output directory and the existing camera preview command, adding:

```bash
--camera-previews --camera-option 6 --camera-right-m 0.30 --camera-z-m 2.0 --scout-snapshot
```

This finite capture writes `scout-state.json`: actual articulation body positions/quaternions and table/bucket/ball transforms. The manifest hashes it. No learned or physical motion is performed. Copy this snapshot, its `camera-options.json`, the generated `dropper.urdf`/`board_arm.urdf`, and the seven original UF850 visual STLs into an ignored local input directory.

With NumPy/SciPy available, build the browser data directory:

```bash
python3 camera_scout_export.py --snapshot INPUT/scout-state.json \
  --manifest INPUT/camera-options.json --assets INPUT --meshes INPUT/meshes \
  --output NEW_VIEWER_DIRECTORY
python3 camera_scout_server.py --data NEW_VIEWER_DIRECTORY --port 18784
```

The exporter uses measured native link transforms and URDF visual offsets. It expands only fixed merged links and rejects missing moving-body transforms. Browser geometry shares/instances the original STLs. Third-party model/license files and generated scene data remain outside Git; preserve the UFACTORY license beside local meshes.

The server binds only `127.0.0.1`, serves a fixed asset allowlist, and has no write, simulator, training or hardware command routes. It is not installed as an automatic service. Closing/restarting this viewer does not redeploy a running simulation or robot dashboard.

## Current Mac instance

Task checkout: `$HOME/Documents/ChatGPT/hackathon/.runtime/worktrees/sircus-camera-scout`.
Local data: `$HOME/Documents/ChatGPT/hackathon/.runtime/media/camera-scout-20261102/viewer`.
The finite native export finished; the loopback server remains running for the user. The 4070 workload and existing services were preserved. [Verification](../../baselines/isaac-camera-scout-20261102/README.md).
