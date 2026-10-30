> Superseded on October 29, 2026: the active dashboard now connects to the two-arm cloud-bound `sircus-sim` runtime on port8082. Robot10 drops; robot14 holds the measured200×300×5mm board. See [current two-arm setup](../two-arm/README.md). The notes below describe the preserved earlier single-arm baseline; use `--offline` to run the old local dashboard engine.

The shared dashboard gained a separate Mac-only physical operator panel on October 29, 2026. Its hardware client/calibration/data are independent of simulation. See [physical panel status](../physical_trials/STATUS.md); physical movement remains operator-controlled and calibration-gated. The native simulation runtime was not restarted.

# Sircus simulation status — October 29, 2026

Implemented a new environment based on the imported Viam scene architecture.
The imported project remains preserved under `../../imports/viam-transfer-20261029`.
Transfer verification: 613/613 manifest files matched; all 50 original checks passed.

Yesterday's exact fake-xArm6 scene has now been separately replayed on native
Viam at 127.0.0.1:8081. Seven obstacle-planned moves and TCP readbacks passed;
the unchanged idealized bounce model hit the bin floor. See
`../yesterday-replay/STATUS.md` and `../yesterday-replay/data/native-replay.md`.
This local baseline is isolated from the new 850 simulation. A separate copy is
now connected to Viam's cloud 3D scene under `../viam-cloud`.

Working fast viewer: http://127.0.0.1:8765. Working native local Viam endpoint:
127.0.0.1:8080. These are local services, not publicly published sites.

The fast viewer includes the official six-joint UFACTORY 850 model, modeled
pickup/release, configurable floating board and container, compliant contact,
center/edge impacts, trajectory replay, saved measurements and short release
searches. Robot 10 is the simulated drop arm. Robot 14 remains a floating-board
proxy; its arm is not included yet. All board/support values and the tool offset
remain provisional pending the user's measurements.

Validation completed:

- 15 new physics checks: timestep/energy sanity, center/edge dynamics, yaw,
  support offset, retained capture, misses, seeded noise and invalid overlap.
- Native Viam server 1.14.3: authentic 850 kinematics reported by the fake arm;
  seven obstacle-planned moves completed with TCP readbacks. One full
  pick/drop/retreat/reset trial completed and the synthetic ball was retained.
  Evidence: `data/viam-native-validation.json`, `data/viam-runs/`.
- Adapter: 102 orientation round-trips and four fake-only configuration rejection
  cases checked; streamed scene geometry and physics frames verified.
- Browser: drop, edge case, shifted container, 24-proposal learning, replay and
  responsive layout checked. No application console errors observed.
- Fresh synthetic release-noise tests are stored separately from training in
  `runs/`. Passing rates describe the provisional model only.

The native server and fast viewer use separate scene configuration files. The
viewer path is kinematic interpolation; native Viam performs its own obstacle
planning. No physical robot motion, camera perception, real grasp validation or
physical calibration has occurred.

The normal Homebrew installation encountered a dependency/alias lock conflict
at OpenSSL. It installed several dependencies before stopping. The native server
now runs from the official downloaded bottle in `.runtime/vendor`, using its
vendored x264 library and installed nlopt library; no system launch service was
enabled. `start-viam-server.sh` starts this local runtime. Use `run-local.sh` for
the fast viewer. The original transfer did not contain credentials or a server.

User completed work-account sign-in in Chrome. Created isolated `sircus-sim`
machine `5db5a9df-2e56-4711-9d39-b42cc86e4d79` in the `hackathons` organization,
`Oct SF Hackathon` location, alongside sfhack10/sfhack14. The cloud-bound native
runtime on port8082 serves yesterday's unchanged fake-xArm6 scene. A fresh replay
passed7/7 moves; browser3Dscene confirmed ONLINE and all7 world-state objects.
See `../viam-cloud/STATUS.md`, `data/cloud-replay.md` and
`data/viam-scene-connected.png` under that directory.

Physical configs were inspected read-only and saved under `../hardware-observed`.
sfhack10 already references both the drop arm and robot14's board arm; no physical
configuration or motion was changed. The localhost8765 dashboard and850native
runtime remain separate from this cloud baseline. Training records are local;
Viam Data cloud syncing and the new two-arm training loop are not wired yet.

Next input: measured board L/W/thickness/mass, suction support offset, box/table
dimensions, arm bases and tool offset, plus center/edge drop recordings at known
heights. These replace parameters in the existing environment; no rebuild of the
scene architecture is required. Logic Pro audio and the second arm are later work.
