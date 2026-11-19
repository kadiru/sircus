# Sircus physical trial operator panel

Open **http://127.0.0.1:8765/#physical** on this Mac. The existing Simulation tab still talks only to sircus-sim; Physical trials uses a separately pinned sfhack10 client, credential file, calibration and trial directory. There is no automatic pose move, tool actuation, recording or upload on startup.

## Start

From `/Users/kadirumacbook/Documents/ChatGPT/hackathon` (only when port 8765 is not already serving):

```sh
sircus/simulation/.venv/bin/python sircus/simulation/server.py --port 8765 --physical-mode hardware
```

This starts only the dashboard and its SDK clients. It does not start or stop a Viam server. Native Mac simulation remains on port 8082. The default `--physical-mode disabled` is appropriate on Ubuntu: no physical control is enabled unless explicitly selected. Credentials default to `sircus/perception/.private/sfhack10.json`, owner-only mode0600, with `address`, `api_key_id`, `api_key`; never copy them into simulation configs or the browser. `--physical-credentials` accepts another private file for the same pinned sfhack10 address only.

Isolated mock panel (no SDK connection or hardware commands):

```sh
sircus/simulation/.venv/bin/python sircus/simulation/server.py --port 8767 --offline --physical-mode dry_run
```

`--offline` refers to the legacy standalone simulation tab. It is used here only to remove dependencies from the mock operator test; it is not the current two-arm simulation. Physical mode remains visibly labeled DRY RUN.

## Calibration for height-adjustable movement

No physical calibration numbers were copied from simulation. The following are initially missing:

1. Operator-taught **loading pose** of arm 10.
2. Operator-taught **fixed drop location**: arm 10 world X/Y and Viam orientation vector. Teaching also records arm 14's reference joints/pose. Teaching only reads; use established operator controls to physically place the arm first.
3. **Reference Z** in the physical Viam world frame, including the held-ball-to-arm-flange vertical offset. The commanded flange height is `reference_z_mm + requested_height_mm`; a raw board-surface Z alone is generally not this reference. No ball-center TCP calibration has been assumed.
4. Operator-validated **minimum and maximum height**, loading/release route, tool orientation, and fixed board/bucket layout. Save calibration only after checking these physical values. Optional setup revision can identify tape marks or survey notes.

All poses use Motion.GetPose for component `arm` in `world`: kinematic readbacks, not camera-calibrated object coordinates. sfhack10's existing planner obstacles are reused through Motion.Move. The passive `sircus-board-provisional` asset now adds a 200×300×5 mm board under `vacuum-board`, yaw +90°, offset (0,0,122) mm. Its mount is proxy-derived and uncalibrated, and its frame geometry affects planning as well as display. The physical bucket still lacks registered geometry. The panel does not manufacture calibration or reconfigure Viam. Route/clearance validation remains operator work. Any board-joint change >0.5 degrees blocks release/movement until the setup is retaught and confirmed. Bucket immobility is operator-confirmed; no vision localization is claimed. Avoid another controller commanding the board arm: sfhack14's previous config still refers to that same physical controller. See [board asset notes](board-asset-notes.md).

Calibration persists in `data/hardware/calibration.json`. Mock calibration and mock trials are in different directories and never loaded by the hardware backend. Unknown limits remain blank, never guessed.

## One-button taught-pose cycle

For a ball resting on a taped pad, use the separate [per-color pickup station workflow](../../docs/TAPED_PICKUP_STATIONS.md). It adds a 100 mm approach, constrained vertical descent/grip/lift and then the existing saved drop routine. The hand-loading cycle below is unchanged.

The **Load → Drop · 2 sec to load** button uses the exact saved loading and drop poses. Confirm that the board/bucket remain fixed and the route is clear, then press it once. It moves to loading, settles, establishes recording, opens, displays a two-second countdown, closes, checks the driver reports an object held, moves to the taught drop pose, holds the intended pose stable for two seconds and validates fresh recording, then releases. It records two seconds afterward. Practice mode defaults on: completed cycles are retained locally as unlabeled (no cloud upload), and the next button press is enabled without an outcome or Save. Turn Practice off for the labeled workflow. Stopped trials can be kept unlabeled with the separate button; faults still require resolution. No automatic repeat.

Closing is automatic when the countdown ends; keep fingers outside the jaws. Stop cancels pending steps and stops only arm 10 without opening a tool or changing suction. Missing/stale recording, moved board, failed grip, motion/readback failure or an unsettled/wrong endpoint abort the cycle.

This fixed-pose cycle does not use an estimated ball-to-board gap. Trial height is explicitly unknown/null. Reference Z and min/max height are still required for height-adjustable controls. Endpoint teaching does not verify the complete physical route; existing Viam motion planning is used with its current, partly provisional obstacle geometry.

## Operator workflow

1. Teach/configure the setup. Go to loading; Open; manually load; Close. A failed driver `grabbed` result does not arm the trial.
2. Enter a height within calibrated limits and choose **Go to drop position**. A unique trial ID is created. The target preserves saved XY and orientation; only world Z changes.
3. The controller awaits one second of stationary, stable joint and intended-pose readbacks (2mm position tolerance, conservative 1-degree orientation-vector comparison). It then starts a bounded recording and waits for an already saved frame plus both-arm state.
4. **Drop** unlocks only with the intended pose, unchanged board, fresh robot readback and fresh recording. The button rechecks these conditions before opening arm 10's gripper. Three seconds of post-release recording follow.
5. Select Captured, Missed or Unsure, optional miss direction/notes, then **Save trial**. Labels require the exact completed trial ID. Aborted trials can also be saved, preserving their aborted status. A new trial cannot overwrite an unsaved one.

No commands are queued behind an active command. A second command is rejected. Stop uses its own request path, cancels the active task/planner operation and calls only dropper Arm.Stop; it never opens either tool, changes suction, moves/stops arm 14, or calls StopAll. Failed stop/planner cancellation latches a fault. A recovered interrupted trial also requires verified operator Stop before further movement. Network/API failure can leave Stop unconfirmed; this software button does not replace the physical emergency stop. The SDK session stays open during normal Stop.

## Trial data and integration status

`data/hardware/trials/<trial-id>/` contains:

- `trial.json`: requested height, fixed target, calibration snapshot/hash, measured pose before release, bracketing robot states, operator/command times, outcome and notes.
- `observations.jsonl`, RGB files and raw depth files when returned: local request/response timestamps, camera metadata and both-arm state. Capture targets 2Hz, bounded at 120 seconds. Actual sample times are authoritative.
- `recording.json`: readiness, sample counts and errors.
- `trial-metadata.json` and `viam-upload.json`: frozen annotated record and checkpointed Viam upload/Sequence results.

`operator-commands.jsonl` journals teaching/loading/tool actions as well as trial actions. Measured release pose is readback bracketing the gripper command, not measured ball separation time. Camera exposure and joint samples are not hardware synchronized. Raw depth remains device counts; no new metric depth or camera-world calibration is asserted.

Save retains local files, then uses Viam DataSync uploads for the sfhack10 part and creates a trial-tagged Sequence when supported. RGB and tabular joints feed Sequences; raw depth/full metadata are Files. Upload acknowledgements are distinguished from independent cloud verification. Upload failures keep checkpointed local data and report `pending_retry`; successful retries can skip acknowledged items. No cloud training job is started.

The refreshed sfhack10 inventory lacks native `data_manager` and `capture-control`. The adapter has an optional capture-only hook when both are explicitly configured; it does not configure them automatically. The current panel therefore relies on the Mac recorder and explicit Viam uploads. Read-only camera and both-arm connectivity are verified. Actual trial recording, native capture-control and cloud Sequence creation remain to be verified after calibration. Details and current official API references: [docs-adapter.md](docs-adapter.md).

## Verification

```sh
sircus/simulation/.venv/bin/python -m unittest discover -s sircus/physical_trials -p 'test_*.py' -v
```

Tests use independent mocks, not physical credentials. Evidence and browser screenshots are in `evidence/`. Startup/read-only connection probes do not authorize any test actuator command. Current readiness and process state are recorded in [STATUS.md](STATUS.md).

### Gripper command tracking

Loading Open is skipped after an acknowledged panel Open (including release). Startup, Stop, failed readback and uncertain tool commands invalidate this cache. This is command-derived, not jaw-position sensing. External gripper controls are not tracked: issue Open through the panel after external use. Manual Open always executes.
