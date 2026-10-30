# sfhack10 operator adapter

## Connection repaired — October 29, 2026

The cloud client had forced `disable_webrtc=True`, copied from the loopback simulator. That direct-only path timed out. It now defaults to the SDK's negotiated WebRTC/TLS transport with TLS verification unchanged. `transport="direct"` exists only as an explicit local diagnostic setting; the dashboard uses the working `auto` default. The transport repair and its probes changed no address, credential or robot configuration. A separate passive board-component addition is documented below.

The repaired adapter connected successfully and saved **three live RGB frames and three paired both-arm state snapshots**. Both arms reported not moving throughout and all six joint deltas on each arm were zero at returned precision. Connection took 9.49 seconds in the final probe. A separate per-RPC diagnostic returned all joint/IsMoving/world-pose reads in about 22–26ms and RGB-D GetImages in 576ms. An initial repaired probe had a transient five-second read timeout; that failed attempt is retained rather than overwritten. Camera image inspection shows the real teal ball and tabletop. No motion, gripper/vacuum, capture-control or configuration commands were sent.

Evidence: `evidence/connection-repair/summary.json`, `verified/probe.json`, `verified/camera-00.jpg` through `camera-02.jpg`, and `readbacks/readbacks.json`. The read-only probes exited. Temporary SDK network logs were removed; no credentials are in this evidence. `probe_connection.py` is the repeatable camera/state check; run it with the existing SDK environment and an outer process timeout as documented in its module header.

The running dashboard's HTTP status, live JPEG endpoint and browser camera/state display were also verified. Final verification at **2026-10-30T02:17:59.564485Z** returned three successive connected snapshots, both arms stationary with zero returned joint deltas, phase `idle` and no trial. Evidence is in `evidence/connection-repair/dashboard/verification.json`. These checks verify read-only connectivity and previews, not physical motion, calibration, Stop response, trial recording, Viam uploads or Sequences. Those limitations below still apply. No command buttons were pressed.

The Mac operator adapter is `viam_adapter.py`. It imports the existing simulation Python environment (`viam-sdk==0.85.1`) but does not reuse its credentials, simulator module, scene, or fake-arm controls.

## Binding and readbacks

The address is pinned to `sfhack10-main.ty1cywh7tq.viam.cloud`, machine `3103375b-9bab-4222-825d-b3fb9b2897fa`, part `29154d15-3b8c-4097-a3c6-2bc9d04a6e12`. Credentials are read from a separate JSON file with `address`, `api_key_id`, and `api_key`, owned by the current user, nonsymlink, mode 0600. The existing file is `../perception/.private/sfhack10.json`. Never include its contents in a handoff or browser response.

Required resources are `arm` (dropper), `arm-board` (read only), `gripper` (dropper tool), `cam`, and `builtin` motion service. Arm poses come from `Motion.GetPose(component, "world")`; they are kinematic poses in the configured frame system, not vision-derived poses or independently calibrated world coordinates. Both arms' joints and IsMoving state are read separately. No vacuum component handle is created. No board-arm mutating method is exposed.

Snapshot times bracket independent RPCs. RGB/depth share the driver's GetImages response metadata; individual exposures and arm observations are not hardware synchronized. Raw depth remains device counts without an asserted metric scale or world transform.

## Command boundaries

The controller must serialize every action except Stop and validate taught poses, height limits, fixture/board constancy, settled state, and recording readiness. The adapter only supplies named methods; it offers no arbitrary component, pose-frame, DoCommand, robot-address, or sequence-execution endpoint to the browser.

`move_pose()` uses Viam Motion.Move on component `arm`, destination frame `world`. It never substitutes direct Arm.MoveToPosition. Collision checks can only cover geometry actually configured in sfhack10. The newly registered `sircus-board-provisional` is a passive generic/fake component with a 200×300×5 mm box, parent `vacuum-board`, yaw +90° and translation (0,0,122) mm. Its frame geometry participates in planning, but the proxy-derived attachment remains uncalibrated. The physical bucket is not yet a registered collision asset. Operator calibration and route validation remain necessary; [board asset notes](board-asset-notes.md) document provenance and runtime verification.

Each Move gets a unique operation tag. Stop immediately schedules dropper Arm.Stop, cancels only operations with that exact marker and component `arm`, cancels the local move task, and reasserts Arm.Stop. It never calls StopAll, Gripper.Stop/Open, board-arm Stop, or vacuum commands. If an active planner cannot be confirmed cancelled, `planner_cancellation_verified` is false; the controller must surface an unconfirmed Stop. No return value is a substitute for readback or a physical emergency stop.

Normal Stop keeps the robot client/session alive. `close()` is only for explicit application shutdown. Viam session-heartbeat behavior can stop actuators previously commanded by that session; a held object should be secured before deliberate shutdown.

## Recording and Viam data

Recording starts before release and saves RGB, raw depth when returned, response metadata, both arms' pose/joints/moving state, and local request/response timestamps to the trial directory. Readiness requires an already persisted JPEG/PNG plus both arm states, no current capture error, and a last saved sample no older than three seconds. The bounded recorder targets 2 Hz and stops after 120 seconds. Slow RPCs reduce achieved frequency; actual timestamps are the evidence.

On operator Save, `trial-metadata.json` stores the supplied trial ID, requested height, measured release pose, command/event times and outcome. The adapter uses Viam DataClient/DataSync uploads associated with the sfhack10 part. RGB gets capture timestamps; joint/pose tabular records get their state-request windows. Raw depth and full metadata/journal files are uploaded to Files. Successful upload IDs are checkpointed in `viam-upload.json`, so a retry can skip completed uploads. Local copies are retained. Cloud acknowledgement and independent DATA-page verification are distinct fields.

After upload, CreateSequence references `cam/GetImages`, `arm/JointPositions`, and `arm-board/JointPositions` for the trial window, tagged with trial ID and outcome. Sequence datasets include JPEG/PNG and tabular data; depth remains in Files. The adapter has no API to execute a robot behavior sequence.

Optional `capture_control_name` enables native capture only when that named sensor and a data_manager service are both discovered. The adapter refuses to take over a sensor that reports an existing active sequence/capture. It uses the documented `start_capture` and `stop_capture` commands, only stops capture it started, and does not mark native file/cloud collection verified based only on sensor acknowledgement. Native capture-control configuration is not auto-applied by this implementation.

The earlier inventory from `../perception/data/20261030T001623700791Z/metadata.json` had **no data_manager or capture-control resource**. The repaired connection's fresh 12-resource inventory likewise has neither. The earlier direct-TLS read-only connection attempt failed with TimeoutError; later WebRTC-enabled readbacks are verified above. No arm, gripper, vacuum, capture-control, or configuration command was issued by either probe. They exited and left no probe running.

## Verification

- Installed SDK signatures inspected for Motion.Move/GetPose, Arm.Stop, GetOperations/CancelOperation, camera GetImages, DataClient binary/tabular/file uploads, and CreateSequence.
- `python -m unittest discover -s sircus/physical_trials -p test_viam_adapter.py`: **5 tests passed**. They cover the WebRTC-enabled cloud dial with TLS validation, persistence before recording readiness, timestamped data/sequence uploads and checkpoint retries, cancellation limited to the panel's dropper operation while an unrelated board-arm operation remains untouched, and repeated Stop retaining an unverified planner cancellation instead of reporting false success.
- Import/compile passed in the existing Mac Viam environment.
- Physical command execution, actual uploads, actual sequence creation, and native capture-control remain **unverified**. These are exercised only by operator actions after calibration, not by the tests. Current camera/state connectivity is independently verified above without issuing commands.

## Official references checked

- [Viam motion API](https://docs.viam.com/motion-planning/reference/api/): Motion.Move planning and execution; built-in StopPlan is not an arm Stop replacement.
- [Viam gripper API](https://docs.viam.com/reference/apis/components/gripper/): Open, Grab, and holding-state semantics.
- [Viam capture on demand](https://docs.viam.com/data/capture-sync/capture-on-demand/): capture-control sensor configuration and commands.
- [Viam Sequences](https://docs.viam.com/data/sequences/): sequence windows, supported image formats, resource filters, and CreateSequence.
- [Viam SDK example](https://python.viam.dev/examples/example.html): timestamped DataClient uploads.
- [Viam client connection behavior](https://docs.viam.com/build-apps/concepts/how-apps-connect/): session safety heartbeats.
