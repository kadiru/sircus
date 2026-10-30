# Skip redundant loading Open — October 29, 2026

Loading Open is skipped only when the last acknowledged dashboard gripper command was Open. The release Open marks that state for the next practice cycle. Every tool command first invalidates the cache; failure/cancellation cannot leave stale Open state. Startup, Stop and failed robot readback reset it to unknown. State is command-derived, not jaw sensing; external Viam/controller changes are not observed. After using external gripper controls, issue Open once through this panel before relying on the cache. Manual Open always sends the command. Release checks, two-second timing and practice retention remain.

34 mock tests passed, including two consecutive cycles with three Open calls instead of four, failed-close invalidation and readback-loss invalidation. Deployed with one idle dashboard reload; no actuator command sent. The first post-restart cycle opens normally. Tracking is visible in Recording and integration details and skip events are recorded in each trial.

---

# Two-second timing — October 29, 2026

Dashboard PID47928. Loading countdown is2s (was5s). Fixed-pose release now uses one continuous2s stable-pose check instead of two successive1s checks. Post-release recording is2s (was3s). RPC and motion/recording readiness may add time; checks are not bypassed. Practice retention remains enabled.32 mock tests pass, including an actual2s drop hold. One idle dashboard reload was performed; no actuator commands. Previous updates also reloaded this process, causing brief reconnects. Logs additionally contain intermittent RPC channel-closed errors after restart; their root cause is not established. Existing adapter reuses its SDK client between cycles; this change does not claim to fix network reliability.

---

# Repeat practice enabled — October 29, 2026, 8:02 PM PDT

Dashboard PID47590. Practice mode defaults on: each successful operator-triggered fixed-pose cycle retains all local observations and trial metadata with outcome=null, label_status=unlabeled, saved=true, and cloud upload deferred. The next Load → Drop press is enabled without labeling. Each cycle still requires a distinct operator press; no automatic loop. Setup confirmation stays checked for successful practice cycles, resets on Stop/error, and resets on page reload. All existing motion, board, grip, recording and Stop guards remain.

Keep unlabeled is also available for finished/stopped trials; it preserves their status and never clears a fault. Existing trial20261030T025828035512Z-9616471b was retained unlabeled with its recorded aborted status intact (Stop was issued after the release). Original record preserved in evidence/practice-repeat/prior-trial-before-retention.json. No motion or tool commands were sent during this update. Live connected stationary state and enabled button verified; operator starts next cycle.31 mock tests passed, including two consecutive practice cycles with independent retained IDs and no cloud upload, plus wrong-ID/active-trial rejection. Simulation was not changed.

---

# One-button fixed-pose cycle — October 29, 2026, 7:58 PM PDT

Deployed on http://127.0.0.1:8765/#physical, dashboard PID47186. New Load → Drop button uses the operator-taught loading and drop poses: planned move to loading, recording readiness, open, visible five-second wait, close with held-object check, planned move to taught drop, settled/pose/board/fresh-recording checks, release, three-second post-recording, then operator outcome and Save. Stop interrupts pending steps and targets only arm10; no board-arm or suction command is included. Setup confirmation is unchecked on the live UI; no physical cycle, motion, tool actuation or Stop was executed during this implementation.

Twenty-nine mock unit tests passed. Browser dry-run completed and saved a captured mock trial with a 5.002-second loading wait and explicitly unknown height. Mock server PID47044 is stopped. Read-only live verification confirms sfhack10 connected, both arms stationary, both taught poses present, no active trial, and fixed cycle eligible after operator setup confirmation. Metric reference Z and height limits remain missing; the estimated 10cm gap was not adopted. Height-adjustable controls remain locked. Physical route execution, actual trial recording and cloud upload still require operator validation.

Changed controller.py, test_fixed_cycle.py, simulation/web/index.html, simulation/web/physical.js and operator README/status. Evidence: physical_trials/evidence/fixed-cycle/ (mock trial, verification.json, read-only live state and dashboard-ready.png). Simulation code/scene/policies and frozen transfer archive were not changed.

Runtime caveat: the previously recorded native simulation PID19813 is no longer present and port8082 has no listener at this check. Its exit time/cause is unknown. This task did not signal or restart it; only dashboardPID42976 and isolated mockPID47044 were terminated. Do not infer the old runtime remains live from earlier status sections.

---

# Physical-trial panel — October 29, 2026

Implemented and deployed on this Mac at **http://127.0.0.1:8765/#physical**. Dashboard PID **42976**, manual detached process, no autostart. The dashboard connection to sfhack10 is restored, with a live camera preview and both arms' state verified through the HTTP endpoints and browser. Native sircus-sim Viam PID **19813** remains running on port **8082**; its runtime, scene, policies, historical results and frozen Ubuntu archive were not changed. The mock server on 8767 and read-only probe clients have exited.

The only physical-machine configuration addition was the passive `sircus-board-provisional` component described below. No physical motion, tool command (including Stop), trial recording or upload was issued.

## Gripper-only hardware check — 7:31 PM PDT, October 29

After the operator confirmed hands clear, one Open then Close cycle was issued through the actual dashboard buttons to sfhack10/gripper. Both Viam calls completed without errors; Grab returned false (no object reported held). Jaw position was not independently visually measured. Both arms reported stationary with zero joint deltas. No arm motion or board-suction commands were issued. Open/Close remain available in Physical trials → Load the ball. Evidence: evidence/gripper-smoke/verification.json and dashboard-after.png. Earlier no-actuation statements below refer to the connection/board work before this test. Physical movement, trial recording and release workflow remain untested.

## Ready in code / mock-verified

- Separate sfhack10 hardware configuration, credentials and data; simulation routing and fake-driver guards retained.
- Open/Close, loading pose, height-controlled drop pose, Drop, independent Stop, Captured/Missed/Unsure, optional direction/notes and Save.
- Read-only teaching of loading and fixed drop location; world XY/orientation preserved and only height varies. Board reference-joint guard; manually confirmed fixed bucket/layout.
- One active command/trial; intended-pose, settled-state and fresh saved recording interlocks; immediate release revalidation. Failed grab, stale state, invalid heights, board changes and unverified Stop block further movement. Interrupted trials recover without automatically restarting motion.
- Local RGB-D/both-arm recording, trial IDs, requested height, measured release pose/state, command timestamps and outcomes. Viam DataSync upload and Sequence creation implemented with checkpointed local retention.
- Stop only targets arm 10 and its tagged pending planner operation. It never opens tools or releases suction. Unverified cancellation remains latched rather than becoming success on a second Stop.

## Verification and limits

**21 unit tests passed** (16 controller + 5 adapter), plus Python/JS syntax checks and HTTP host/origin/token rejection checks. The earlier full mock browser workflow saved a 100mm Captured trial and a 150mm Stop-before-release Unsure trial. Recording was ready before the mock release. Both mock trials are explicitly labeled dry_run and stored separately from hardware. A cancellation-resistant transport regression verifies that a late move completion cannot resume recording/release after Stop; the additional adapter test covers the repaired cloud connection options.

The original forced direct-TLS connection failed. The adapter now uses the SDK's negotiated transport with WebRTC enabled and TLS validation unchanged. Repaired read-only probes saved three live camera frames and three paired both-arm snapshots. Actual dashboard HTTP status, JPEG delivery and browser camera/state display were subsequently verified. Final dashboard verification at **2026-10-30T02:17:59.564485Z** returned three successive fresh connected snapshots: both arms stationary with zero returned joint deltas, phase `idle`, no trial, and simulation PID19813 still present. Evidence: `evidence/connection-repair/dashboard/verification.json`. An initial transient read timeout is preserved in the earlier probe evidence.

Calibration remains blank, recording inactive and no trial is active. Loading/drop movement and release remain disabled. **Open/Close gripper controls are available to the operator when fresh stationary states are present**, even before pose calibration; they were not pressed during this work. Hardware motion, tool actuation, trial recording, actual Viam uploads/Sequences and Stop response remain unverified. No physical trial files exist.

Native data_manager/capture-control remain absent from the refreshed physical inventory. Optional capture-control integration is implemented but disabled/unconfigured. The immediate path is Mac recording plus explicit DataSync uploads on operator Save. Cloud acknowledgement and independent cloud verification remain separate.

## Board representation installed

Added `sircus-board-provisional`, a passive `rdk:component:generic` / `rdk:builtin:fake` component parented to `vacuum-board`. Its box is **200×300×5 mm**, with **+90° yaw** and translation **(0, 0, 122) mm**. The size and landscape orientation come from the user. Centered X/Y and the 122 mm offset are provisional: the offset follows the configured vacuum-tool geometry proxy, not a measured AS1200 mounting transform.

Saving and reloading confirmed that only this component was appended. Runtime checks passed for parent, pose and box dimensions; both arms stayed stationary with zero returned joint deltas. Viam 3D Scene displayed the attached board. A brief scene pose-update warning resolved; no physical movement was used to test the representation. `frame.geometry` participates in motion planning as well as display, but this provisional attachment is not a calibrated clearance model. Physical bucket geometry is still absent.

Asset and provenance: [assets/board.json](assets/board.json), [board-asset-notes.md](board-asset-notes.md). Verification: `evidence/board-asset/runtime.json` and `evidence/board-asset/viam-board.png`.

## Missing before first physical trial

1. Taught arm-10 loading pose.
2. Taught fixed release XY/orientation and arm-14 reference pose.
3. Physical world reference Z accounting for the held-ball-to-flange offset.
4. Validated minimum/maximum height and loading/release route clearances.
5. Measured board attachment and registered physical bucket geometry.
6. Operator confirmation of fixed board and bucket placement, plus exclusive board control (sfhack14's previous config references the same board-arm controller).

Camera-world calibration and visual bucket/board localization remain unverified; the nominal board representation does not establish observed object poses or replace operator calibration.

Instructions: [README.md](README.md). API/integration detail: [docs-adapter.md](docs-adapter.md). Current connection evidence: `evidence/connection-repair/summary.json`, `verified/`, `readbacks/` and `dashboard/` beneath that directory. Earlier mock evidence remains in `evidence/final-tests.json`, `final-tests.txt`, `http-guards.json`, `integration-ui.json`, `deployment.json`, and screenshots. Mock browser evidence predates the final clarification splitting prior operator context from current-trial events; final source keeps each trial's event list separate.
