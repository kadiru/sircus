# Varun Ubuntu package delivered — October29,2026

Separate package handoff/sircus-varun-ubuntu.tar.gz prepared and emailed from kadir@scirobot.com to varun.2891@gmail.com, with inline architecture PNG and checksum attachment. Gmail SENT readback verified attachment names/sizes. Package includes credential-free local8083 dashboard, new fixed-world-XY roll/pitch/gap optimizer and selected-policy native replay.46 baseline tests plus new contract checks passed;4-candidate/2-test synthetic smoke and8-motion native replay passed on Mac. Ubuntu execution remains unverified. Isolated test servers8083/8767 stopped. Physical dashboard, arms, tools and original baseline archive untouched. Current package instructions override historical transfer README startup details. Delivery metadata: ../../handoff/varun-delivery.json.

---

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

# Physical dashboard connection and board representation — October 29, 2026

The Mac dashboard's **Physical trials / sfhack10** tab at http://127.0.0.1:8765/#physical now has a verified live camera feed and both arms' state. The hardware client uses negotiated transport instead of forced direct TLS. Twenty-one mock unit tests pass; physical calibration remains blank, so loading/drop movement and release remain disabled. Open/Close is available to the operator when fresh stationary states are present; neither tool nor any arm was commanded, including Stop.

Added only the passive `sircus-board-provisional` component to sfhack10: measured 200×300×5 mm box, parent `vacuum-board`, yaw +90°, provisional proxy-derived offset (0,0,122) mm. Config reload, runtime geometry and native 3D display were verified. It participates in planning but is not a calibrated collision envelope; physical bucket geometry is still absent. Both arms stayed stationary with zero returned joint deltas. Trial recording, uploads/Sequences and physical command execution remain unverified.

Final dashboard verification at 2026-10-30T02:17:59.564485Z returned three successive fresh connected, stationary snapshots with no trial active. Dashboard PID42976 is running; read-only probe clients exited. Native simulation PID19813 remains live on8082. No simulation scene, policy, prior results or frozen Ubuntu archive changed. See [physical status](../physical_trials/STATUS.md), [operator instructions](../physical_trials/README.md) and [board provenance](../physical_trials/board-asset-notes.md). Historical sections below retain their original scope and results.

# Ubuntu baseline frozen; stationary perception checked — October 29, 2026

Transfer archive: `../../handoff/sircus-ubuntu-baseline.tar.gz` (1,333,072 bytes; SHA-256 `11decff3f9fa9956bd4aba6d727f76b506d868fc0b0679bfa47360dc041d592e`). It contains the minimum simulation code/assets/dependencies/licenses, handoff, credential placeholders, current policy, separate bucket/rectangular evidence, and checksum manifests. Independent path/checksum/Python-syntax/dependency/known-secret checks passed. No credentials, Mac binaries, venvs, caches or upload queues are included. The original bucket fingerprint and saved policy restore unchanged. The archive was frozen before perception work and is not being modified afterward. The Mac simulation remains running; shutdown/cutover to the same cloud part on Ubuntu must be coordinated explicitly.

Stationary sfhack10 perception is separate under `../perception/`. User confirmed camera on arm 10's tool bracket, unchanged from photos; configured parent is `arm`. Read-only status identified assigned serial 112322070710. Later user photographs identify the camera as D435 with the same serial and corroborate its tool-flange bracket mount; camera-to-tool extrinsics remain unverified. Preferred dataset `../perception/data/20261030T001623700791Z/` contains three RGB/depth pairs, reported intrinsics/configured frames and both arms' before/after joint states. Both arms reported not moving and all returned joint deltas were zero. No movement, tool actuation, camera tuning, physical configuration change or simulation behavior change was issued.

RGB and raw depth capture succeeded. SDK point-cloud retrieval failed, but a later Viam-browser raw PCD download succeeded (921,600 points; 86.47% positive finite). It is saved separately, with unknown exposure time and no matching joint-state bracket. A cross-time depth/PCD consistency check supports 0.001 m/count; known-distance calibration is still missing. Original depth arrays retain device counts. RGB/depth alignment has configuration/source support and a limited edge check, not calibrated registration. The visible ball has a repeatable RGB center/bearing, but poor depth makes its metric 3D center unknown. Board and bucket are outside the view; their poses/normals are missing. No world-frame pose is claimed calibrated. Historical board photos support ChArUco-style 5×5 markers, but printed square/marker dimensions and layout need confirmation.

Full capabilities, initial localization, exact missing measurements and runnable commands: [stationary report](../perception/STATIONARY_RGBD_REPORT.md), [capture instructions](../perception/CAPTURE_README.md). Capture clients have exited; no capture loop remains. Next suggested perception step is a dimensioned matte reference target at this same arm pose to establish depth scale/registration before metric localization. Earlier simulation results below retain their original meaning.

# Tapered bucket and landscape board — October 29, 2026

The active scene uses the measured open-top bucket: inner diameters 177 mm top / 150 mm bottom; outer diameters 200 mm top / 158 mm bottom. A straight-sided taper is the initial approximation. Analytic ball contact and capture use the inner geometry; robot clearance uses 24 conservative outer wall boxes and a floor box, with no lid. Live WorldStateStore inspection confirmed 28 objects, including all 24 bucket walls, with no stale rectangular-container transforms.

Height remains **reported 175 mm**, unresolved as overall height versus interior depth. Simulation assumes interior depth 175 mm and an unmeasured 8 mm bottom thickness, deriving 183 mm outer height. Bucket center [680,245] mm and inner floor Z 27 mm remain provisional. The measured 200×300×5 mm board now has a 90° suction mount rotation for landscape orientation; board Euler angles [0,30,90]° preserve the prior plane normal and cup orientation. Arm14 base spacing stays 1337.31 mm.

Bucket training session `20261029T232904069256Z-2a7bbd1f`: 24 proposals and 10/10 retained synthetic held-out tests (±2 mm XY, ±3 mm gap). Best U/V/gap: −16.408 / −44.821 / 161.715 mm. The policy was frozen before evaluation. Selected-release replay `20261029T233247252271Z-67818bd3` passed eight native Viam motion plans/readbacks and retained the modeled ball. All 37 training files and 3 native replay files were visibly verified in Viam DATA > Files.

Current fingerprint: `bb6bc13e1d38ac1bba4d12b11f3c249396e06acd4540097bfcbb3455706c30eb`. The policy cache restored the matching bucket session and rejected four in-memory geometry/mount changes. All 75 earlier archive files match `history/rectangular-20261029/archive-before-bucket-sha256.json`. The original rectangular scene and sources are preserved there; historical results below do not describe the bucket.

Validation: 13 bucket physics checks, 15 legacy physics checks, 5 native geometry/mount checks, 9 training/data checks, and 4 cloud guard checks passed. Dashboard inspection confirmed open tapered geometry, landscape board, measured/provisional labels, restored policy, and successful replay. No physical robot actuation was performed. These results concern one simulated board pose and provisional contact dynamics; they do not establish hardware success or generalization.

Evidence: [combined audit](data/bucket-verification.json), [training](data/bucket-training-live.json), [native replay](data/bucket-native-live.json), [cloud file verification](data/bucket-cloud-upload-verification.json), [live geometry inventory](data/bucket-wss-verification.json), [dashboard](data/bucket-dashboard.png), and [browser checks](data/bucket-dashboard-verified.txt).

## Physical-machine wall-right configuration

In sfhack10, wall-right frame Y changed from −450 to −750 mm and box X size changed from 1500 to 1000 mm. UI save, reload, and full JSON comparison confirmed exactly these two fields changed. No physical motion was commanded. Evidence: `../hardware-observed/wall-right-20261029/change.json` and `wall-right-verified.png` in that directory. This physical-machine obstacle edit is separate from the isolated simulation results above.

## Earlier rectangular-container history

# Spacing update — October 29, 2026

User measured52.65in, converted exactly to1337.31mm. Updated simulation arm14 baseX and physical sfhack10 arm-board frameX from1370mm. Y/Z and180°yaw retained. Physical config change was limited to this one field; no physical motion was commanded. Simulation controller and dashboard were refreshed. A new simulated drop passed eight native planned motions and retained the modeled ball; evidence `data/spacing-1337_31-validation.json`, session `20261029T223843892714Z-a36e7683`. Previous trained policy no longer matches the geometry and is not restored; run Learn release for a fresh policy. Earlier results below apply to the1370mm setup.

# Verified October29,2026

The two-arm setup is active on cloud machine sircus-sim. Arm10 drops; arm14 positions the200×300×5mm measured board. The localhost8765 dashboard sends commands through the SDK to the Viam module on8082. The module runs simulation search and uses Viam Data Manager to upload JSON records.

The dashboard-triggered session `20261029T221947799789Z-282ce37b` completed24 proposals and10 held-out noisy tests:10/10 synthetic captures. Best release:U0/V50/gap140mm. Policy was saved before tests. The following dashboard Run drop session `20261029T222019143752Z-6cfdf8d8` passed eight Viam motion plans/readbacks and retained the modeled ball. All37 training files and3 native replay files were visibly confirmed in Viam DATA > Files, including the policy. Local archives remain intact.

Evidence: `data/end-to-end-verification.json`, `data/viam-cloud-files-visible.txt`, `data/archive/<session>/`, and the scene/dashboard screenshots. Earlier isolated8083 test records were also synced; the provenance-correction session98370fd1 identifies them. The temporary8083 server is stopped.

Validation:15 physics checks,9 training/data checks,4 cloud guard checks, and native logging regression. Dashboard controls, learning, native drop, reload, both arm renderings and correct dimensions were checked. The runtime rejects physical drivers and is pinned to the simulation part. Physical machines10/14 were not actuated or reconfigured.

This is a fixed-board-pose release optimizer with modeled contact. Training proposals receive kinematic endpoint checks; the selected release was separately collision-planned through Viam. Mass, suction/tool offsets, support stiffness/damping and other installation dimensions remain provisional. Physical calibration, vision-based success detection and transfer to real hardware remain next steps.
