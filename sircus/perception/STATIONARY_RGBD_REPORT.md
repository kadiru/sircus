# Stationary sfhack10 perception validation

October 29, 2026, America/Los_Angeles. Preferred capture: **17:16:23–17:16:35 PDT** (`20261030T001623700791Z` in UTC). This work was performed after the Ubuntu simulation snapshot was frozen. No arm movement, gripper/vacuum actuation, camera tuning, or Viam configuration changes were issued. The Mac simulation remains running.

## Transfer package completed first

Archive: `/Users/kadirumacbook/Documents/ChatGPT/hackathon/handoff/sircus-ubuntu-baseline.tar.gz`.

- Size: **1,333,072 bytes**; **194 regular files**.
- SHA-256: `11decff3f9fa9956bd4aba6d727f76b506d868fc0b0679bfa47360dc041d592e`.
- External manifest: `handoff/sircus-ubuntu-baseline.manifest.json`; internal `MANIFEST.json` and `SHA256SUMS`.
- Independent checks passed: safe archive paths, no links/native binaries/excluded directories, file checksums, Python syntax, dependencies/licenses, and known-credential byte checks. Sensitive JSON/private-key scans found no credentials.
- The current policy restores with fingerprint `bb6bc13e1d38ac1bba4d12b11f3c249396e06acd4540097bfcbb3455706c30eb`. The bucket training and native replay have their own 37+3 records; 75 earlier archived files remain separate and unchanged.
- Bucket measurements and uncertainty are preserved: inner diameters 177/150 mm; outer 200/158 mm; reported height 175 mm remains unresolved; modeled interior depth 175 mm plus assumed 8 mm base; placement provisional.
- Linux setup and cutover instructions are in the archive. No Linux runtime has been started or tested here. Do not serve the same Viam cloud part from Ubuntu until the Mac shutdown is explicitly coordinated. This archive is immutable; later perception work is intentionally outside it.

## Camera identity and mounting evidence

Physical machine: **sfhack10**, resource **cam**, address `sfhack10-main.ty1cywh7tq.viam.cloud`.

1. **User-confirmed physical arrangement:** camera on **arm 10's tool bracket**, unchanged since the supplied photos. `../measurements/photos-9304/IMG_9300.png` corroborates a camera housing on that bracket above the two-finger gripper. Mounting is not inferred merely from a configured parent.
2. **Observed configuration:** parent frame `arm`; translation `[83,−14,18]` mm. Viam orientation vector approximately `[−0.030391,−0.003538,0.999532]`, angle `−97.731173°`. The frame is inherited from the existing workcell configuration.
3. **Read-only device status:** at 17:18:35–17:18:39 PDT, `Camera.get_status()` returned assigned serial **112322070710**, `streaming=true`, `physical_camera_assigned=true`, and `is_recovery_mode=false`. See `camera_status_followup.json`. This status was collected after, not simultaneously with, the images.
4. **Photo follow-up:** the supplied `IMG_9307.HEIC` label identifies **D435**, serial **112322070710**, matching the API status. `IMG_9308` and `IMG_9309` show the metal bracket beside the tool flange above the gripper, with the optical face directed toward the gripping workspace. Evidence and source checksums: `../measurements/photos-9309/PHOTO_EVIDENCE.json`. These photos establish identity and mounting arrangement, not metric extrinsics. Configured serial selection remains empty; no device-selection configuration was changed.

The inherited camera-to-arm transform is **not independently verified calibration**. The identity extrinsics in Camera GetProperties describe the driver's internal camera frame convention, not a tool-to-camera or world calibration.

## Capabilities actually tested

| Capability | Result and evidence | Limit |
|---|---|---|
| RGB | Three 1280×720 JPEG images saved in the preferred sample | One stationary view; current scene shows ball, tabletop and part of the gripper |
| Depth | Three paired 1280×720 Viam DEP payloads decoded to uint16 arrays | Values retained as **raw device counts**, not labeled mm/meters |
| Global depth support | 86.82–86.99% nonzero pixels; median 283 counts in each frame | Nonzero does not mean accurate; ball depth is much worse |
| Temporal depth consistency | First/last common-valid pixels: median absolute change 0 counts, 95th percentile 8 counts | A static tabletop check, not calibrated range accuracy |
| Point cloud | Viam browser GetPointCloud succeeded; raw binary PCD downloaded at 17:25:49 PDT, containing 921,600 XYZ/RGB records, 796,948 positive finite points | Earlier SDK RPC failed after 10.70 s and direct TLS timed out. Browser cloud is a later frameset with unknown exposure time, not synchronized with preferred images/joints |
| Intrinsics | Retrieved fx **906.8481**, fy **906.8774**, cx **651.4015**, cy **362.5878**, dimensions 1280×720 | Driver-reported, not independently recalibrated |
| Distortion | Reported `inverse_brown_conrady`, five coefficients all zero | Metadata, not proof that residual distortion is zero |
| Depth units/scale | DEP stores raw uint16 counts; later PCD XYZ uses meters. Cross-time comparison supports **0.001 m/count** (96.57% of comparable pixels within 1%) | Scale consistency, not known-distance calibration; driver units setting was not directly queried |
| RGB/depth alignment | `align_color_depth=true` observed in configuration; driver source applies software alignment; outputs have matching dimensions | Empirical calibration not established. Limited table-edge check below |
| Joint states | Both arms read before/after images, each reporting not moving; six joint deltas per arm all **0° at returned precision** | Not exposure-synchronized; does not establish all-between-samples motion or camera calibration |
| Device status | Assigned serial and streaming/recovery flags read successfully | Follow-up observation, not camera exposure metadata |

### Alignment check

An offline heuristic compared RGB and depth discontinuities along the visible right tabletop edge over 31 sampled rows per frame. Median absolute offset was **4.5 pixels**, 95th percentile **11.5 pixels**, maximum **18.5–20.5 pixels**. These are manually chosen edge-search regions with holes/outliers and ambiguous boundaries. They provide limited consistency evidence only, not a calibrated registration error or proof that ball pixels have valid corresponding depth. Full details: `alignment_edge_check.json`; implementation: `inspect_alignment.py`.

### Units: do not apply an undocumented conversion

The exact deployed module version inspected was RealSense **0.22.6**, commit `8caec2f4b60887e71b7c12d0d27fad0bff5740da`. Its depth encoder copies raw uint16 values without multiplying by the device's depth scale. Its PCD encoder writes RealSense vertex XYZ directly, whose source contract uses meters. Therefore generic Viam millimeter conventions must not be substituted for this specific encoding. A later browser PCD contained 921,600 records with 796,948 positive finite points (86.47%). Its Z 5th/50th/95th percentiles were 0.282/0.283/1.040 m; the 9.646 m maximum is an outlier. Projection against the earlier depth image produced 784,070 nonzero comparisons: median ratio **0.00100000008 m/count**, with **96.57% within 1% of 0.001**. This supports the usual 1 mm/count interpretation. Because these are different times/framesets and share the same sensor, this is a consistency check, **not independent metric calibration**. Original DEP arrays remain stored in raw counts.

Primary sources and retained excerpts: [exact module source](https://github.com/viam-modules/viam-camera-realsense/tree/8caec2f4b60887e71b7c12d0d27fad0bff5740da), `reference/SOURCE_PROVENANCE.json`, `reference/encoding.cpp`, and `reference/realsense.hpp`. RealSense [point-cloud conversion](https://github.com/realsenseai/librealsense/blob/master/src/proc/pointcloud.cpp) applies the [depth-frame units scale](https://github.com/realsenseai/librealsense/blob/master/include/librealsense2/hpp/rs_frame.hpp). Source files were read, not executed on the robot.

## Observation dataset and synchronization

Preferred dataset: `data/20261030T001623700791Z/`.

- `frame_00/01/02_color.jpg`: original RGB.
- `frame_00/01/02_depth.dep.gz`: losslessly compressed original DEP responses.
- `frame_00/01/02_depth_raw.npz`: decoded raw counts; depth preview PNGs are contrast-scaled displays only.
- `camera_properties.json`, `configured_frames.json`, `metadata.json`: driver metadata, configured frame graph, RPC timing and arm states.
- `camera_status_followup.json`: later read-only status/serial response.
- `initial_localization.json`, `alignment_edge_check.json`, `frame_*_fiducials.json`, `inspection.json`: derived inspection results.
- `MANIFEST.json` preserves the original capture file checksums; `DERIVED_MANIFEST.json` records subsequent analysis/context files separately.

Reported image-response timestamps: **00:16:31.157Z, 00:16:32.558Z, 00:16:33.692Z on October 30 UTC** (17:16:31–17:16:33 PDT on October 29). Local request/receive intervals are retained. RGB and depth come from one driver frameset response, but individual sensor exposure timestamps/hardware synchronization were not returned. Camera/server/Mac clock synchronization was not verified. Joint reads bracket the images and are not hardware-triggered with them. The successful browser point cloud was downloaded later at **00:25:49.121Z** into `data/20261030T002549121000Z-browser-pointcloud/`. Its exposure time and exact retrieval interval are unavailable. It has no contemporaneous arm-state bracket and is **not paired** with the preferred RGB-D sample. The browser cloud preview was turned off afterward; RGB preview remained on manual refresh.

Joint readbacks, degrees, identical at the before/after samples:

- `arm` (dropper 10): `[-87.891721,−24.697925,−49.114931,−1.332422,−24.103787,8.059570]`.
- `arm-board` (board holder 14): `[186.302084,67.923854,−240.481262,−203.125127,87.693304,27.573815]`.

Earlier attempts are retained, not silently overwritten:

- `20261030T001321684082Z`: three RGB/depth pairs and before joint reads succeeded; PCD broke the connection, so after reads failed. It does not establish a complete stationary bracket.
- `20261030T001436735115Z`: direct-TLS connection timeout, metadata only.
- `20261030T001617665332Z`: local execution sandbox blocked the transport proxy; no successful sensor dataset. The authorized retry produced the preferred sample.

## Initial localization, with provenance

| Quantity | Result | Evidence class |
|---|---|---|
| Ball image location | Mean silhouette center **[367.53,539.34] px** | Observed RGB segmentation, repeatable over three frames; not a 3D center |
| Ball camera bearing | Approximate unit direction **[−0.29370,0.18287,0.93824]**, axes +X right, +Y down, +Z forward | Computed from silhouette center and reported intrinsics; range unknown |
| Ball 3D center/radius | **Missing / unknown** | Only 15.16–16.55% of mask pixels have nonzero depth, with large far outliers. Unconstrained sphere fits failed consistency checks |
| Board impact surface/normal | **Missing / unknown** | Board not visible in current camera frames; zero fiducials detected |
| Bucket top-opening center/normal | **Missing / unknown** | Bucket not visible in current camera frames |
| Calibrated world poses | **None** | No known physical reference or verified camera-to-robot transform |

Ball sphere fitting used no assumed 75 mm diameter. Candidate fits varied across frames and mask erosions, and disagreed with the RGB silhouette by **3.48–4.56° RMS**; all were rejected. The tabletop depth must not be mistaken for the ball's surface. Code and rejection criteria are in `inspect_ball.py`.

The supplied board dimensions (200×300×5 mm) and bucket diameters are **manual measurements**. The bucket height interpretation and placement remain provisional. The ball's nominal Ø75 mm is a **simulation assumption**, not a new measurement from this task. No simulated scene coordinates were inserted as physical observations. Joint readbacks and configured frame transforms are **kinematic/configuration evidence**, not calibrated visual localization.

## Printed-board pattern

Historical phone photo `IMG_9304.png` shows a checkerboard with binary square markers, consistent with a **ChArUco-style** board. Offline OpenCV detection found **43 exact-code ArUco 5×5 markers**, with error correction disabled. `DICT_5X5_100`, `_250`, and `_1000` return the same observed IDs because they share a prefix; the precise original dictionary is therefore not uniquely identified.

Marker-center geometry supports a 14-column layout; a 14×9 chess-square board is plausible, but full rows/layout remain unconfirmed. Black tape occludes central markers and may change the impact surface height. These findings use a phone photograph, not the physical RealSense dataset, and establish no metric camera pose. Results are in `fiducial-audit/`; the offline script is `inspect_fiducials.py`.

Exact inputs needed for metric marker/board pose estimation:

1. Chess-square pitch/side in **mm**.
2. Marker side in **mm measured around the outer black border**, excluding its white margin.
3. Chess-square rows/columns, dictionary, marker-ID arrangement, and preferably the original print PDF/generator settings. The photo cannot uniquely choose between dictionary supersets.
4. Printed-grid origin and axes, and grid-to-physical-board edge offsets/orientation.
5. Offset and flatness of the printed face and tape relative to the actual ball-impact surface.

OpenCV's [ChArUco board definition](https://docs.opencv.org/4.12.0/javadoc/org/opencv/objdetect/CharucoBoard.html) requires the square/marker geometry and dictionary; the 200×300 mm panel dimensions alone are insufficient.

## Remaining calibration requirements and next step

Required to turn these observations into reliable metric/localized poses:

- Read the active device's depth scale/units through an appropriate host-side diagnostic, and verify depth against a matte target at measured distances. Record where the distance is measured from; a housing-edge distance is not automatically an optical-center distance.
- Verify the RGB/depth registration on a known target with usable depth and clearly identifiable boundaries/markers. Current ball depth is insufficient.
- Provide the marker measurements above, and identify the actual printed surface normal/origin relative to the physical board.
- Device-label identity is now corroborated: D435 serial 112322070710 matches status. Establish the camera-to-tool transform using known references; phone views of the bracket supply no metric transform. A single stationary view cannot independently solve all camera/tool/world transforms or certify full hand-eye calibration.
- Confirm the real ball diameter if using a future size-constrained estimate. Confirm bucket interior-depth versus overall-height meaning and the opening plane/pose from actual observations.
- Board/bucket must be visible for their camera-frame poses. Accomplishing that may require later, separately planned repositioning; no arm or tool was moved for this task.

**Recommended next perception step:** retain this arm pose and place a matte, dimensioned reference target in the current camera field of view, measure its range/reference geometry, then validate depth scale and registration. Browser point-cloud availability is now established; resolve the SDK transport failure before relying on unattended PCD capture. Only after those checks, use the confirmed printed-board definition and verified transforms for metric board/ball/bucket localization. Any future arm-pose sequence for hand-eye calibration requires its own planned task.

## Runnable tools and stable state

See `CAPTURE_README.md` for the exact tested commands. `capture_stationary.py` performs only camera/resource/joint getters; `inspect_capture.py` is offline depth inspection. The later browser cloud and cross-time inspection are retained separately; `inspect_browser_pointcloud.py` reproduces that inspection offline. Browser evidence: `pointcloud-ui-verification.png`. Optional `inspect_ball.py`, `inspect_alignment.py`, and `inspect_fiducials.py` run with `requirements-fiducials.txt` in a separate perception environment. OpenCV was installed into a temporary isolated environment, not into the frozen simulation environment.

Credential example contains placeholders only. Actual authorized physical connection credentials remain in ignored owner-only `.private/`, outside the transfer and dataset. All capture clients exited; no capture loop is left running. No configuration, motion, tool command, simulation policy, or simulation asset was changed by Part B. The native Mac simulator and dashboard remain available; the Ubuntu archive was not modified after readiness was announced.
