# Stationary sfhack10 RGB-D capture

This folder is independent of simulation behavior development. Its scripts read the physical `cam`, `arm`, and `arm-board` resources on sfhack10. They contain no motion, gripper, vacuum, tuning, configuration-write, or DoCommand calls. Capture exits after a small finite sample and closes its client connection; it does not stop the machine or its services.

## Run

Use the existing Mac Python environment from the project root:

```bash
sircus/simulation/.venv/bin/python sircus/perception/capture_stationary.py --credentials sircus/perception/.private/sfhack10.json --frames 3 --skip-pointcloud
sircus/simulation/.venv/bin/python sircus/perception/inspect_capture.py sircus/perception/data/20261030T001623700791Z
sircus/simulation/.venv/bin/python sircus/perception/inspect_browser_pointcloud.py sircus/perception/data/20261030T002549121000Z-browser-pointcloud --compare sircus/perception/data/20261030T001623700791Z
```

The scripts require Python 3.12, viam-sdk 0.85.1, and NumPy. For another machine, create its own environment; do not copy the Mac virtual environment. The credential JSON requires `address`, `api_key_id`, and `api_key`, is owned by the current user, and has mode `0600`. Keep it inside ignored `.private/`; never include it in a transfer. The script is explicitly scoped to the observed sfhack10 main address. A placeholder template is provided separately.

Omit `--skip-pointcloud` to request one point cloud after the image-bracketing arm readbacks. Each RPC is bounded. The optional `--direct` requests authenticated direct TLS without WebRTC, but that transport timed out during this session. It does not disable TLS checks. The SDK's native transport may print network diagnostics even when Python logging is disabled; do not publish those logs wholesale.

## Saved observations

All dataset names are UTC timestamps, while the capture happened on October 29 locally in California.

| Dataset under `data/` | Result |
| --- | --- |
| `20261030T002549121000Z-browser-pointcloud` | Later Viam-browser raw PCD download succeeded: 921,600 records, 796,948 positive finite points. Exposure time unknown; not paired with the preferred RGB-D frames or joint-state bracket. |
| `20261030T001623700791Z` | Preferred stationary sample: 3 paired RGB/depth responses, camera properties, configured frames, both arms' before/after readbacks. All joint differences were exactly zero at returned precision and `is_moving` was false at both observations. Point-cloud request explicitly skipped after the earlier failure. |
| `20261030T001321684082Z` | Initial 3 paired RGB/depth responses and before readbacks succeeded. Point-cloud retrieval failed with `StreamTerminatedError: Connection lost`; subsequent readbacks failed. This attempt does not establish a before/after stationary bracket. |
| `20261030T001436735115Z` | Authenticated direct-TLS connection timed out; metadata only. |
| `20261030T001617665332Z` | Local sandbox denied the transport proxy; metadata only. Authorized run outside that sandbox produced the preferred sample above. |

Each capture retains `metadata.json`, request/response timestamps, relevant errors, and `MANIFEST.json`. Original RGB is JPEG. Original Viam DEP bytes are losslessly gzip-compressed; decoded `*_raw.npz` holds unsigned 16-bit raw counts. Offline inspection produces `inspection.json` and contrast-scaled depth preview PNGs; previews are not metric images. `DERIVED_MANIFEST.json` covers these later inspection products separately from the original capture manifest.

## Verified and unresolved

- RGB and depth capture worked at 1280×720. The preferred sample had 86.8–87.0% nonzero depth globally. Ball-region depth is much less reliable; a global valid percentage must not be used as evidence of reliable ball range.
- `GetProperties` reported `supports_pcd=true`. SDK PCD retrieval failed, but a later **Viam browser raw PCD download succeeded**. Its separate directory records provenance and timing limitations; do not imply unattended SDK PCD capture is working.
- Reported intrinsics: fx=906.84814453125, fy=906.87744140625, cx=651.4014892578125, cy=362.5877990722656; reported distortion is inverse Brown–Conrady with all coefficients zero. These are driver-reported values, not an independent calibration.
- The configured color/depth alignment and matching image sizes do not independently verify RGB/depth registration. A measured target is still needed to check registration and range.
- **Depth values remain device counts.** The exact module's encoder copies the RealSense depth uint16 values without applying `get_units()`. Neither captured camera properties nor the module's status/depth-option getters expose that scale. A later, cross-time comparison with the browser PCD supports 0.001 m/count (96.57% of comparable pixels within 1%). This does not replace known-distance calibration or a direct units query.
- The module copies RealSense XYZ vertices directly into binary PCD, where the RealSense source contract uses meters. Generic Viam millimeter documentation should not override this exact module encoding. The saved PCD median Z is 0.283 m, consistent with the earlier median 283 depth counts; outliers remain.
- Camera properties' identity extrinsics describe the camera's internal color/depth convention. They do not establish the physical tool-to-camera transform or hand-eye calibration. Configured frame transforms are configuration evidence, not measured world calibration.
- RGB/depth share a driver frameset response, but individual exposure timestamps and hardware synchronization were not supplied. Camera/host clock synchronization is unverified. Joint readbacks bracket the images, without hardware exposure synchronization. The point-cloud RPC uses another latest frameset.
- User confirmed that the camera remains attached to arm10's tool bracket, consistent with supplied photos. A later read-only `Camera.get_status()` returned assigned serial **112322070710**, streaming true, recovery mode false; see the preferred dataset's `camera_status_followup.json`. It was not synchronized with the images. Subsequent user photo `IMG_9307` shows D435 serial 112322070710, matching this status; `IMG_9308/9309` corroborate the bracket arrangement. Photo evidence/checksums are retained under `../measurements/photos-9309/`. These photos do not calibrate the camera-to-tool transform. Future runs also save `camera_status.json` through the same tested getter.

For the combined capability/localization assessment and exact remaining measurements, read [STATIONARY_RGBD_REPORT.md](STATIONARY_RGBD_REPORT.md). `inspect_ball.py`, `inspect_alignment.py`, and `inspect_fiducials.py` are offline-only analysis scripts; use the separate dependencies in `requirements-fiducials.txt`, without changing the frozen simulation environment.

## Primary source evidence

Read-only source excerpts are retained in `reference/`, with license and checksums in `SOURCE_PROVENANCE.json`. They were inspected, not executed. The exact module version is [0.22.6, commit 8caec2f4](https://github.com/viam-modules/viam-camera-realsense/tree/8caec2f4b60887e71b7c12d0d27fad0bff5740da).

- `encoding.cpp:12–28`: unchanged uint16 depth encoding.
- `encoding.cpp:142–180`: unchanged RealSense XYZ copied into point data.
- `realsense.hpp:538–548`: read-only status exposes assigned serial/streaming flags, not depth scale.
- `realsense.hpp:450–486`: read-only depth options omit depth units.
- RealSense's [point-cloud deprojection](https://github.com/realsenseai/librealsense/blob/master/src/proc/pointcloud.cpp) multiplies depth counts by `get_units()`; the [depth-frame API](https://github.com/realsenseai/librealsense/blob/master/include/librealsense2/hpp/rs_frame.hpp) defines that scale in meters.

Metric localization and world-frame use require a verified depth scale/known-distance reference, usable target depth, empirical RGB/depth registration check, and a measured or properly calibrated tool-to-camera transform. Nothing here commands either arm to obtain those measurements.
