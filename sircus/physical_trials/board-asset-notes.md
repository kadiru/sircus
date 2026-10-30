# Physical board representation — sfhack10

Prepared and installed 2026-10-29. The main task saved and reloaded the sfhack10 configuration, confirming that only the passive board component was appended. Runtime frame/pose/box checks passed, and the representation was verified in Viam 3D Scene. Evidence: `evidence/board-asset/runtime.json` and `evidence/board-asset/viam-board.png`.

## Asset and attachment

`assets/board.json` keeps provenance outside the Viam component definition. Add only its `viam_component` object to the existing sfhack10 configuration. Do not replace the full machine configuration with this asset file.

| Property | Value | Evidence / status |
|---|---|---|
| Resource / frame | `sircus-board-provisional` | New passive generic component |
| API / model | `rdk:component:generic` / `rdk:builtin:fake` | Supported built-in primitive pattern |
| Parent | `vacuum-board` | Existing board-arm tool frame, under `arm-board` |
| Board local dimensions | X=200, Y=300, Z=5 mm | User measured; `sircus/measurements.json` |
| Mount rotation | +90 degrees about tool +Z | User requested landscape; sign follows simulation convention |
| Mount translation | (0, 0, 122) mm | Provisional, derived from configured tool proxy, not measured |
| Impact surface in board frame | z=+2.5 mm; normal +Z | Nominal outward-face convention |
| Nominal impact surface in tool frame | z=124.5 mm | Derived from the same uncalibrated attachment |
| Mass and actual suction offset | Unknown | Still require measurement |

Board-local X/Y remain 200/300 mm. Applying +90 degrees rotates the footprint to 300/200 mm in the tool's axes. Swapping the dimensions too would undo the intended landscape correction. The photo `sircus/presentation-real-photos/assets/board-bucket.jpg` supports the mounted-board arrangement but cannot establish metric flange-to-board offsets or calibrated axis alignment.

The configured tool proxy was inspected in `sircus/perception/data/20261030T001623700791Z/configured_frames.json`, selecting only the `vacuum-board` entry. Both model links have zero translation and identity orientation. Its main box is centered at z=58.5 mm with total depth 117 mm. The final tube proxy is centered at z=117 mm with total depth 5 mm. Therefore its modeled front is 119.5 mm, and placing the board back face there gives a board center of 122 mm. The configured `vacuum-board` frame itself is identity relative to `arm-board`. **These are model-derived values, not AS1200 physical measurements.** Centered X/Y, contact normal, and mounting rigidity are assumptions.

## Supported deployment and planning semantics

A new built-in generic/fake component with `frame.geometry` is the documented pattern for one primitive without adding a module. It does not implement an arm or physical gripper. Keep the existing arm, vacuum, driver-module, camera, and motion settings unchanged when adding it. No motion or actuation is necessary to configure or inspect it. [Viam workspace obstacle configuration](https://docs.viam.com/motion-planning/obstacles/configure-workspace-obstacles/)

This choice makes the board visible and also adds its geometry to the frame-system motion planner. It is **not display-only**. Because the attachment is provisional, its visual alignment does not certify physical clearance. No collision exemptions are supplied. World-state-store visualization transforms, by comparison, are display-only unless geometry is separately provided to planning. [Viam visuals and collisions](https://docs.viam.com/visualization/visuals-and-collisions/)

The SDK's WorldStateStore API has `list_uuids`, `get_transform`, and `stream_transform_changes`; there is no standard public add/upsert transform method. `do_command` is model-specific, so sending a guessed write command to `builtin` is not supported. A strictly display-only board could instead be published by a dedicated WorldStateStore module, but that is a separate runtime addition. [WorldStateStore API](https://docs.viam.com/reference/apis/services/world-state-store/), [publishing visuals from a module](https://docs.viam.com/visualization/publish-visuals-from-a-module/)

## Read-only verification

The saved config and runtime frame system were inspected after installation. These SDK calls can repeat the transform checks without commanding motion:

```python
from viam.proto.common import Pose, PoseInFrame

# `robot` is an authenticated RobotClient already connected to sfhack10.
frames = await robot.get_frame_system_config()
board_entries = [entry for entry in frames
                 if entry.frame.reference_frame == 'sircus-board-provisional']
assert len(board_entries) == 1
entry = board_entries[0]
assert entry.frame.pose_in_observer_frame.reference_frame == 'vacuum-board'

# This is a kinematic prediction, not a perception-calibrated world pose.
board_center_world = await robot.transform_pose(
    PoseInFrame(reference_frame='sircus-board-provisional', pose=Pose(o_z=1)),
    destination='world',
)
```

Also inspect the runtime geometry dimensions and confirm the new frame appears under the tool in Viam 3D Scene. A static screenshot alone cannot prove that it follows arm state; parent-frame readback establishes the kinematic attachment without moving the arm.

Completed: JSON parses; the component contains the 200×300×5 mm box, correct parent and +90-degree rotation; tool-model proxy and identity links were inspected; installed Python SDK signatures for both read-only methods were verified. The main task then confirmed the exact appended config, runtime parent/pose/box and native 3D display. Both arms reported stationary with zero returned joint deltas during verification. A brief 3D Scene pose-update warning resolved. No movement was used to test the representation, and no arm/tool commands were issued. The asset-preparation subtask itself made no live changes.

## Perception integration

Keep an eventual `sircus-board-observed` transform separate from this nominal asset. Include its source camera, capture time, estimated surface normal, measurement uncertainty, and calibration identity. A camera-frame observation may be displayed as such before camera-to-world calibration; it must not silently become a calibrated world pose. The current geometric box describes board shape only; it does not model the compliant board/suction bounce response.

Measurements still needed: actual tool-frame-to-board center translation, suction contact position on the board, board normal relative to the tool, and any camera-to-tool transform used for world localization. If printed patterns are used for pose estimation, record their dictionary/pattern identity and actual marker or grid dimensions separately.
