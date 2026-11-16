# Board tilt sweep at bucket (1028.9, −19.0) mm

The starting shot released the ball 162.7 mm above a board tilted 22.6°.
It cleared both arms and crossed rim height 150 mm short and 5 mm to the
side. The dropper's wrist reach bounds the drop gap at about 204 mm for
this board center. A ballistic estimate from the saved rebound suggested
about 283 mm would be needed if release height alone supplied the extra
range. [Previous arm-clearance trials](MJLAB_FIXED_POSE_JOINT_BRANCHES.md).

The sweep holds the bucket position, board center, board heading, and
board twist fixed. For each tilt of 25°, 28°, and 31°, it searches release
gaps downward in 5 mm increments within the wrist reach bound. It
freezes the highest tested gap with an admissible arm endpoint and
approach path. Native trajectories determine flight clearance and capture.

The necessary wrist reach check ruled out all three requested tilts at the
fixed board center. The board arm would need to reach 3.0 mm farther at
25°, 11.5 mm at 28°, and 19.8 mm at 31°. Changing release height moves
only the dropper and cannot change those board-arm reach requirements.
The fallback tests a higher release at the original 22.6° board tilt.
The exact wrist bound at that tilt is 203.7 mm. The 200 mm release was
the highest point on the 5 mm grid and passed the arm-pose and path checks
with four joint configurations. The first keeps the archived board-arm
joints exactly.

The 4090 study replays the exact starting shot alongside the accepted new
settings at 64 substeps. Every first-pass capture receives three fresh
repeats with the same pose and joint angles. The existing native controls
run before and after. The CPU audit checks the frozen gap selection and
rescores each recorded drop.

## Results

| Board tilt | Drop gap | Simulated drops | Robot contacts | Captures |
|---|---:|---:|---:|---:|
| 22.6° baseline | 162.7 mm | 1 | 0 | 0 |
| 22.6° higher release | 200 mm | 4 | 4 | 0 |
| 25° | — | 0 | — | — |
| 28° | — | 0 | — | — |
| 31° | — | 0 | — | — |

The baseline again crossed rim height 150 mm short. The four higher
releases hit the board arm 492–505 ms after release. Reconstructing
the ball and arm geometry at the first recorded force sample identified
board-arm link 2 in each case. The next closest robot collider was
21.0–46.7 mm away. None qualified for repeats.

The 4090 ran all five study drops and 18 controls at 64 substeps. Both
control sets passed, and the independent CPU audit passed 5,077 checks.
The Mac repository suite passed. The pinned Ubuntu suite ran 2,287 tests
with seven skipped and zero failures.

## Reproduction

Use the pinned mjlab environment with the `intergrid-joint-branch-v1`
and `fixed35-sixhour-v2` archives and their audit receipts. Run
`python -m sircus.mjlab.intergrid_case33_tilt_plan --help` with CUDA
hidden to prepare the requests. Pass the saved preparation to
`python -m sircus.mjlab.intergrid_case33_tilt_study --help` and select
only the 4090 UUID with the existing shared lease. Follow with
`python -m sircus.mjlab.intergrid_case33_tilt_audit --help` on CPU
to verify the preparation and native trajectories.

- Frozen preparation plan SHA-256 `c334819435804fb543be5171cbf558409d9a51090f4fea00f375913d0e5036d1`
- Preparation manifest SHA-256 `6adb292c1be4f341a24a8d35ef9809e6439599cf0ea4e3f514281369c11a06a2`
- Native producer commit `7b99d77`
- Native report SHA-256 `885e35d0705dd149310d52232f9c425798cbad7f86014c9055739b5d14c1bbd8`
- Evidence manifest SHA-256 `7db2c118dd88399a4b28519b2c8976414013e9e756a1cb4f2083acb363d13510`
- Native audit SHA-256 `430b027932d13883c643d1440ddbe7d0422970cf6295d74e274ea691489ca21d`
