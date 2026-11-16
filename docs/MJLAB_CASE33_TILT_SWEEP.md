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

The 4090 study replays the exact starting shot alongside the accepted new
settings at 64 substeps. Every first-pass capture receives three fresh
repeats with the same pose and joint angles. The existing native controls
run before and after. The CPU audit checks the frozen gap selection and
rescores each recorded drop.
