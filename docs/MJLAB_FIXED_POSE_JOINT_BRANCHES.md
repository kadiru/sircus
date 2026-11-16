# Fixed-pose joint search at the remaining bucket position

The bucket stayed at (1028.9, −19.0) mm. The four previously tested commands
each bounced the ball off the board and then hit a robot. [Prior study](MJLAB_SETUP_DROP_RECOVERY.md).

The saved native force trace records when the robot hit occurred. For each
impact, CPU replay places the held arms at their saved joint angles and
computes the signed gap between the ball and each compiled robot collider.
This identifies the nearest link at the first force sample and at peak force.
The saved robot force channel combines contacts across both arms.

| Archived shot | Closest link at impact | Impact after release |
|---|---|---:|
| Direct proposal 12 | Board arm link 4 | 480.924 ms |
| Table proposal 12 | Board arm link 3 | 514.518 ms |
| Table proposal 14 | Board arm link 3 | 488.737 ms |
| Table proposal 15 | Board arm link 3 | 400.130 ms |

At the first force sample the named link was already in contact. The next
closest robot collider had 7.7–37.1 mm of clearance depending on the shot.
The same link ranked closest at peak force.

The joint search holds all four board and release targets fixed, including
the board orientation and gripper orientation. It checks the existing
motion and endpoint constraints before a candidate can enter the drop
test. The 64-substep test replays the exact old commands alongside the
accepted alternatives. Every first-pass capture receives three fresh repeats
with the same joint angles. The same nine controls run before and after the
study. The CPU audit checks the pose identity and rescores each native
trajectory.
