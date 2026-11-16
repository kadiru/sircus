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

| Archived pose | Alternate shots | Cleared both arms | Captured |
|---|---:|---:|---:|
| Direct proposal 12 | 3 | 0 | 0 |
| Table proposal 12 | 3 | 2 | 0 |
| Table proposal 14 | 7 | 2 | 0 |
| Table proposal 15 | 3 | 0 | 0 |
| Total | 16 | 4 | 0 |

All four exact baselines hit a robot again. Four alternatives cleared both
arms and missed the bucket. One of the clear variants at table proposal 14
kept the original dropper joints and changed board-arm joints 2 and 3 by
12.0° and 22.4°. The board and release targets matched the archived pose.
No shot qualified for repeats. The search found board-arm shoulder or elbow
changes at one of the four poses. Other tested alternatives changed wrist
joints.

All 20 study drops and 18 controls ran at 64 substeps on the 4090. Both
control sets passed. The independent CPU audit passed 7,393 checks. The
Mac repository suite passed. The pinned Ubuntu suite ran 2,287 tests with
seven skipped and zero failures.

Use the pinned Ubuntu mjlab environment and the audited
`intergrid-setup-drop-v1` and `fixed35-sixhour-v2` archives.

1. Run `python -m sircus.mjlab.intergrid_setup_contact_diagnosis --help`
   for the CPU contact reconstruction. Its result passed 75 checks.
2. Run `python -m sircus.mjlab.intergrid_joint_branch_search --help` with
   CUDA hidden. Use the original study and parent audit, reference, and
   assets. The search uses three deterministic seeds with 32 starts per
   arm and seed. It prioritizes changes to board arm shoulder and elbow
   joints while holding the archived dropper joints when feasible.
3. Run `python -m sircus.mjlab.intergrid_joint_branch_study --help` with
   the preparation, source study, and the shared 4090 lease. The same
   native64 runner schedules three repeats per successful first drop.
4. Run `python -m sircus.mjlab.intergrid_joint_branch_audit --help` with
   CUDA hidden. Supply the stored preparation, study sources, and new
   output folder, plus a fresh path for the audit receipt.

The contact reconstruction receipt SHA-256 is
`b5b62961adcc18eeb12ecef3ec88a644accf05f8401c8eeeceb9f55633ca68ce`.

- Native producer commit `4875eb4`
- Frozen branch plan SHA-256 `f998b5a4cd7d409c70cd08d249a09fa824735c6bc5484189ee013d47f97e7393`
- Native report SHA-256 `a59f58475fad480589b4de6c0de7d66d416ad5f03709d0a4a28c540e6e112660`
- Native evidence manifest SHA-256 `c2e27ab3b0746d4fe1259a970a955e7c366dbfc6a038be29d6644509f68b5fdb`
- Independent audit SHA-256 `c59380b20065798f815204608845f2671ee3d2a368a62b08cacc6d07f172e029`
