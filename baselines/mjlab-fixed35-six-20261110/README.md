# Fixed-grid alternative poses — completed first pass

**One new recipe passed 3/3 fresh repeats; all 29 prior recipes captured again.**
The collection now covers **30/35 fixed grid positions**. The historical policy
remains **24/35**: no training, promotion or UI publication occurred.

| Cell | Bucket X/Y (mm) | Search requests | Admitted drops | Captures | Result |
|---|---|---:|---:|---:|---|
| 01-05 | 1050, −425 | 15 | 13 | 1 | New recipe confirmed 3/3 |
| 03-04 | 860, 25 | 192 | 8 | 0 | Finite search exhausted |
| 03-05 | 1050, 25 | 192 | 8 | 0 | Finite search exhausted |
| 04-06 | 1240, 250 | 192 | 8 | 0 | Finite search exhausted |
| 03-00 | 100, 25 | 0 | 0 | 0 | Actual fixed-base overlap |
| 03-06 | 1240, 25 | 0 | 0 | 0 | Actual fixed-base overlap |

The 591 search requests contain **296 distinct numeric proposals**, with up to
two joint branches each. Only 19 numeric proposals admitted any flight. Each
unresolved searched cell has 184 setup declines: 03-04 has 172 finite IK declines
and 12 endpoint-clearance/pose declines; the other two have 178 and six each.
These finite results do not prove that those three positions are impossible.

## New saved recipe

Tabletop Z=0; values are synthetic simulation commands, not hardware calibration.

| Quantity | Value |
|---|---:|
| Bucket X/Y | 1050, −425 mm |
| Board center X/Y | 563.185, −61.757 mm |
| Board center height | 497.609 mm |
| Tilt from horizontal | 22.330° |
| Drop gap | 171.575 mm |
| Released ball-center height | 706.684 mm |
| Board–bucket horizontal distance | 607.400 mm |
| Fresh confirmation captures | 3/3 |

The gap excludes the 37.5 mm ball radius. Exact unrounded commands and joint
vectors are in [summary.json](summary.json). This is a table-assisted,
rim-contacting capture, not a clean direct shot. The discovery records board
contact at 0.183919 s, table at 0.594922 s, bucket at 0.924349 s, whole-ball entry
at 0.987760 s and one-second retention at 1.987760 s, with no robot/floor contact.

## What this pass did and did not explore

All three remaining cases have four admitted numeric poses and two selected
joint branches per pose. **Those selected branches changed the dropper wrist
configuration but kept identical bouncer joints.** The first-two-path selection
rule omitted 18 distinct endpoint-valid bouncer alternatives across 11 poses.
Seven omitted variants across four poses change shoulder/elbow configuration;
the others principally change wrist configuration. They were not path-tested
or flown in this pass. A separate targeted follow-up will preserve these results
and test those alternatives; no capture improvement is assumed.

Not every recorded robot contact caused the initial miss. Of the eight flights
at 03-04, six contact the robot before bucket/table and two contact it later.
At 03-05 all eight have early robot contact. At 04-06, four have early contact,
two later contact and two none. No whole-ball entry occurred in these 24 flights.
The concrete saved examples and event ordering are retained in the summary.

The actual fixed colliders at 03-00 and 03-06 have deepest validated pair signed
distances −71.035 and −73.579 mm, with 24 common-interior certificates each.
These are fixed bucket/base overlaps, not elbow-placement problems. The clear
control is +47.423 mm. Geometry, bucket positions and guards were not changed.

## Verification and scope

- 641 total requests: 591 search, three confirmations, 29 preservation replays
  and 18 original controls. 87 native releases; zero invalid-physics attempts.
- Both control gates passed: direct 3/3, table-assisted 3/3 and negative 0/3,
  before and after. All 29 saved action/joint/connection identities stayed exact.
- Independent audit: **170,755 checks passed**. Worker 1,704.930 s; pipeline
  1,724.412 s; CPU audit 16.227 s. All candidate caps completed; no time truncation.
- [Isolation](isolation.json): 112 protected UI/evidence/checkpoint/Isaac files
  unchanged, service inventory unchanged, both GPUs free of compute processes.
- Native64/v3 retained: board bounce, whole-ball entry and one continuous second
  inside; table/rim assistance accepted. No noise, 128-substep, dynamic arm
  transit, grasp, perception or hardware validation is claimed.

Frozen producer: `e6a11db6b0c00c278874d04800b1b38e78080040`.
[Draft PR #123](https://github.com/kadiru/sircus/pull/123), stacked on #120.
Full local suite passed (2,290 Python tests, 106 skipped; 390 JavaScript tests),
final 56 focused Ubuntu CPU tests passed, and source CI passed. Raw trajectories,
logs, full audit and runtime configuration remain local. Portable hashes and
aggregate results are in the summary; original evidence remains immutable.

[Protocol](../../sircus/mjlab/FIXED35_SIX.md) ·
[Workflow and handoff](../../docs/MJLAB_FIXED35_SIX_20261110.md)
