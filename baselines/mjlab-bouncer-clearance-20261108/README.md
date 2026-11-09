# Bouncer-clearance XY search

**One inspected case now clears the robot, but no new capture was found.**
Case 00's +120 mm Y setting recorded zero robot contacts in 3/3 fresh repeats,
versus contacts in 3/3 exact original repeats. Both settings captured 0/3; the
clear setting still failed whole-ball entry. Cases 18 and 26 remained colliding.

| Stage | Requests/proposals | Native drops | Captures |
| --- | ---: | ---: | ---: |
| Three 5×5 XY grids | 75 | 9 | 0 |
| Original matched repeats | 9 | 9 | 0 |
| Selected case 00 repeats | 3 | 3 | 0 |
| Bracketing controls | 18 | 18 | 12 |
| Total | **105** | **39** | **12, all controls** |

All 75 proposals received setup checks; 66 finite IK searches were unresolved.
They remain unflown setup declines, not observed misses or proofs of impossibility.
Every admitted proposal was flown once at 64 substeps. Seven of nine discovery
shots contacted a robot; all nine bounced from the board, none entered the bucket.
Both control gates passed, with six direct captures, six table-contact captures
and six expected misses. No native physics-invalid rows occurred.

The selected case 00 setting is board center **(597.992, −23.557, 460) mm**,
bucket **(1109.540, −221.464) mm**, tilt **28°**, gap **251.819 mm** and release-ball
center Z **749.319 mm**, in the simulator table frame. It is a **contact-free miss**,
not a successful recipe. The original board Y was −143.557 mm; moving +120 mm
changed its heading and increased horizontal bucket distance from 517.446 to
548.496 mm while preserving height, tilt and gap.

Failures are retained as learning evidence. This posthoc diagnostic did not train
PPO or install a hard refusal filter. Next: search tilt/drop-gap variations around
the cleared case 00 family and investigate new admitted branch/posture choices
for cases 18/26. Successful entry and one-second retention remain the objective.

- [Exact tested settings, timing and interpretation](../../docs/MJLAB_BOUNCER_CLEARANCE_20261108.md).
- [Frozen protocol and reproduction](../../sircus/mjlab/BOUNCER_CLEARANCE.md).
- [summary.json](summary.json): compact audit-bound results and identities.
- [grid-results.csv](grid-results.csv): all 75 proposals with explicit admission and observed flight outcomes.
- UI: `#bouncerClearanceStudy`, separate discovery, matched repeats and controls.
- A 43.4-second recorded native64 replay shows the original/clear Case 00 pair,
  unresolved cases 18/26 and a successful control. It contains actual recorded
  ball centers and held arm poses; no new physics. Five chapters, slow motion
  and MP4 download are available. Video hashes and metadata are in `summary.json`.

The independent audit passed **17,380 checks**. CPU admission took 34.5 seconds;
native phase took 208.3 seconds of worker wall time. Frozen producer:
`a6fe6af7afce0ecfe667bbe7efbfdd9f8f39b437`.

Native64 physics and v3 scoring remained unchanged, including allowed table
contacts and one-second whole-ball retention. This is three previously inspected
layouts, not a generalization result. No new policy was promoted; the 4070 and
Isaac remained untouched. Raw trajectories, runtime configurations and full
movies stay outside Git.
