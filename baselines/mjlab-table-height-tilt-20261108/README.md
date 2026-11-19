# Fixed-XY board height and tilt study

**58 of 120 proposed configurations passed setup admission; none of the 58
native drops entered or captured the bucket.** This bounded experiment found
setup-admitted combinations but did not find a successful replacement shot.
The independent audit passed **36,371 checks**. No model was trained or promoted.

The four cases come from the earlier [PPO failure diagnosis](../mjlab-table-training-diagnosis-20261108/README.md).
They are posthoc development positions, not a fresh generalization test.
Each case kept its bucket XY, board XY and drop gap fixed while testing:

- Board-center height: **320, 360, 400, 440, 460, 480 mm** above the tabletop.
- Board tilt from horizontal: **16, 20, 24, 28, 32°**, facing the bucket.
- 30 combinations per case; **120 total**, frozen before any new flight.

The 460 mm row was included to cover a narrow candidate region suggested by
advisory wrist geometry. That approximation did not filter proposals or replace
the actual IK, clearance and sampled setup-path checks. Native flights retained
the original 64-substep physics and v3 scoring, including permitted table contacts.

## What ran

| Stage | Configurations/requests | Native drops | Captures |
| --- | ---: | ---: | ---: |
| Height/tilt discovery | 120 | 58 | 0 |
| Original exact-branch baselines, three per case | 12 | 12 | 0 |
| Controls before and after | 18 | 18 | 12 |
| Selected-success repeats | 0 | 0 | 0 |
| Total | **150** | **88** | **12, all controls** |

The 62 declined configurations remain in the 120-proposal denominator. Their
flights are unobserved, not simulated misses. Every admitted discovery candidate
was flown once; none was left waiting. The selection rule required a discovery
capture, so no candidate qualified for the optional three fresh confirmation
drops. There were no physics-invalid native shots.

Both control gates passed: six direct captures, six captures with table contact
and six expected misses. CPU admission took **101.2 seconds**; the native phase
took **218.9 seconds** of worker wall time. These are not isolated GPU kernel
timings. The fixed budget allowed at most 162 requests and one hour per CPU/native
phase; the study completed without exhausting those caps.

## Results by case

| Case | Setup admitted / 30 | Board bounces | Whole-ball entries | Captures | Robot-contact drops |
| --- | ---: | ---: | ---: | ---: | ---: |
| 00 | 13 | 13 | 0 | 0 | 9/13 |
| 01 | 14 | 14 | 0 | 0 | 2/14 |
| 18 | 17 | 17 | 0 | 0 | 17/17 |
| 26 | 14 | 14 | 0 | 0 | 14/14 |
| Total | **58** | **58** | **0** | **0** | **42/58** |

All 58 balls bounced from the board, but none achieved whole-ball entry or
retention. These were not successful captures rejected only because of the route.
The 42 robot-contact drops account for all discovery shots with forbidden contact;
the other 16 also missed. Contact presence alone does not establish why an
individual shot missed or identify the responsible robot surface.

## Concrete admitted configurations

These are setup-admitted examples, **not successful shot recipes**. Each listed
example missed and recorded robot contact.

| Case | Board height | Tilt | Fixed drop gap | Native result |
| --- | ---: | ---: | ---: | --- |
| 00 | 460 mm | 28° | 251.819 mm | No entry |
| 01 | 440 and 460 mm | 28° | 239.833 mm | Neither entered |
| 18 | 460 and 480 mm | 28° | 240.200 mm | Neither entered |
| 26 | 480 mm | 24° | 242.833 mm | No entry |

Case 00's only admitted 28° grid point was 460 mm. Case 01 admitted 28° at 440/460 mm;
case 18 at 460/480 mm. Case 26 admitted 24° at 440/460/480 mm but no 28° combination.
The exact XY coordinates, all other tested parameters and their outcomes are in
[grid-results.csv](grid-results.csv).

All 24 tested 32° configurations failed finite IK admission. All 62 setup declines
were `finite_ik_unresolved`: 52 lacked a board-arm branch only, eight lacked a
dropper branch only, and two lacked both. At 480 mm, cases 00/01 lacked dropper
branches at every tested tilt. Raising the board can accommodate a larger tilt,
but with the gap fixed it also raises the dropper target. This maps the tested
finite-search outcomes; it is not proof that an unadmitted pose is impossible.
No solver budget, clearance threshold, orientation convention or exact-action
guard was relaxed.

## Next step and limits

Before more PPO, inspect the saved ball trajectories against the robot geometry
to identify the ball/robot clearance problem, then freeze one bounded change and
test admitted alternatives with the same controls. More training on these misses
does not by itself establish a solution.

This study does not cover new bucket positions, varying drop gap or board XY,
alternative board spin, moving-arm execution, noise, timestep convergence,
perception or physical transfer. The earlier 15/50 policy result remains unchanged.

## Recorded video and contact localization

A **47.63-second replay** is available in the interface at `#heightTiltReplay`:
three native misses followed by a successful table-assisted reference, at normal
and quarter speed. It renders saved ball samples and held arm joints using the
original collision geometry. There are no new physics steps or invented arm
movements. The successful reference is a control, not a discovery-grid capture.

CPU geometry reconstruction corroborates first robot contacts with the bouncer's
link 2 (case 00 g23, 0.476 s) and link 3 (case 18 g23, 0.468 s). The original
force log identifies the robot category; the named links are reconstructed.
Case 01 g17 avoids robot contact but hits the bucket's exterior below the rim at
0.596 s. See [video interpretation and reproduction](../../docs/MJLAB_HEIGHT_TILT_REPLAY_20261108.md),
[contact evidence](video-contact-evidence.json) and [video manifest summary](video-summary.json).
The MP4 and full per-frame map remain local runtime artifacts, outside Git.

## Compact evidence

- [summary.json](summary.json): independently audited counts, all 120 cells,
  per-case results and exact report/audit/contract/matrix/row hashes.
- [grid-results.csv](grid-results.csv): one derived row for every proposal;
  explicit setup-declined/not-flown status, native results where observed and
  record/admission proof hashes. Blank flight metrics mean no observed flight.
- [Frozen experiment protocol](../../sircus/mjlab/TABLE_HEIGHT_TILT.md) and
  [independent checker](../../sircus/mjlab/height_tilt_verify.py).

Frozen producer: `380ac3e73b8d6b08504a31ad3797e5088b78af4e`.
The compact package contains no raw trajectories, checkpoints or host paths.
Full native evidence remains external. The existing 4070 workload and Isaac setup
were left untouched; no services were restarted.
