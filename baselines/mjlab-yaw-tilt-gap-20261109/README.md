# Farther shot: fixed gripper yaw, tilt/drop-gap results

**Ten configurations captured all three fresh repeats** at a fixed horizontal
board-to-bucket distance of **562.434 mm**. The unchanged bucket-facing
baseline captured **0/3**. No discovery or confirmation drop contacted a robot.
This recovers one inspected development layout; no model was retrained.

## Successful settings

All rows use board center **(653.661, −203.412, 449.818) mm**, bucket center X/Y
**(1107.027, −536.266) mm**, and dropper world yaw **−36.285503°**. Board tilt and
drop gap are the only changed shot parameters; arm joints change as required
by strict IK. The ball radius is 37.5 mm.

| Grid | Tilt from horizontal | Drop gap | Ball-center release height | Fresh captures |
| --- | ---: | ---: | ---: | ---: |
| 02 | 24.0000° | 115.650 mm | 602.968 mm | 3/3 |
| 03 | 24.0000° | 135.650 mm | 622.968 mm | 3/3 |
| 08 | 26.0000° | 95.650 mm | 582.968 mm | 3/3 |
| 09 | 26.0000° | 115.650 mm | 602.968 mm | 3/3 |
| 15 | 28.0000° | 95.650 mm | 582.968 mm | 3/3 |
| 16 | 28.0000° | 115.650 mm | 602.968 mm | 3/3 |
| 22 | 30.3307° | 95.650 mm | 582.968 mm | 3/3 |
| 29 | 32.0000° | 95.650 mm | 582.968 mm | 3/3 |
| 36 | 34.0000° | 95.650 mm | 582.968 mm | 3/3 |
| 43 | 36.0000° | 95.650 mm | 582.968 mm | 3/3 |

Nominal drop gap is ball-bottom height minus board-center height. Ball-center
release height is board height + gap + radius; it is neither flange height nor
clearance normal to the tilted board. Numbers above are commanded settings.
Actual admitted poses were independently checked from native reset joints.

The smallest successful change retaining the original **30.3307° tilt** was
**+20 mm gap**, from 75.650 to 95.650 mm: baseline 0/3 → changed 3/3.
This does not prove every intermediate gap or tilt succeeds. All ten settings
first captured in discovery, then captured three additional nominal repeats.

## Complete accounting

| Group | Requests | Native releases | Captures | Setup declines | Robot-contact shots |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen 7 × 7 discovery grid | 49 | 35 | 10 | 14 | 0 |
| Ten selected recipes × 3 fresh repeats | 30 | 30 | 30 | 0 | 0 |
| Unchanged farther rotated baseline | 3 | 3 | 0 | 0 | 0 |
| Preserved farther successful reference | 3 | 3 | 3 | 0 | 0 |
| Preserved nearer successful yaw recipe | 3 | 3 | 3 | 0 | 0 |
| Original pre/post controls | 18 | 18 | 12 | 0 | 6 intentional negatives |
| Total | 106 | 92 | 58 | 14 | 6 |

All **14 declines were finite IK failures** at the two largest tested gaps,
195.650 and 235.650 mm, across all seven tilts. No declined setup has a fabricated
flight; the finite solver result does not prove physical unreachability.
There were **zero invalid-physics trials**. The 18 control requests passed their
original gates; both successful reference recipes retained **6/6 captures**.

All 40 new discovery/confirmation captures recorded table contact and satisfy
the unchanged v3 criterion: board contact, whole-ball entry and one continuous
second inside. Table/rim assistance is allowed; robot/floor/unattributed contact
invalidates capture. Table-contact classification alone does not establish a
causal bounce. For the unchanged-tilt/+20 mm recipe's first repeat, actual saved
samples show board contact at **0.1330 s**, table contact at **0.4777 s**, bucket
contact at **0.7208 s**, whole-ball entry at **0.8770 s**, and retention completion
at **1.8770 s**. The table contact precedes bucket entry; the recorded trajectory
is available for inspection.

Independent audit: **50,615 checks passed**. Worker elapsed time:
**2000.63 seconds** (33.34 minutes), including setup work.
This is not a measure of GPU kernel time. Native physics remained at 64 substeps.

## Scope and publication

This is fixed-layout search followed by deterministic nominal repeats. It does
not establish noise robustness, 128-substep consistency, unseen-position
performance, all-table coverage, learned arm transit or physical validation.
Arms remain held at admitted endpoints during flight. No policy weights, shared
IK defaults, scoring, historical results or successful recipes are replaced.
The historical **40/70 current versus 38/70 corrected imitation** scores remain.

The additive UI panel is `#yawTiltGap`; its results and video passed visual,
scoped-DOM and browser-playback checks. The **55.8-second, 1,674-frame** movie
was rendered with CPU Mesa llvmpipe and fully decoded. It replays saved native
states with no added physics or trajectory interpolation. Seventy-two preview
frames were generated; representative board/table/bucket views were inspected.
The movie shows reference/baseline repeat zero and the first three grid-ordered
selected recipes at fresh repeat zero, independent of repeat outcomes.
[Video provenance](video.json) binds the clips, native archives and MP4 hash.
Media and raw trajectories remain outside Git.

All 11 inventoried files stayed unchanged outside this panel's HTML/CSS blocks;
the viewer retained its process identity and the 4090 study worker exited.
Only the local SSH tunnel was restored. The 4070 fountain retained its original
process through the post-study check; a later final check observed the same
fountain command under a new PID. This task issued no stop/restart command or
4070/Isaac configuration change. The changed external process identity is
recorded, not represented as an unchanged final-state check.

## Provenance

- [Compact audited summary](summary.json), including all 88 non-control requests.
- Frozen producer/UI: `c9d0248743790e862cd8053f1681f4ab24867e31`.
- [Draft PR #116](https://github.com/kadiru/sircus/pull/116), stacked on draft #114.
- Contract: `f0b63420c08265bb17e2a3e5cd40342ca0c492598f55d6e1772bf786e586737e`.
- Report SHA-256: `39847d03be006b64af49da3ab1a6a8cc319883c7b1a8d81f37660e1d67f66a92`.
- Independent audit SHA-256: `918f9d6d5cfa5cad93ba6baec2b5e4f46051c253c8459e7c09cf7396be732746`.
- [Protocol](../../sircus/mjlab/YAW_TILT_GAP.md) and
  [handoff](../../docs/MJLAB_YAW_TILT_GAP_20261109.md).
