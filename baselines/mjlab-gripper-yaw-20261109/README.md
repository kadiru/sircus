# Bucket-facing gripper rotation — audited native64 results

Rotating the dropper gripper passage toward the bucket removed robot contact
from both inspected recipes: **6/6 colliding drops → 0/6**. Accepted captures
improved **0/6 → 3/6**. The nearer layout recovered in all three repeats; the
farther layout cleared the robot but still missed. Successful historical
reference recipes retained **6/6 captures**.

| Case | Board–bucket XY distance | Original corrected capture / robot-contact shots | Bucket-facing capture / robot-contact shots | Reference captures |
| --- | ---: | ---: | ---: | ---: |
| `r0-c6-n1` (farther) | 562.434 mm | 0/3 / 3/3 | 0/3 / 0/3 | 3/3 |
| `r1-c2-n0` (nearer) | 172.610 mm | 0/3 / 3/3 | 3/3 / 0/3 | 3/3 |

The complete context prefix is `corrected-imitation-test-`. Only the original
corrected versus bucket-facing pair isolates commanded orientation; the
successful reference recipes have different shot coordinates.

## Exact changed and preserved settings

| Case | Board center X/Y/Z mm | Board tilt | Drop gap | Ball release-center Z | Original → bucket-facing world yaw |
| --- | --- | ---: | ---: | ---: | ---: |
| `r0-c6-n1` | 653.661 / −203.412 / 449.818 | 30.3307° | 75.650 mm | 562.968 mm | 0° → −36.2855° |
| `r1-c2-n0` | 646.740 / −329.902 / 291.275 | 27.2919° | 75.836 mm | 404.611 mm | 0° → −150.5791° |

All five shot values, bucket coordinates, original board joints and release
centers stayed fixed between each corrected/rotated pair. New strict dropper
IK and sampled setup proofs bound the full rotation explicitly. Native reset
joint FK confirmed target rotation errors below **8.3 × 10⁻⁸ rad** for both
rotated first repeats; board joint deltas and measured board-center deltas were
zero. This is model FK of observed float32 reset joints, not camera sensing.

All **36 requested slots released**, with zero setup declines or invalid
physics. The 18 bracketing controls passed: 12 expected captures and six expected
misses. Independent audit passed **18,330 checks**. Worker elapsed time was
**315.254865 seconds**. The totals include intentional robot-contact negative
controls; use the paired diagnostic counts above to assess the intervention.

Native physics stayed at 64 substeps with unchanged v3 scoring: board bounce,
whole-ball entry and one continuous second retained; table/rim assistance is
allowed and robot/floor contact disallows capture. The three new captures were
direct board-to-bucket routes, which does not imply rim-free entry.

These are two selected development layouts and three nominal repeats per
variant, not a new policy score, all-table coverage or noise qualification.
Both arms are held at admitted endpoints during flight. No moving-arm transit,
learned grasping or physical deployment is validated. Model weights and all
successful reference recipes remain unchanged; historical **40/70 current vs
38/70 corrected imitation** scores are not replaced. The 4070 and Isaac are
outside this experiment; no service restart or hardware command is part of it.

Both [comparison movies](videos.json) are published in `#gripperYaw`: 33.067 s
(farther) and 25.867 s (nearer), 1,768 frames total at 1280×720/30 fps. Every
variant uses its first repeat regardless of outcome. Full decode, source/frame
bindings and Chrome playback passed. CPU llvmpipe rendered actual saved ball
centers and held joints; no new physics or invented arm motion. UI publication
commit `ba79932` retains the archive hash and row identity; native evidence and
movie bytes are unchanged. All seven checked report hashes, all 57 existing
display keys/bytes, unrelated HTML/CSS and protected process identities stayed
unchanged. The 4090 worker exited.

## Provenance

- [Compact audited summary](summary.json); raw traces remain outside Git.
- Frozen producer/UI: `26ba197bc1f85150f6b76f52c27ed0173f9b5c0c`.
- [Draft PR #114](https://github.com/kadiru/sircus/pull/114).
- Contract: `e31c8db6e8b9a2b16f7213b3440396cccf83004e64988575f694b2b007cbf683`.
- Native report SHA-256: `2cd3644a19653efc157fe697557ebc861e40ab6b97ceeb3faeb48aaf04b923fa`.
- Independent proof SHA-256: `4f044f4c3b4dc3ffd79dcb65f42214ceb3f597d129afb0c79728697137b9ec21`.
- [Frozen protocol](../../sircus/mjlab/GRIPPER_YAW.md) and
  [integration handoff](../../docs/MJLAB_GRIPPER_YAW_20261109.md).

The remaining farther-layout miss should be diagnosed with its now-clear ball
path before another intervention or policy update. No additional trial is
implied by these results.
