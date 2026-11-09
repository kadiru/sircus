# Case 00 tilt/drop-gap results — November 9, 2026

**Two configurations captured the ball in all three fresh confirmation drops.**
Both use the cleared Case 00 board position and the original **251.819 mm gap**;
the successful tilts are **26° and 32°**. The original 28° setting captured **0/3**.
These are repeated simulation successes at fixed 64, not a newly trained policy
or a measured success rate for other bucket positions.

## Exact successful settings

Coordinates are the simulator's table frame, in millimetres. Board Z is center
height above the tabletop and tilt is from horizontal. Full precision remains
in the evidence ledger.

| Setting | Board X, Y, Z (mm) | Bucket X, Y (mm) | Tilt | Drop gap | Release-ball center Z | Discovery | Fresh repeats |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| g34 | 597.992, −23.557, 460 | 1109.540, −221.464 | 26° | 251.819 mm | 749.319 mm | 1/1 | **3/3** |
| g64 | 597.992, −23.557, 460 | 1109.540, −221.464 | 32° | 251.819 mm | 749.319 mm | 1/1 | **3/3** |
| Original baseline | 597.992, −23.557, 460 | 1109.540, −221.464 | 28° | 251.819 mm | 749.319 mm | Grid cell missed | **0/3** |

Horizontal board-center to bucket-center distance is **548.496 mm**. The exact
gap is `0.25181883224993984 m`. Both successful settings changed only tilt from
the baseline; their dropper joint branch stayed the same. Changed board poses
passed the unchanged exact-action IK/clearance admission before flight.

All eight successful flights (two discoveries plus six fresh repeats) recorded
bucket/rim contact before entry, no table contact and no robot contact. They are
**assisted or marginal captures**, not clean entries. The scorer's `direct_capture`
flag here means no table contact; it does not mean untouched passage through the
opening. At 26°, fresh repeats entered at 1.311–1.313 s and completed retention at
2.311–2.313 s. At 32°, they first entered at 0.945 s, partially exited, reset the
retention clock, and completed a later continuous second inside at 2.191 s.

## Complete search accounting

The [frozen grid](../sircus/mjlab/CASE00_TILT_GAP.md) tested seven tilts
20/22/24/26/28/30/32° against ten gap offsets
−60/−40/−20/−10/0/+10/+20/+40/+60/+80 mm. Board XYZ, bucket XY and heading stayed
fixed. All 70 proposals received CPU admission; all 42 admitted candidates were
flown once. Both discovered captures were selected before confirmation and each
received three fresh repeats. No successful candidate was omitted.

| Stage | Requested configurations/repeats | Native drops | Captures |
| --- | ---: | ---: | ---: |
| Tilt/gap discovery | 70 | 42 | 2 |
| Exact original baseline | 3 | 3 | 0 |
| Fresh repeats of both captures | 6 | 6 | 6 |
| Direct/table/miss controls before and after | 18 | 18 | 12 |
| Total | **97** | **69** | **20, including 12 controls** |

Discovery produced 40 misses, including 19 robot-contact shots. All 42 bounced from
the board and only the two successful settings achieved whole-ball entry. The
28 setup declines were unflown: seven at gap 271.819 mm failed strict endpoint
clearance or pose checks; 21 at gaps 291.819/311.819/331.819 mm had finite IK searches
unresolved. No guard was relaxed. These are outcomes of the tested finite solver,
not proof that the poses are physically impossible. Every tilt admitted the six
lower tested gaps 191.819/211.819/231.819/241.819/251.819/261.819 mm.

Both control gates passed: six direct captures, six table-contact captures and six
expected misses. No physics-invalid rows occurred. The independent audit passed
**27,547 checks**, with full native verification and no failures. CPU admission
used 69.9 s and the acquired 4090 native phase 258.8 s of wall time; these are not
isolated GPU kernel timings.

## Meaning and next step

This is the first successful continuation of the cleared Case 00 family: moving
the arm out of the rebound path was useful, and a tilt change then produced
retained captures. Misses and collision examples remain in the dataset. Three
nominal repeats per setting do not establish robustness to noise, unseen layouts
or a continuous interval of successful angles/gaps.

Next, map a bounded local success window around 26° and 32° using finer angle/gap
increments, keeping the same scoring and admission checks. Then test a frozen
choice on new bucket positions. This experiment did not run PPO, change a trained
checkpoint, activate a selector or test 128 substeps/noise. Table-assisted success
remains accepted even though these two recipes did not need it.

## Recorded comparison and validation

The UI embeds a **34.83-second, 1280×720, 30 fps** MP4 with three chapter buttons:
the 28° baseline miss, a fresh 26° capture and a fresh 32° capture. Each shows the
complete scored interval at normal speed, then a quarter-speed impact view.
Frames replay saved native ball centers with the recorded held arm joints;
no unrecorded arm transit, new physics or trajectory interpolation is added.
The renderer used CPU Mesa llvmpipe with both GPUs hidden. Full MP4 decoding and
checksum validation passed; media and the complete frame map remain outside Git.
The video SHA-256 is
`b3b163c75b5a8bd0081a68bc7186134379e07feb704a5946485c3e1d1398c67f`.

The focused 66 producer/audit/UI/video checks and full `scripts/test.sh` passed.
The frozen producer also passed GitHub's Mock and simulation checks. Publication
preserved all other HTML panels and display-report keys, without restarting the
existing viewer. The 4070 process and viewer retained their original PIDs/start times.

## Evidence

- [Compact results and grid](../baselines/mjlab-case00-tilt-gap-20261109/README.md); UI anchor `#case00TiltGapStudy`.
- Frozen producer: `71fb34aa8fd35ebfc1cd0d1ee522ad5181a0234d`.
- Draft [PR #100](https://github.com/kadiru/sircus/pull/100), stacked on [PR #98](https://github.com/kadiru/sircus/pull/98).
- Contract fingerprint: `151710711047521544f20624a312882d16ff6ff7a1b6d782fcad784ff7550856`.
- Report SHA-256: `f17a0cfaa661524ea329cdca8f6ceee529af4ce9c38bdf56c7d1ab6ce5c1dadc`.
- Audit SHA-256: `73bf2ff91b2e0824eaaeee10a0e9df84512800f2b4d5435776f96de5dc074a87`.

The 4070 workload and Isaac setup were left untouched. Raw observations and full
movies remain outside Git. Recorded replay must use saved ball samples and held
endpoint joints; it must not invent unrecorded arm transit. This single previously
inspected layout is development evidence, not a new generalization benchmark.
