# Fixed-grid reach diagnosis and saved-trial videos

**Most rejected setups were outside the arms' reach.** This read-only diagnosis
examines the three remaining searchable cells in `fixed35-six-v1`. It changes
no policy, physics, guard, or recorded result. The existing collection remains
**30/35 confirmed recipes**; the frozen policy remains **24/35**.

## Why only four of 96 settings passed per cell

| Grid cell | Bucket X/Y (mm) | Outside geometric reach | IK unresolved inside bound | Endpoint collision/clearance | Admitted settings |
|---|---|---:|---:|---:|---:|
| 03-04 | 860, 25 | 86 | 0 | 6 | 4 |
| 03-05 | 1050, 25 | 88 | 1 | 3 | 4 |
| 04-06 | 1240, 250 | 89 | 0 | 3 | 4 |
| Total | Three fixed bucket locations | **263** | **1** | **12** | **12** |

These are **288 numeric settings**, each offered with two joint-branch slots:
576 requests, 24 admitted drops. The two slots are not two different numeric
shots. All 24 selected setup paths passed; no setting failed only at path
planning. Neither the 64-pair nor 12-endpoint-candidate cap was reached.

The pinned arm geometry bounds shoulder-to-wrist distance by
`390 + sqrt(150² + 426²) = 841.637 mm`. The requested flange pose determines the
wrist center after removing the final link's axial offset. The check includes
the rounded URDF's 0.000331 mm transverse remainder and a conservative 0.136 mm
allowance for the existing position/orientation tolerance. Joint limits and
obstacles can only reduce this outer reach region.

**263 of 264 IK failures exceed that bound**, even with the allowance. For
example, 03-04/proposal 3 requests a wrist radius of **845.411 mm**. Its best
saved IK solution misses by 3.443 mm and 0.245°. The geometric violation is
3.773 mm before the allowance; this is not a borderline 0.1 mm solver threshold.
The old workspace hint was false for those 263 settings, but was not a veto:
the solver still tried 66 starts per arm, up to 360 iterations each.

The single failure inside the outer bound is 03-05/proposal 79: board height
153.44 mm, tilt 22.828°, gap 165.950 mm. The dropper solves, but the best bouncer
misses by **81.406 mm / 5.578°**, with joint 2 at its upper limit. That is an
unresolved finite IK search, not proof of impossibility.

## Were clearance checks too restrictive?

Of the 12 endpoint-rejected settings, **10** had recorded overlap in every
tested joint pair. **Two** had a branch with positive clearance below the 5 mm
buffer: bouncer links 4/6 were **4.9549 mm** apart at 03-05/proposal 3, and only
**0.1856 mm** apart at proposal 53. The first is a near-threshold case worth
separate inspection; it does not explain most rejections. No buffer was relaxed.

The 277 checked joint pairs comprise 64 endpoint passes, 155 native-collision
rejections and 58 strict-clearance rejections. These are pair counts, not extra
shots. No rejected pair required the unproven-clearance fallback; no independent
full-pose mismatch occurred. Recorded endpoint causes were arm/table,
arm/bucket or arm self-overlap, not the two arms intersecting each other.
Later ball/robot flight contact is a separate failure mode.

The next useful search change is to generate board and release targets within
**both arms' compatible reach regions**, then keep collision and path checks.
This is a recommendation, not a tested improvement. None of these findings
proves the three **bucket locations** impossible; different requested poses
may still work. The two other cells with actual fixed bucket/base overlap
remain a separate problem described in the [first-pass report](MJLAB_FIXED35_SIX_20261110.md).

## Completed videos

The authenticated media inventory contains **100 saved native drops across
four historical cohorts**, not 288 new or rendered shots:

| Cohort | 03-04 drops | 03-05 drops | 04-06 drops | Total |
|---|---:|---:|---:|---:|
| Fixed-grid imitation comparison | 2 | 1 | 0 | 3 |
| Earlier recovery search | 31 | 23 | 4 | 58 |
| Latest 96-setting search | 8 | 8 | 8 | 24 |
| Saved bouncer-joint alternatives | 3 | 5 | 7 | 15 |
| Total | **44** | **37** | **19** | **100** |

These four cohorts contain 837 requests at the three cells: 100 native drops
and 737 setup declines. No whole-ball entry or capture was recorded in this
inventory. The 288-setting rejection diagnosis above covers only the latest
search cohort; its denominator must not be mixed with this cumulative inventory.

The films replay saved ball-center samples with the observed initial arm
joints held fixed. They do not run new physics, invent rejected flights, or
show recorded arm transit. All 737 setup declines are indexed in the companion
JSON receipts. The four-cohort scope includes all 100 recorded drops at these
three cells, including repeated failures. No favorable subset was selected.

| Local media filename | Drops | Frames | Duration |
|---|---:|---:|---:|
| `case-03-04.mp4` | 44 | 2,416 | 100.667 s |
| `case-03-05.mp4` | 37 | 1,917 | 79.875 s |
| `case-04-06.mp4` | 19 | 1,079 | 44.958 s |
| `all-100-trials.mp4` | 100 | 5,412 | 225.500 s |

All videos are H.264, 1280×720, 24 fps, with shot settings and recorded contact
timelines. They include complete original scored horizons at 1× playback.
The combined container duration is 225.500326 s: 0.326 ms of stream-copy mux
rounding, with the same 5,412 frames and strictly increasing timestamps.

The contacts identify another limitation: 98 of 100 drops contacted a robot
during the recorded scored flight (91 before any bucket contact, seven after).
Contact channels do not identify the particular robot link or establish that
every contact caused the failure. Passing setup checks did not guarantee a
clear rebound path.

Rendering used committed source `e05679a6386ba32224afec67855166aed8435162`
and forced Mesa llvmpipe on CPU. Independent verification passed 61,218 ledger
checks and 440 media checks, including full error-free decoding of all four
files. Contact sheets cover every drop; representative frames were visually
inspected. All 127 protected evidence/source/asset files retained their hashes,
and the existing 4070 workload retained its process identity. Neither GPU was
used by rendering. Ten focused tests and the source commit's GitHub CI passed.

The sanitized summary records all media/receipt hashes. Videos and full receipts
remain local artifacts, outside Git. No dashboard update or service restart
was performed.

## Evidence and limits

The CPU diagnosis performed 3,768 checks, including saved best-solution FK
reconstruction and pinned source/URDF hashes. Frozen native producer:
`e6a11db6b0c00c278874d04800b1b38e78080040`; the original native audit passed
170,755 checks. Report SHA-256:
`629360baed3cd23261c125679202b8d129ffb28b9cec0c1c8709d0ef19eeeb32`.

The [sanitized summary](../baselines/mjlab-fixed35-reach-and-trial-video-20261110/summary.json)
binds all four cohort reports, audits, manifests and row ledgers, and records
the exact thresholds and examples. Raw observations, host paths, runtime
configuration, calibration and logs remain outside Git. No new IK search,
physics stepping, training, hardware operation, service change, or 4070/Isaac
work was performed for this diagnosis.
