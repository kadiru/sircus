# Case 00 tilt/drop-gap search

**Two settings captured 3/3 fresh repeat shots each.** Both hold the board at
**(597.992, −23.557, 460) mm**, bucket at **(1109.540, −221.464) mm**, and drop gap at
**251.819 mm** in the simulator table frame. Only tilt differs:

| Setting | Tilt | Release-ball center height | Discovery capture | Fresh repeat captures |
| --- | ---: | ---: | ---: | ---: |
| g34 | 26° | 749.319 mm | 1/1 | **3/3** |
| g64 | 32° | 749.319 mm | 1/1 | **3/3** |
| Original baseline | 28° | 749.319 mm | 0/1 | **0/3** |

All eight successful flights had bucket/rim assistance, no table contact and no
robot contact. These were not clean entries. At 32°, partial exit after first entry
reset retention; each shot subsequently stayed fully inside for a continuous
second. Table-assisted captures remain valid under the unchanged v3 rules.

| Stage | Configurations/requests | Native drops | Captures |
| --- | ---: | ---: | ---: |
| Tilt/gap discovery | 70 | 42 | 2 |
| Original baseline | 3 | 3 | 0 |
| All discovered captures, three fresh repeats each | 6 | 6 | 6 |
| Bracketing controls | 18 | 18 | 12 |
| Total | **97** | **69** | **20, including 12 controls** |

All 70 proposals were checked and every admitted candidate flew at 64 substeps.
The 28 declines remain unflown setup results: 21 finite IK failures and seven strict
endpoint clearance/pose failures. They do not prove impossibility. Discovery
included 40 misses and 19 robot-contact shots; all failures remain evidence.
Both control gates passed and there were no physics-invalid rows.

- [Exact settings and interpretation](../../docs/MJLAB_CASE00_TILT_GAP_20261109.md).
- [Frozen protocol and reproduction](../../sircus/mjlab/CASE00_TILT_GAP.md).
- [summary.json](summary.json): compact audit-bound results and identities.
- [grid-results.csv](grid-results.csv): all 70 proposals, admission and observed outcomes.
- UI: `#case00TiltGapStudy`, with discovery and fresh repeats shown separately.
- Recorded replay: 34.83 seconds, baseline miss then the two captured fresh repeats;
  playback chapters and MP4 download are embedded in the UI. The compact summary
  binds media hashes to the audited study; binary media remains outside Git.

Independent audit: **27,547 checks passed**. CPU admission 69.9 s; native phase 258.8 s
of wall time. Frozen producer `71fb34aa8fd35ebfc1cd0d1ee522ad5181a0234d`.

No PPO, model promotion, noise, 128-substep test or new generalization evaluation.
Next: map finer local tilt/gap success windows, then assess a frozen choice on
new bucket positions. Existing 4070/Isaac work stayed untouched; raw observations
and full media stay outside Git.
