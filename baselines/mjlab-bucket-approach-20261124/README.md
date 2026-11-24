# Bucket-approach correction — November 24, 2026

**All three previously missed placements captured in 3/3 corrected drops.**
Their unchanged original settings stayed at 0/3. The four previously successful
settings still captured 12/12 repeats. This is a small development experiment
on already inspected positions; the previous 50-position comparison remains
**44/50 for each planner**. No model was trained or promoted.

| Bucket X, Y (mm) | Original | Corrected | Corrected route |
|---|---:|---:|---|
| 616.5, −504.4 | 0/3 | 3/3 | Board → bucket |
| 1179.2, −488.1 | 0/3 | 3/3 | Board → table → bucket |
| 854.6, −408.6 | 0/3 | 3/3 | Board → bucket |

The separate unchanged arm-contact miss at **(1178.8, 216.5) mm** stayed at
0/3, with robot contact in all three drops. It was included as a comparison,
not given a bucket-wall correction. All standard pre/post controls passed.

## What changed

The old forecast evaluated a later descending crossing of the bucket opening.
It could overlook the rising ball hitting the outside wall first. The new
`BucketApproach` check follows the predicted release, board flight, and optional
table rebound against the **compiled bucket collision geometry**, including the
ball radius and a conservative bound between sampled times. Its smallest
clearance joins the opening score when ranking candidate shots.

This is a forecast, not a new success rule. It predicted early wall overlap in
two of the three original misses. For the third, it predicted only **1.8 mm**
clearance despite the actual miss. It also warned about **1.8 mm overlap** on a
previously successful rim-assisted shot. Contact warnings therefore remain
ranking evidence; they do not ban a shot or invalidate a measured capture.
The successful controls were repeated regardless of forecast warnings.

Candidate preparation tried local tilt/drop-gap changes and settings transferred
from the existing training library. Two CPU workers performed **120 setup
checks**, admitting **115 configurations**: 40, 39 and 36 by target. Per-case
preparation took **36.5, 66.2 and 32.4 seconds**. Each command retained the existing
arm-pose and path checks. One correction per target was selected and frozen
before any new native drop; no outcome-dependent search followed.

**Only the first target changed its selected winner because of the new bucket
check.** At the other two, the old ranking would also have selected the winning
correction from the larger candidate menu. The experiment demonstrates three
recovered settings, but does not attribute all three gains to the new check.

## Exact selected geometry

Heights are relative to the table plane; tilt is from horizontal. Drop gap is
the ball-bottom height above the board-center plane. Ball-center release height
adds the **37.5 mm radius**. All simultaneous board XY, heading, twist and joint
changes remain recorded in the frozen preparation and UI details.

| Target | Version | Board X/Y/Z (mm) | Tilt | Gap (mm) | Release Z (mm) |
|---|---|---|---:|---:|---:|
| 04 | Original | 565.2 / −79.2 / 502.6 | 12.14° | 179.8 | 719.9 |
| 04 | Corrected | 706.8 / −374.7 / 291.3 | 27.29° | 75.8 | 404.6 |
| 10 | Original | 635.3 / −237.8 / 433.8 | 31.73° | 133.3 | 604.6 |
| 10 | Corrected | 635.3 / −237.8 / 433.8 | 23.73° | 153.3 | 624.6 |
| 19 | Original | 565.2 / −79.2 / 502.6 | 12.14° | 179.8 | 719.9 |
| 19 | Corrected | 627.0 / −225.7 / 410.0 | 32.11° | 136.1 | 583.6 |

04, 10 and 19 are row IDs in the previous 50-position test, not identifiers from
the older 33-point training grid. Corrected 04 and 10 still contacted the bucket
before full entry and successfully retained the ball. The objective permits
such assistance; the aim is capture, not avoiding every bucket contact.

The first corrected repeat at each target shows why that distinction matters:

- **04:** two board contacts at approximately 0.118 and 0.274 seconds, then
  far-side rim contacts; entry at 0.638 seconds and capture at 1.638 seconds.
  The forecast does not model the second board rebound.
- **10:** the rising ball still contacted the near rim at 0.690 seconds. Its
  center was 24 mm above the rim and moving upward at 0.619 m/s; it subsequently
  entered at 0.911 seconds and captured at 1.911 seconds.
- **19:** entry at 0.454 seconds was followed by a partial exit. The retention
  timer correctly reset, and capture completed at 1.726 seconds.

These observations establish successful retention, not elimination of every
bucket collision or accurate prediction of every contact. They also explain
why the check remains a ranking aid rather than a rule that rejects all contact.

## Execution and evidence

- **51 native64 releases:** nine original wall-miss repeats, nine corrected
  repeats, 12 successful-setting preservation repeats, three unchanged
  arm-contact comparisons and 18 standard controls.
- **419.733 seconds** of acquired RTX 4090 time. The cooperative lease and
  foreign-process checks remained active. The 4070 and Isaac setup were untouched.
- Success retains version 4: board rebound first, whole-ball entry and one
  continuous second inside. Later table/robot/other contacts remain permitted.
  All measured trajectories remain saved for independent rescoring.
- Native arms were held at checked endpoints. This does not validate moving
  arms, perception, noise robustness, hardware or broad table-wide performance.

Computational producer: **`5153f94214e09b030932b8795482d5a456a37846`**.
UI publisher: **`4f21d7491039c7b9d8f89af21d33a39ee2ecbbeb`**.
Draft [PR #168](https://github.com/kadiru/sircus/pull/168).
Published and visually verified in the [local results UI](http://127.0.0.1:18781/#bucketApproachStudy).
The page shows all three comparisons, the unchanged successful settings, the
remaining arm-contact miss, and the forecast limitations. The update preserved
all 17 existing videos and other result sections without restarting the server
or changing the interactive planner. The browser reported no console errors.

The first CPU preparation attempt was stopped by an immutable-proof collision:
the training menu reintroduced the exact original setting whose proof had
already been copied. The corrected producer excludes that duplicate geometry.
The failed v1 evidence is preserved; it performed **zero native drops**. The
completed v2 uses a separate output directory and unchanged physics.

Full evidence is kept outside Git in
`/mnt/robot-training/sircus-mjlab/output/bucket-approach-v2`.
The [frozen protocol](../../sircus/mjlab/BUCKET_APPROACH_STUDY.md) explains the
preparation, freeze, independent preflight, native run and final audit commands.
Use the pinned mjlab environment, original assets and matched-control reference.

The independent preflight passed **1,372 checks**. After execution, the full
read-only audit passed **13,083 checks**, including saved native trajectory
rescoring, exact requests, proof bindings and repeat/control accounting.
[Compact results and frozen provenance](summary.json) accompany this baseline.
[First-repeat contact diagnosis](first-repeat-diagnosis.json) records the six
original/corrected flights, including entry/retention times, contact heights and
velocities, and saved-trace hashes without raw trajectory arrays.

| Artifact | SHA-256 |
|---|---|
| Completed report | `446288079f2b25f41ee70793d3f5d4e25ea3df44d70f3a5fe50685c1eee19283` |
| Evidence manifest | `c2c2b52b8ff485eb0adc84b342a2126987eff6325524e1b13422290a2b854341` |

The full local suite passed with **3,163 Python tests run, 302 skipped**, and
**504 JavaScript tests passed**. Two subsequent focused study/publisher regression
tests also passed. GitHub CI passed for the computational producer
([run 45499326900](https://github.com/kadiru/sircus/actions/runs/45499326900)) and
the UI publisher
([run 45500020403](https://github.com/kadiru/sircus/actions/runs/45500020403)).

## Next

Keep these successful settings and the existing live planner. Test the candidate
expansion and full-approach ranking on fresh bucket placements under a fixed
planning budget before changing the interactive default. Preserve valid
multi-bounce solutions and report forecast false positives as well as captures.
