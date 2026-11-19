# Completed seeded bucket-distance study — November 3, 2026

**Two selected configurations passed the all-resolution repeat gate:** original-ray
450 and 472.826 mm, each **33/33 captures at 32, 64 and 128 substeps**. Original-ray
550 mm passed the prespecified fine-pair gate but failed the 32-step fresh-repeat
threshold. No qualified direct capture was found at 600/640 mm on the original ray
or 650–770 mm on the corner ray in this finite search.

This is DEVELOPMENT parameter search and engineering repeatability, **not new RL,
held-out bucket-position generalization, or a guarantee of future success**. The
[height/tilt parent](../mjlab-height-tilt-20261102/README.md), older rectangular/bucket
results and existing policy/solver routing remain unchanged.

## Executed study and independent verification

- [Prespecified protocol](../../sircus/mjlab/DISTANCE_STUDY.md), frozen execution
  commit `9ed20bc7abf167e69c1338bf38678f1ff8900241`.
- Native 4090 run completed in **2,389 s (39.8 min)**: **3,891 requested attempts,
  1,387 native releases, 589 strict-route captures**. All captures were rim-assisted;
  clean captures: zero. The adaptive total is not a policy capture-rate evaluation.
- 2,504 geometry rejections remain in the full denominator: 2,036 workspace-prior,
  60 strict-clearance and 408 declared table-footprint failures. 124/208 board
  contexts were admitted. Board admission, paired endpoints and flight success
  are distinct checks.
- All 24 exact seed controls captured; eight source joint pairs were preserved.
  1,904 coarse +1,344 conditional far-extension +124 refinement requests preceded
  495 paired repeat requests (five recipes ×33 requests ×three resolutions).
- **9,962 independent audit checks passed**, including scalar rescoring of all
  1,387 native releases; source/assets/packages/options, exact requests, recipe
  selection, cache associations and pairing were verified. No audit failures.
- [Summary](completed/summary.json), [exact recipe CSV](completed/recipes.csv),
  [all targets CSV](completed/target-search.csv), [provenance and input hashes](completed/provenance.json),
  [curated scene geometry](completed/scene-geometry.json) and [checksums](completed/SHA256SUMS).
  Raw trajectories, joints, queues and logs remain outside Git.

## Exact selected configurations

All heights are above table top. Board height is its impact-surface center;
release height is the ball center. Tilt is from horizontal. Board XY stays
470.875779 / −277.893401 mm and its heading faces each bucket. The two rays have
different headings and are reported separately.

| Recipe / ray | Distance mm | Board H mm | Tilt ° | Gap mm | Release H mm | Captures /33 at 32 /64 /128 | Gate |
|---|---:|---:|---:|---:|---:|---|---|
| recipe-00 · original | 350.000 | 460.000 | 36.00000 | 205.781 | 703.281 | 31 / 29 / 24 | Failed |
| recipe-01 · original | 450.000 | 397.388 | 36.15479 | 311.219 | 746.107 | 33 / 33 / 33 | All resolutions |
| recipe-02 · original | 472.826 | 397.388 | 36.15479 | 321.219 | 756.107 | 33 / 33 / 33 | All resolutions |
| recipe-03 · original | 550.000 | 377.388 | 32.15479 | 371.219 | 786.107 | 29 / 33 / 33 | 64/128 only |
| recipe-04 · corner | 550.000 | 377.388 | 32.15479 | 381.219 | 796.107 | 8 / 9 / 16 | Failed |

Each recipe has three nominal plus 30 fresh jointly perturbed H/tilt/gap
requests: sigma 1 mm /0.25° /2 mm, seed 20261206, no clipping or retries. Identical
requests and joint arrays are reused across resolutions; 99 requests per recipe
are not 99 independent randomized layouts. Across five recipes, 165 paired repeat
IDs represent 155 distinct requested scene/action vectors: five nominal plus
150 fresh variations. The two strongest recipes each had
3/3 nominal and 30/30 fresh at every resolution. Finite perfect counts do not
establish a continuous successful interval or statistical certainty.

The fine-pair gate requires 3/3 nominal and ≥27/30 fresh captures at both 64/128,
≥30/33 outcome agreement and ≤5 mm nominal position difference at 350 ms. The
all-resolution gate additionally requires the same repeat thresholds at 32; it
makes no 32-step positional-convergence claim. Recipe03 had 26/30 fresh captures
at 32 versus30/30 at 64/128. Recipe00 failed fine-pair agreement; recipe04 was
unreliable even nominally. Small early-position differences alone do not ensure
matching late rim/retention outcomes.

## All requested distances

Search counts exclude controls and repeats and retain rejected requests.

| Ray / distance mm | Outer-footprint constraint | Search requests | Native releases | Captures | All-resolution qualified recipes |
|---|---|---:|---:|---:|---:|
| original · 350.000 | Admitted | 168 | 84 | 24 | 0 |
| original · 450.000 | Admitted | 168 | 81 | 62 | 1 |
| original · 472.826 | Admitted | 168 | 80 | 51 | 1 |
| original · 550.000 | Admitted | 152 | 65 | 9 | 0 |
| original · 600.000 | Admitted | 136 | 52 | 0 | 0 |
| original · 640.000 | Admitted | 136 | 52 | 0 | 0 |
| original · 650.000 | Below 5 mm margin | 136 | 0 | 0 | 0 |
| corner · 550.000 | Admitted | 148 | 62 | 9 | 0 |
| corner · 650.000 | Admitted | 472 | 98 | 0 | 0 |
| corner · 700.000 | Admitted | 472 | 98 | 0 | 0 |
| corner · 750.000 | Admitted | 472 | 98 | 0 | 0 |
| corner · 770.000 | Admitted | 472 | 98 | 0 | 0 |
| corner · 800.000 | Outside table | 136 | 0 | 0 | 0 |
| corner · 950.000 | Outside table | 136 | 0 | 0 | 0 |

The table's full outer bucket footprint plus 5 mm edge margin permits about 648.8 mm
on the original ray and 772.8 mm toward the supported corner. Original 650 mm has
4.076 mm clearance, failing the declared margin by 0.924 mm; it is not physically
outside the tabletop. Corner 800/950 mm are outside. None was clipped, repositioned,
or made valid by enlarging the table.

## Why the far shots failed

[Post-hoc contact diagnostics](completed/far-diagnostics.json) independently
rescore all 392 admitted native flights across corner650/700/750/770 mm. The first
external contact is the **table in every case**. Some touch the bucket before
that table contact; every assisted retained result contacts the table before
whole-ball entry. Across those 392 flights, 81 entered and 70 met the historical
retention condition after assistance. These are not qualified board-to-bucket captures even when
subsequent retention reaches one second.

The best coarse first descents to rim-center height were roughly 138–235 mm short
of the bucket center. This diagnostic interpolates 240 Hz saved samples before
any nonboard contact; it is not a substitute for whole-sphere entry/rim/retention
scoring. Initial gripper contact was not the cause. Recorded runtime bucket
collision-transform guards had zero error; this is not a separate GPU transform
remeasurement. Contact timings resolve 4.17 ms saved intervals.

The farther low-height/low-tilt expansion often failed workspace admission;
15/24 board contexts per target were rejected before release-gap queries. A
useful next bounded study would decouple these changes: lower board height while
retaining moderate tilt near the dropper's admitted height ceiling, then consider
new board XY under a new contract. This is a proposed experiment, not verified
feasibility or proof that farther shots are impossible. No additional RL was run.

## Interface and stable runtime

`http://127.0.0.1:18781/#distanceStudy` shows the completed matching audit, both
rays, all five exact recipes, repeat/gate tables and 14 saved native examples:
ten selected nominal trajectories plus four explicitly post-hoc far misses. The
optional far-miss adapter verifies records/cache hashes and exact column/geometry
identity before display; it changes no score or gate. Select a far `corner-d01`
through `corner-d04` miss to inspect the extra tabletop rebound.

Immutable execution source: `~/sircus-mjlab/studies/distance-v1`; raw evidence:
`~/sircus-mjlab/output/distance-v1`. Both finite study and CPU-audit workers exited.
Existing viewer static files/derived data were updated without restarting it or
activating a policy. 4070 workload, Isaac installation/workloads, Viam/hardware and
audio were untouched. [CPU preflight](cpu-preflight.json) remains separately
identified geometry-only evidence from before the native run.

Held joint endpoints and strict geometric checks do not establish arm transit,
actuator tracking, perception or physical validation. The bucket's reported 175 mm
height remains unconfirmed; provisional 175 mm interior plus 8 mm base is retained.
