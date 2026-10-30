# Front-bucket study record

Implementation and reproducible commands are in
[`sircus/studies/README.md`](../sircus/studies/README.md). Evidence is synthetic;
there was no robot, Viam, camera, or audio interaction during numerical training.

The corrected profile is `front-negative-y-v1`: dropper base origin, +X toward
arm14, +Z up, operator and bucket toward -Y. Nominal board center
[600,-150,350] mm and RPY [0,-30,90] are provisional. The bucket dimensions and
landscape board dimensions are unchanged; reported bucket height 175 mm remains
unresolved. This profile is separate from the historical positive-Y scene.

## Initial corrected run

Run directory: `.runtime/overnight-study/front-initial-20261030`.
Seed 20261030; train/validation/test 24/6/8; teacher budget 12; separately labeled
held-out search budget 4. Source/dependency/asset guards and manifest checks passed.
Elapsed 69.95 s with at most two numerical workers.

| Method | Captures | Rate | 95% layout Wilson interval |
|---|---:|---:|---:|
| Frozen learned checkpoint | 3/8 |37.5%|13.7–69.4%|
| Predeclared fixed action |2/8|25.0%|7.1–59.1%|
| Limited4-proposal search oracle |2/8|25.0%|7.1–59.1%|

Validation 3/6 selected degree 2 ridge regularization 100 from three predeclared
candidates. All 24 training scenes had valid labels; 21 teachers captured. Test
failures were `no_entry`; no test endpoint infeasibility occurred. The modest
sample does not establish generalization or physical transfer.

Checkpoint ID:
`8ca17aa286758350c0074bbb770f1225b239539f348743b82689b872b5d51716`.
Checkpoint file SHA256:
`6fc92e729ad5411487b0fdb5ade7b1c293c1ce1ab2c67928e3a8efd4e970a120`.

Historical positive-Y 24/6/8 evidence remains separately at
`.runtime/operator-lab/training/75dd3f096989450b8dbba38bf7c84cc6`:
frozen 8/8, fixed 6/8, search 8/8, as recorded by the integration owner. It uses a
different geometry/distribution and is not a controlled comparison with the new
front profile or larger-study splits. Historical files were not rewritten.

## Paired learning curves and challenges

The larger run freezes three nested training pools and shared validation 16/test 48
layouts before training. Seeds 20261030/31/32 each use N 8/24/64/128. Teacher and
oracle budgets are 16. Each model sees only training labels; validation selects
one of three model/regularization choices; inference on test/challenges is a
frozen matrix prediction without per-scene optimization. The default checkpoint
is prespecified N 128/seed 20261030, never best test.

The fixed shared-test comparator captured 25/48 (52.1%,95% 38.3–65.5), including 2
endpoint-infeasible failures. The separate limited 16-proposal search oracle
captured 37/48 (77.1%,95% 63.5–86.7). These held-out layouts differ from the initial
24/6/8 run, so changes in raw percentages across those runs are not learning gains.

All 12 frozen models have now been evaluated on the same 48 held-out layouts:

| Sampled training layouts | Seed 20261030 | Seed 20261031 | Seed 20261032 | Mean capture rate |
|---:|---:|---:|---:|---:|
| 8 | 28/48 | 28/48 | 29/48 | 59.0% |
| 24 | 31/48 | 32/48 | 34/48 | 67.4% |
| 64 | 32/48 | 34/48 | 31/48 | 67.4% |
| 128 | 33/48 | 34/48 | 33/48 | 69.4% |

The denominator is 48 independent layouts, shared across seeds and sizes, not
144 independent tests. Seed variability is separate. Performance is not
strictly monotonic and appears to plateau around two-thirds in this narrow
synthetic distribution; these results do not establish reliable deployment.

N denotes sampled training layouts. Seeds 20261030/31 fit all sampled labels.
Seed 20261032 fits 8, 24, 63, and 125 valid labels at the respective sizes.
Its training indices 64, 82 and 96 (one-based) had all 16 proposed actions fail
dropper endpoint IK while the board-arm endpoint was reachable. This is failure
within the finite teacher budget, not proof the layouts are inherently infeasible.
All 48 failed proposals remain in the raw evidence. Across full pools there are
384 sampled layouts, 381 valid labels, and 291 teacher captures, from 6,144
proposals. The initial run adds 288 proposals separately.

Final curves, challenge tables, manifests and replay frames are generated under
`.runtime/overnight-study/front-curves-20261030`. The compact UI artifact is
`report.json`; the default checkpoint is `checkpoint.json`. The complete portable publication is
[`baselines/front-bucket-20261030`](../baselines/front-bucket-20261030/README.md),
with all checkpoints, final summaries, manifests, scorecards and checksums.
All 18 study regressions pass, including artifact integrity, all 576 saved
frozen actions, and numerical trajectory reproduction.

Distance/lateral-offset challenges extend beyond core training bounds. Release
noise uses 24 independent layouts with three replicates each. Intervals cluster
repeats and training seeds by layout; seed-level rates/ranges remain separate.
A declared contact intervention separately forces center/edge/corner U/V while
holding each frozen prediction's board roll/pitch and gap fixed. Its results
measure sensitivity, not policy inference performance.

All reachability results are endpoint IK only. Native collision-path replay,
calibration, realistic contact identification and physical evaluation remain
outstanding; compatibility guards stay enabled.

## Completed challenge interpretation

The default checkpoint captures 33/48 (68.75%, 95% Wilson 54.7–80.1%). Across
three fixed training seeds the mean is 69.4%, with layout interval 55.4–81.9%
and seed range 68.8–70.8%. The same 48 layouts are reused; this is not 144
independent tests. The finite search comparator is 37/48, not a learned model.

Default distance results degrade from 24/24 at 220 mm to 17/24 at 280 mm and
0/24 at 340/400/450 mm. The 340 mm layouts are still within core training
bounds. At offset ±200 mm the model is 0/24. Post-study 16-proposal controls
find 18/24 captures at −200 mm and 12/24 at +200 mm, exposing extrapolation
failures with available bounded-search solutions. At 340 mm search finds 2/24;
at 450 mm it finds 0/24. Neither zero proves impossibility. No checkpoint was
refitted or selected in response to these diagnostics.

Noise-cohort capture rates are not overall held-out accuracy: its 24 unique
layouts are closer on average (207.1 mm, only 2 beyond 280 mm) than the shared
48-layout test (241.2 mm, 16 beyond 280 mm). Original manifests were not
resampled. Published intervals cluster the three noise replicates per layout;
raw stdout generic per-attempt noise intervals are not inference-quality.

Forced center/edge/corner experiments change trajectory, endpoint reachability
and compliance together. They do not isolate suspension. For the default model,
center and +65 U each capture 31/48, while −65 U, ±95 V and corners capture 0/48.
The +65 U group includes 14 unreachable endpoint attempts. Conditional support
tilts are larger away from center, but do not justify a causal suspension claim.

All four same-layout progression clips (the first predeclared test layout at
N8/24/64/128) miss. They remain in the gallery unchanged rather than being
replaced by a favorable demonstration. The gallery also retains successes and
near/far failures from separately labeled layouts.

Exact core protocol counts: 6,144 teacher proposals, 576 validation attempts,
576 frozen test attempts, 48 fixed and 768 search proposals, 2,376 frozen
challenge attempts, 1,296 forced-contact attempts, and post-study controls of
1,920 search proposals plus 120 fixed attempts. There are 832 distinct core
layouts; including the initial study gives 846, because its 24 training layouts
overlap the first nested pool. Render/replay checks are excluded from these
protocol counts. All numerical jobs finished; raw evidence remains ignored.

Independent augmented-least-squares reconstruction matched all 12 model fits
from their exact train-only labels within 1.89e-15. This and the saved evidence
guards establish numerical reproducibility, not physical transfer. Next work
should improve teacher/action coverage and identify real contact/tool geometry
before calibrated native-path and real-world validation.

The completed 78-second H.264 study film is included in the results interface
and under `sircus/simulation/web/media/front-bucket-20261030/`, with a poster,
source-report hash, replay IDs and SHA256 checksums. It renders actual saved
ball frames and static arm endpoint geometry at15fps; it is not a screen recording
or native/physical motion validation. Full video decode completed without errors.
