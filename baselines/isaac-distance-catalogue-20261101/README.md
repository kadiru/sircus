# Bucket-distance catalogue — 2026-11-01 UTC

Completed synthetic PhysX development search on RTX 4090. No new RL policy was trained. Prior fixed-distance results and scene fingerprints remain unchanged.

## Frozen winners and fresh confirmation

Each winner was selected using separate screening trials, then frozen before 50 new seeded variations. Board height and gap vary independently by ±2 mm; tilt varies by ±0.5°. Heights are above tabletop Z=0. Gap is vertical from board center to the ball bottom at release.

| Bucket distance mm | Board H mm | Tilt ° | Gap mm | Ball bottom Z mm | Ball center Z mm | Fresh clean / assisted / miss / invalid | Capture rate |
|---:|---:|---:|---:|---:|---:|---|---:|
| 300 | 350 | 24 | 200 | 550 | 587.5 | 49 / 1 / 0 / 0 | 50/50 |
| 400 | 400 | 21 | 130 | 530 | 567.5 | 50 / 0 / 0 / 0 | 50/50 |
| 500 | 450 | 19.5 | 150 | 600 | 637.5 | 0 / 44 / 6 / 0 | 44/50 |
| 550 | 550 | 13.5 | 80 | 630 | 667.5 | 0 / 41 / 9 / 0 | 41/50 |
| 600 | 450 | 13.5 | 130 | 580 | 617.5 | 0 / 49 / 1 / 0 | 49/50 |
| 650 | 500 | 18 | 80 | 580 | 617.5 | 0 / 37 / 13 / 0 | 37/50 |

Provisional dropper-flange Z is ball-center Z +95 mm; the CSV includes it. These are endpoint targets, not a collision-checked approach path. A clean capture means no detected pre-entry/near-rim bucket contact, ≥1 mm sampled mouth clearance and no later partial exit. Assisted/marginal captures still count as success. Both tiers require a board bounce, whole-ball entry and one continuous second inside. Multiple board contacts are allowed.

50/50 is finite evidence under this narrow simulated variation, not a guarantee. Its Wilson 95% interval lower bound is about 92.9%; additional physical, perception and material variation is untested.

## Search coverage

| Distance mm | Searched | Clean | Assisted | Released misses | IK/settling rejected |
|---:|---:|---:|---:|---:|---:|
| 300 | 276 | 58 | 100 | 51 | 67 |
| 400 | 1 | 1 | 0 | 0 | 0 |
| 500 | 276 | 0 | 26 | 182 | 68 |
| 550 | 276 | 0 | 45 | 166 | 65 |
| 600 | 276 | 0 | 51 | 158 | 67 |
| 650 | 276 | 0 | 49 | 158 | 69 |

The 400 mm row is a control that rechecks the previous strongest setting, not a new broad search. Each other distance uses 216 unique coarse proposals and up to 60 local refinements; up to three separated candidates receive five unchanged and 20 varied screening trials. See [protocol](../../sircus/isaac/DISTANCE_CATALOGUE.md) for exact grids, deterministic selection and disjoint seeds. No captured setting found means this bounded search did not find one—not proof of impossibility.

## Verification and provenance

- Run: `distance-catalogue-20261101T125810Z`; finite run duration 72.4 minutes.
- 2081 attempts: 1738 released, 343 rejected; every trial ID unique.
- Maximum accepted board position error 0.609447 mm; normal error 0.076630°; joint speed 0.00025272 rad/s. Pose/drift/settling gates were not relaxed.
- Floor/rim/outside physical-contact filter probes passed at every distance. The exporter checks grid/refinement membership, candidate selection, exact perturbation seeds, frozen winner selection, trial aggregates and capture/quality gates.
- Actual source hashes, contract fingerprint and raw artifact checksums are retained in `results.json`. Compact synthetic trial parameters/results are included; raw trajectories, joint states and logs remain on Ubuntu outside Git.
- Runtime: Isaac Sim 5.1 / Isaac Lab 2.3.2; executed immutable source snapshot `studies/distance-catalogue-v1`. Implementation introduced in `d868fdd46c8ef927a0295c2bdd512070acb798b1`; the later diagram/report changes do not alter the running experiment.
- Local full mock/numerical regression and 75 dependency-light Isaac tests passed; distance-panel tests include pending/no-solution separation and selected-geometry switching.

## Additional path audit — original scores unchanged

The original success criterion allows intermediate contacts. Saved 30 Hz paths were checked for low external rebounds; these flags suggest tabletop/other contact, not a measured table-contact identity. Missing flags do not certify direct travel.

| Distance mm | Captured trials with low external rebound flag | Missing captured paths |
|---:|---:|---:|
| 300 | 0/50 | 0 |
| 400 | 0/50 | 0 |
| 500 | 7/44 | 0 |
| 550 | 39/41 | 0 |
| 600 | 49/49 | 0 |
| 650 | 36/37 | 0 |

Do not use high capture rates with these warnings as evidence of direct board-to-bucket shooting. Before a direct-shot learning study, add table/robot contact instrumentation and a separate versioned path criterion. The present catalogue remains valid only under its original broader capture definition.

## Scope and next decision

The 300–400 mm settings are the strongest candidates under this narrow test.
At 500 mm all 44 captures were assisted/marginal and seven had external-rebound
warnings. At 550–650 mm nearly all captured paths were flagged. Do not promote
these farther settings as reliable direct shots based on their capture totals.
Instrument intermediate contacts before selecting demonstrations for RL.

Board/release XY=(670,−100)mm; bucket center moves toward−Y. The 650 mm placement leaves 25 mm between its outer top footprint and the table edge. Board 200 × 300 × 5 mm, bucket inner 177/150 mm and outer 200/158 mm remain preserved. Reported bucket height 175 mm still has an unresolved overall-height versus interior-depth interpretation; this run uses provisional 175 mm interior depth plus 8 mm base. Ball radius 37.5 mm, mass, contact response, tool offsets and rigid support remain provisional.

This is a lookup catalogue at tested distances, not a distance-conditioned policy or evidence that interpolation works. Reachability/settling are checked at endpoints; grasping, arm travel, compliance and perception-driven control are unvalidated. Preserve weak/no-solution distances when designing the next study; do not replace them with optimized inference labeled as policy performance.

## Files

- [Validated aggregate evidence](results.json).
- [Selected settings and fresh outcomes](selected-settings.csv).
- [Independent source/artifact/raw-record integrity check](integrity.json).
- Read-only preview: `http://127.0.0.1:18780/#distance-catalogue` through the Mac-to-Ubuntu SSH forward. It shows counts, clean/assisted rates, screening comparison and nominal side-view geometry.

To inspect this frozen report without a simulator, from the repository root:

```bash
SIRCUS_REVIEW_DIR="$(mktemp -d)"
cp baselines/isaac-distance-catalogue-20261101/results.json "$SIRCUS_REVIEW_DIR/distance-catalogue.json"
cp baselines/isaac-distance-catalogue-20261101/selected-settings.csv "$SIRCUS_REVIEW_DIR/distance-settings.csv"
python3 sircus/isaac/preview.py --output "$SIRCUS_REVIEW_DIR" --port 18861
```

Open `http://127.0.0.1:18861/#distance-catalogue`. Other study panels have no data
in this isolated directory. No Isaac dependencies or hardware credentials are needed.
