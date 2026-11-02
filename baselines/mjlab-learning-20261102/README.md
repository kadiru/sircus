# Bounded mjlab learning and interactive shot selection

These are native mjlab/MuJoCo-Warp simulation results on the RTX 4090. Arms are
initialized at audited joint endpoints and held during each ball flight. Pickup,
arm transit, perception and physical validation are not part of these policies.

## Continuous configuration policy

8,507 training attempts: 4,096 exploration, 315 perturbed development confirmations
and 4,096 on-policy PPO attempts (32 updates). A train-only imitation warm start
precedes PPO. Inputs encode bucket XY and optional board/release position locks;
actions choose board X/Y/height, tilt and drop gap. Heading faces the bucket and
release XY equals board-center XY.

The 35-position training grid is separate from 20 validation and 50 continuous
test coordinates within X100–1240 / Y−650–250 mm. Validation selected update24.
The original test used three nominal repeats per position:

| Method | Captures | Clean | Invalid |
|---|---:|---:|---:|
| Initial fixed actor | 27/150 | 0 | 66 |
| Imitation warm start | 58/150 | 23 | 18 |
| Selected PPO | 65/150 | 16 | 9 |

Separate conditional tests scored20/50 with board XYZ locked and23/50 with
release XYZ locked. These locks were anchored to the selected free policy's
endpoints; they do not establish reliability for arbitrary operator constraints.
The development search found captures at23/35 positions;16 selected settings
passed3/3 small-perturbation confirmations. Three repeats are limited evidence.

## Relative-bank policy: a fresh comparison

Sixteen train-only teachers with3/3 confirmations form256 fixed candidate slots
(16 rotations per teacher). Geometry-only wrist, IK, native FK and collision
checks mask unavailable candidates. A shared categorical actor then selects one
candidate; each eligible episode receives a fresh native ball simulation.

The additional2,048 PPO training attempts use78 endpoint-eligible contexts out of
99 declared training layouts. Twenty new validation positions selected update12.
The following comparison uses **another50 previously unused positions**, three
nominal repeats each, and includes all empty-mask cases as failures:

| Method on the fresh split | Captures | Clean | Invalid |
|---|---:|---:|---:|
| Frozen continuous PPO, rerun | 75/150 (50.0%) | 12 | 12 |
| Uniform feasible-bank selection | 94/150 (62.7%) | 59 | 36 |
| Frozen learned bank selector | 93/150 (62.0%) | 42 | 36 |

The selector **did not outperform uniform feasible-bank selection**. Its gain
over the continuous actor is not evidence that PPO improved the bank. Thirty-two
test positions captured at least once;12/50 had no endpoint-feasible candidate
under this finite bank and bounds. Those positions remain unresolved, not proven
impossible. No per-scene rollout search repairs these policy scores.

## Scoring and geometry

All runs preserve `ShotScore`, `EntryQuality` and `BoardRoute`: measured board
rebound, whole-ball entry and one continuous second inside, without intervening
table/floor/robot/unattributed contact. Rim-assisted captures count; partial exits
reset retention. Physics1920Hz; scoring240Hz; fixed calibrated-seed contact
parameters `solref=(.004,.15)`, friction.35. These are uncalibrated simulation
materials, not an assertion of physical equivalence to Isaac/PhysX.

Board200×300×5mm body dimensions plus the90° landscape mount; measured base
spacing52.65in (base-to-base interpretation assumed). Bucket inner diameters
177/150mm and outer200/158mm are preserved. Reported height175mm is unresolved;
the model assumes175mm interior depth plus8mm bottom. Ball and mount properties
remain provisional. Historical Isaac and rectangular-container results retain
their original scenes and definitions.

## Evidence and reproduction

- `continuous-*`: frozen contract, selected model, compact results, full selected
  teacher catalogue, runtime-asset hashes and407-check independent audit.
- `bank-*`: fresh contract, selected categorical model, compact results,
  first-repeat examples and170-check independent audit. `proposal-bank.json`
  preserves source trial IDs and qualifications.
- `manifest.json`: checksums for the small published evidence. Full raw JSONL,
  geometry arrays, trajectories and optimizer checkpoints remain on the Ubuntu
  study host outside Git. Both original source snapshots and reports are intact.

See [learning protocol and commands](../../sircus/mjlab/LEARNING.md),
[interface contract](../../sircus/mjlab/web/README.md) and the two CPU auditors
`verify_learning.py` / `verify_bank.py`. The continuous and categorical
checkpoints are different model classes; load each with its matching policy.
Artifact guards bind runtime source, assets, scorers, dependency versions and
checkpoints before a native interface request can run.

The earlier split-screen video compared identical recorded release states in
Isaac PhysX and mjlab using a common MuJoCo renderer. It compared contact physics
at fixed arm poses; it was not a comparison of trained policies or learning speed.
