# Dense fixed-plane shot catalogue

This is a bounded development search in articulated Isaac/PhysX, not reinforcement
learning or a claim that simulation shots are physically calibrated.

## Coordinates and search

The conceptual vertical plane is Y = −100 mm. Board center and ball release share
X = 670 mm, Y = −100 mm. Bucket center is (670, −500) mm: **400 mm horizontal
plane-to-bucket-center distance**. Tabletop is Z = 0. Tilt is from horizontal,
with the board facing the bucket toward −Y. The plane is not an obstacle.

- Board-center H = 300 through 500 mm, in 25 mm increments (9 values).
- Gap G = 80, 120, 160, 200, 240 mm (5 values).
- Tilt = 12 through 42 degrees, in 3 degree increments (11 values).
- 495 coarse combinations, including rejected endpoints in the denominator.
- Refine the top 12 captured combinations by score, then entry clearance:
  G ±10 mm and tilt ±1.5°, including unchanged values and deduplicating.
- Select six geometrically separated captured settings before confirmation.
  Each receives 20 unchanged trials and 20 independent seeded trials with
  H ±2 mm, G ±2 mm and tilt ±0.5° (uniform distributions).

Ball-bottom release Z = H + G. Ball-center release Z = H + G + 37.5 mm.
Dropper-flange release Z is another provisional 95 mm above the ball center.
G is a vertical gap relative to board center, not the perpendicular gap from
a sphere to a tilted surface. Initial ball velocity is zero.

Board size is 300 × 200 × 5 mm. Bucket internal top/bottom diameters are177/150 mm;
external200/158 mm. Reported175 mm height remains ambiguous; this model assumes
175 mm interior depth and8 mm bottom. Ball size, materials, masses, tool offsets
and rigid support behavior remain provisional and unchanged from the parent run.

## Quality scoring (additional, versioned)

The historical capture definition is unchanged: measured board contact and
rebound, whole-ball entry, one continuous second inside. A partial exit resets
retention. The new tier is **additional** to that scorer, not a replacement reward
for old policies or results.

| Score | Meaning |
|---|---|
| 2 | Captured, with ≥1 mm sampled polygon-mouth clearance, no detected bucket contact before entry, no near-rim bucket contact during flight, and no later partial exit |
| 1 | Captured, but contact-assisted, marginal/unknown clearance, or partial exit before eventual retention |
| 0 | Released but missed the complete-shot capture criterion |
| × | IK/settling rejected; no release, separately reported |

“Clean” describes bucket entry, not exactly one board contact. The existing scorer
permits repeated board bounces or sliding before entering the bucket; a strict
single-board-impact objective would need its own event-count contract.

Both contact filters (board and bucket) are sampled at240 Hz. A bucket force above
0.01 N is contact; the near-rim band conservatively includes ball-top heights
within2 mm below the rim or higher. Deep interior/floor bounces after a clean
entry do not reduce its tier. Bucket forces alone cannot identify exactly which
wall/rim point was touched, so score1 is called **assisted or marginal**, not
always a proven rim strike. The clearance is the sphere cross-section's minimum
margin against the64 polygon mouth planes at sampled heights. Sampling/force
thresholds cannot establish that every arbitrarily small contact is detected.

Before searching, three simulator probes must pass: centered drop onto the bucket
floor and a rim drop produce bucket contact; an outside drop does not. This gate
prevents silently labeling clean shots with a broken sensor filter.

## Settling, provenance and limitations

Uses the repaired endpoint IK and measured board settling controller from
`BALANCED_PILOT.md`. Board position error ≤2 mm, normal error ≤0.5°, joint speed
≤0.005 rad/s, drift ≤0.1 mm /0.05° are required before release. Both arms initialize
at their solved endpoints; this does not validate collision-safe travel, pickup,
gripper release dynamics, suction compliance or physical reachability.

Selection is frozen in `selected.json` before confirmation. `trials.jsonl` assigns
unique trial IDs, preserves requested and measured state, and separates grid,
refinement, nominal and perturbed stages. Source/contract/artifact hashes are
saved. Raw observations and host state stay outside Git; compact aggregate rows
and confirmation results may be committed separately.

No added randomness is used in nominal repeats. Zero failures in20 nominal trials
is repeatability evidence, not an independent estimate of physical reliability.
Perturbations test only the prescribed small input variations; they do not test
new distances, material changes, perception or robot execution. The stored Wilson
lower bound is a binomial reference under an independence assumption, not a
physical reliability guarantee. It is not used to select candidates.

## Reproduce and inspect

In the isolated Sim5.1 / Lab2.3.2 environment, on the4090 only:

```bash
export SIRCUS_ISAAC_SOURCE=/workspace/studies/YOUR_IMMUTABLE_SNAPSHOT
# Optional probe-only run: add --catalogue-probe-only
flock -n /workspace/scene.lock "$SIRCUS_ISAAC_SOURCE/python.sh" \
  "$SIRCUS_ISAAC_SOURCE/launch_scene.py" \
  --headless --enable_cameras --device cuda:0 --shot-catalogue
```

Use a finite timeout and preserve the3 GiB free-space guard. The read-only preview
serves `shot-catalogue.json` at `#shot-catalogue`: exact coordinates, separate
confirmation counts, quality heatmaps and all successful searched candidates.
Historical panels and scoring contracts are preserved. The preview reload is
separate from simulation and does not restart training, hardware or Viam.

## Independent follow-up

After reviewing the six initial confirmation sets, select the candidate with most
perturbed captures, then most clean captures, nominal captures and deterministic
ID tie break. Freeze it and run100 further perturbations using indices10000–10099
(seeds disjoint from the initial0–119 indices). No action retuning occurs in this
follow-up. Run with `--shot-catalogue --catalogue-confirm-source PATH/summary.json`.
The separate `catalogue-confirmation.json` must match the original run before the
interface displays it. Its report retains trial IDs, raw trajectory evidence and
source/artifact hashes.

[Completed search and initial confirmation](../../baselines/isaac-shot-catalogue-20261031/README.md).
