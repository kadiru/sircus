# Seed22: repeated versus broader TRAIN locations

Prepared only. No policy loading, simulator, training, or shots ran. The new
coordinate package has execution and release authorization set to false. It
does not modify any completed benchmark, checkpoint, result or native guard.

The question is whether spending the same number of additional attempts across
more bucket locations improves the existing neural network's ability to catch
at new locations. Both copies restore the **original seed 20261122 weights and
Adam optimizer**, including its 10,000 previous requests. Neither copy starts
from random weights or the later 8,960-request continuation checkpoint.

## The two groups

| | A: original usable locations | B: broader locations |
|---|---|---|
| Additional TRAIN requests, proposed | 8,960 | 8,960 |
| Distinct bucket locations | 31 | 8,960 |
| Attempts per location | 289 or 290 | 1 |
| PPO updates | 64, each using 140 requests | Identical |
| Optimizer passes | 4 per update | Identical |

A balances a shuffled cycle through the 31 admitted original targets. B draws
new coordinates uniformly in the original goal rectangle, conditioned on valid
bucket placement and exclusion of the reserved held-out areas. Every B coordinate
is distinct, but there is no minimum spacing between B TRAIN points. Nearby
points are expected. The whole schedule is fixed before any model output or
shot outcome is available. This tests spatial breadth, not a new exploration
algorithm or more action attempts per bucket.

Both groups use the same network, action bounds, reward, physics, materials,
continuous geometry wrapper and motion/settling/release guards. There is no shot
library, imitation or old-shot replay. PPO continues to use the original sampled
latent and log probability; a geometry-adjusted execution setting is not
substituted into the PPO likelihood. Separate coordinate/order/action-noise
streams prevent extra coordinate draws from changing exploration randomness.
The action-noise and optimizer streams are paired across groups; their different
target distributions do not provide a tile-by-tile context pairing.

Invalid bucket placements are removed **before** schedules are created. They
do not spend a proposed shot request. Once a policy request is reserved, every
geometry or native rejection remains charged, without retries or refunds. The
35 parallel simulator slots are capacity, not a requirement to keep 35 distinct
original target locations. A future adapter must support per-request target
placement and identical scene resets in both groups.

## The blocked bucket placements

The mjlab chat's two blocked fixed-grid positions are not this experiment's
original targets. Isaac's original 35 were randomized within tiles. Checking
those actual points found **four authored bucket/base overlaps**, leaving 31:
`train-021`, `train-027`, `train-028`, `train-034`.

The private source receipt verifies the original split, scene, source and ten
asset hashes. Each of the four exclusions has a saved interior-overlap witness
between the authored convex base collider and the bucket's bottom cylinder.
The smallest common interior sphere is 1.433 mm; the other three are 4.000 mm.
These are source-geometry proofs, not new PhysX cooking or physical measurements.
Inherited provisional bucket-depth/bottom assumptions remain provisional.

For new targets, require the whole bucket footprint inside the table and at
least 10 mm between conservative bucket/base enclosing circles. A failed circle
test is generally conservative; it alone does not prove impossible placement.
The four original exclusions additionally have the positive overlap witnesses.
The 31 admitted originals all have at least 15.587 mm enclosure clearance.
This target-placement rule does not certify moving links, arm reach, the drop
gap, paths, settling, release safety or a successful bounce. Those existing
shot guards still apply.

The preparer pins the verified receipt fingerprint as well as source identities.
Changing a radius or transform and re-signing the receipt cannot change the
target-placement rule. Raw assets, calibration and runtime configurations stay
outside Git.

## Fresh measurements

The coordinate inventory includes **1,131 known historical positions**, discovered
from 150 fetched refs, 502 distinct baseline JSON blobs and nine named private
history/split manifests. Only coordinates influenced selection. This is not a
claim of discovery of every unpublished or remote experiment.

- **35 monitoring positions**: the shared unchanged parent at zero, then each
  copy at 2,240 / 4,480 / 6,720 / 8,960 requests. **315 total monitor requests**.
- **70 primary TEST positions**: the same positions for unchanged parent +
  checker, A + checker and B + checker. One deterministic mean proposal each.
- **25 secondary TEST positions**, 50 mm from original admitted anchors, to
  measure nearby-location sensitivity separately. Six anchors yielded no valid
  offset within the finite 100,000-direction search allowance; they are listed
  in the report and are not replaced or given relaxed rules.
- Final TEST therefore has **285 requests** across the three models. Combined
  with monitoring, the proposed assessment total is **600**, separate from the
  **17,920 proposed TRAIN requests**.

All 130 held-out positions are at least 25 mm from discovered historical
coordinates and from each other. Every TRAIN coordinate is at least 25 mm from
all new held-out positions. New B TRAIN points exclude exact old coordinates;
they may be near historical points, avoiding the severe area distortion caused
by placing 25 mm exclusion circles around all historical TRAIN locations.

Freshness and placement condition the spatial distribution. A full original
7-by-5 tile quota is inappropriate because fixed bases occupy entire tiles.
The actual B schedule covers 33 tiles; primary TEST covers 25. Report these
counts and the distribution's limits. Neither cohort is selected by IK or catch
success. The local 25-position result applies only to sampled anchors; six
unsampled anchors are not counted as misses or silently removed from a claim
about all 31.

Primary outcome: **captures / all 70 primary TEST requests**, including
rejections as failures. Also report clean captures, released misses, rejection
causes, paired per-target changes, and explicitly labelled capture/release
rates. Keep local-offset results separate. Monitoring cannot select an earlier
checkpoint, tune the protocol or stop training based on score. Freeze all final
model hashes before inspecting final TEST outcomes. Rotate the three models'
evaluation order by target index, reset the scene between requests and use no
held-out search, retries or updates.

Report actual proposal, geometry, native rollout, PPO, checkpoint, monitoring,
TEST and CPU audit times, plus RSS, sampled GPU memory and request/release
throughput. Geometry time nested inside rollout time must not be added twice.
Save prescheduled videos independently of success. A then B is a fixed sequential
order, so differing host conditions limit causal claims about runtime.

## Proposed bounds and remaining implementation

The request cap is 8,960 per group. Each group also has a proposed eight-hour
limit including startup and its four monitor passes; shared baseline/final
assessment has one additional hour and the CPU audit two hours. These are new
proposals, **not reuse of the completed 30,000-request/eight-hour authorization**.
They could require up to 17 native hours, not one combined eight-hour run. Keep
the existing 6 GiB storage floor and preserve every partial request and artifact.

If either group misses its exact budget due to time, resource or execution
failure, the full equal-budget comparison is incomplete. Do not evaluate
unmatched terminal models as if budgets were equal, auto-retry paid work or
select a better checkpoint. One paired continuation remains a single-parent
comparison, not repeated-seed reliability or real-hardware evidence.

`prepare_location_comparison.py` writes a fresh private package and refuses to
overwrite one. `location_comparison.py` builds and regenerates the deterministic
contract. The old fixed-35 continuation launcher cannot consume this new schema.
Before any later launch, implement and mock-test the production adapter,
independent request/geometry/model/physics audit and bounded launcher, including
per-request target placement, slot resets and matching PPO batch boundaries.
Obtain execution authorization separately. No launcher is added by this task.

Preparation evidence and hashes: [report](../../baselines/isaac-seed22-location-comparison-prep-20261116/README.md).
