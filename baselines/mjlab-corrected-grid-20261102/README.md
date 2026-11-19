# Corrected-contact spatial study — November 2, 2026

Completed synthetic mjlab/MuJoCo Warp feasibility study on RTX 4090. Both audits
passed. The 4070 workload and Isaac setup were left untouched. No new PPO,
checkpoint replacement, hardware, Viam or audio work occurred.

## Results

| Stage | Shot requests | Native releases | Captures | Clean captures |
|---|---:|---:|---:|---:|
| Initial geometry |11,760|0|—|—|
| Search after strict endpoint admission |460|460|349|141|
| Selected-source nominal repeats |87|87|83|63|
| Selected-source fresh perturbations |580|567|495|337|

Thirteen fresh geometry rejections remain failures in the 580-request denominator.
All 1,127 shot requests and 1,114 releases are accounted for. Combining search
and selected-source repetitions into 927/1,127 would be selection-biased; it is
not a frozen-policy success rate. No learned checkpoint was produced here.

Of 35 development bucket positions:

- **12 qualify**, with 17 source settings. Each has a checked path from the same
  home and back, 3/3 nominal captures and at least 18/20 independent fresh captures.
- **5** have captures and paths but fail the repeat-test gate.
- **5** have captures but no admitted path from the prescribed common home.
- **5** have admitted endpoints but no search capture.
- **8** have no admitted endpoint in this finite proposal family.

These mutually exclusive categories retain all 35 requested positions. In total,
27 positions have an endpoint, 22 have a search capture, 17 have a common-home
connection and 29 clear the starting posture. Five positions have two qualified
sources. All ten positions in the X=1.05/1.24m grid columns had zero captures;
more updates to this same finite bank cannot invent their missing actions.
A failed finite search/path family does not prove universal impossibility.

[Final native results](native-results.json) include all 35 cell summaries, every
qualified source and its audit/provenance identities.
[Exact qualified configurations](qualified-configurations.csv) list bucketXY,
boardXYZ, tilt, drop gap, ball-center release height and both 6-joint vectors.
Heights are relative to the table; release height is ball center, not flange.

## Audits and physics

Original audit: **3,115 checks passed**. Supplemental admission-coverage audit:
**3,165 passed**. Each independently rescores the same 1,114 releases from saved
240Hz trajectories and categorized contacts. Their checks overlap; the sum is
not 6,280 unique tests. Both bind contract fingerprint
`8edba8a3349aa8bcb4c06b97f3f61a302684856513b0ac972cf849afd3f4f037`.
Native execution completed in 3,546.90s, including CPU admission and queue time;
this is not a physics-kernel benchmark. Finite native/audit/display jobs exited.

The model uses an articulated board, passive gravity compensation only and
64 substeps per 240 Hz scorer(15,360Hz). Material and capture definitions are
unchanged: board rebound, whole-ball entry, continuous one-second retention;
rim-assisted captures count and externally assisted routes fail. Original results
retain their original physics contracts. Joint endpoints are prescribed during
flight. Sampled geometric paths are separate from actuator dynamics.

The measured board 300×200×5mm and tapered bucket are preserved. Reported 175 mm
height remains unresolved; provisional 175 mm interior plus 8 mm bottom remains.
Ball, mount and contact properties are synthetic, not physically calibrated.

## CPU preparation and fixed home

[Initial CPU admission](cpu-admission.json) is preserved unchanged, separately
from final native results: 11,760 proposals, 460 admitted endpoints and 29 home-clear
positions. Bucket centers span X100–1240mm/Y−650–250mm. This finite grid does not
measure the fraction of the continuous workspace covered.

The same frozen home has dropper flange(250,−100,950)mm and board center
(670,−220,300)mm. Four layouts obstruct that posture. Two near-base layouts fail
initial clearance; [supplementary convex-hull evidence](base-overlap.json),
reproduced by [this CPU script](base_overlap_probe.py), verifies modeled fixed-base
intersection at exactly(100,25) and(1240,25)mm. The positive intersection-ball
radius is not penetration depth. Original conservative categories are preserved.

## Are we ready to train?

**Evidence is ready for a limited pilot; the matching learner adapter and smoke
checks remain a launch gate. Broad all-table training is premature.**

The 17 source records are distinct absolute poses/joint pairs, but they represent
only 12 oriented relative geometry prototypes and six height/tilt/gap triples.
Deduplicate prototypes before sampling, retain every source branch's provenance,
and share the same geometry/path masks and joint caches between learned and
uniform policies. Reused endpoint geometry does not prove interchangeable
clearance or dynamics. If branch variants remain actions, expose distinguishing
branch/joint features to the actor.

Existing policy-network/PPO mathematics is reusable. Existing training/inference
runners still instantiate legacy physics and historical masks; they must not be
used as a corrected-contract learner unchanged. Implement and smoke-test corrected
64-substep/passive-only contact, strict endpoints and common-home admission first.
Then run a fresh-init 2,048–4,096-throw centralized contextual shot-selector pilot
against identical uniform feasible selection. This is not joint-trajectory RL.

Freeze the candidate bank and existing 35/30/100 train/validation/test coordinates.
Future validation/test were fixed before this study's native outcomes and have
received no outcome search. Keep failed layouts in headline denominators, select
checkpoints by validation only, and evaluate frozen policies without test search
or outcome-dependent resampling. Additional far-position feasibility work is
separate from this pilot and cannot be presented as held-out evaluation.

See [protocol and reproduction](../../sircus/mjlab/CORRECTED_GRID.md) and the live
read-only `#correctedGrid` panel. Historical CPU artifacts and prior baselines are
unchanged. [Manifest](manifest.json) covers the curated artifacts; raw arrays,
logs and immutable runtime snapshots remain outside Git.
