# Local height/tilt/timestep study

## Completed corrected study — November 3, 2026

The corrected4090 study completed in1,816.4 seconds of execution. Its independent
CPU audit passed3,430 checks and rescored all2,156 native releases. The immutable
study and a separate repeat audit agree. Raw observations remain on Ubuntu;
[curated contracts, counts and comparisons](completed/) are published here.

- 2,562 requested attempts:27 controls,1,275 coarse gaps,432 refinements and828
  paired repeat requests.2,156 actual releases,406 geometry rejections,1,507
  captures and39 clean captures. Adaptive search totals are not policy accuracy.
- 72/75 local height/tilt contexts had strict endpoints and an observed capture;
  this does not certify intermediate settings or other bucket positions.
- 12 frozen selected recipes received3 nominal +20 fresh joint H/tilt/gap
  perturbations at each of32/64/128 substeps. Noise sigma:1 mm height,0.25° tilt,
  2 mm gap. The same23 requests/joint arrays were reused across resolutions; these
  are not69 independent randomized scenes per recipe.
- 10/12 passed the prespecified64/128 engineering gate. All10 also had3/3 nominal
  and at least18/20 fresh captures at32; this latter extension is a descriptive
  all-resolution capture check, not a new prespecified gate.
- Eight scored23/23 at **every** resolution:

| Recipe | Board-center height mm | Tilt from horizontal | Gap mm | Ball-center release height mm | Captures at32 /64 /128 |
|---|---|---|---|---|---|
|00|397.388|36.1548°|321.219|756.107|23/23 ·23/23 ·23/23|
|02|440.000|32.0000°|285.781|763.281|23/23 ·23/23 ·23/23|
|03|437.388|40.1548°|301.219|776.107|23/23 ·23/23 ·23/23|
|04|480.000|28.0000°|265.781|783.281|23/23 ·23/23 ·23/23|
|08|377.388|32.1548°|331.219|746.107|23/23 ·23/23 ·23/23|
|09|480.000|40.1548°|278.531|796.031|23/23 ·23/23 ·23/23|
|10|460.000|36.0000°|265.781|763.281|23/23 ·23/23 ·23/23|
|11|480.000|32.1548°|258.531|776.031|23/23 ·23/23 ·23/23|

Recipe01 (H500 mm /36.1548° /gap238.531 mm) scored22/23,23/23,23/23; its32-step
fresh group was19/20. Recipe07 (H357.388 mm /40.1548° /gap386.219 mm) scored21/23,
23/23,23/23; its32-step fresh group was18/20. Both had3/3 nominal at every step.
Recipe07 had mixed clean/rim-assisted captures; its entry-quality labels changed
with resolution. All eight perfect-capture recipes were rim-assisted.

Do not promote recipes05/06 at H540 mm:05 scored13/23,21/23,21/23 and failed
nominal repeatability;06 scored7/23,10/23,12/23, including four rejected fresh
geometries per resolution. Both failed the original gate.

Recipe08 is a useful conservative candidate: all observed captures, no rejected
repeats and0.285 mm fine-pair nominal displacement at350 ms. The full table and
post-hoc all-pair displacement diagnostic remain separate from capture counts.
For example, recipe04 captured every time but its largest32-versus-finer nominal
position difference exceeded5 mm. Capture robustness is not proof of trajectory
or numerical convergence. The original gate applies its5 mm limit only to64/128.

The bucket and board XY remain fixed472.826 mm apart. This is local DEVELOPMENT
engineering evidence with privileged geometry and prescribed held joint endpoints;
no new RL, policy activation, unseen-bucket result, transit/actuator tracking,
perception or physical validation. Bucket175 mm interpretation and contact/mount
parameters remain provisional. Historical window/SAC and older scene results are
unchanged. Existing4070/Isaac workloads and the viewer service were not restarted.

`scene.json` preserves inherited asset definitions and historical PhysX labels.
The authoritative physics for this run is the MuJoCo/Warp configuration in
`completed/contract.json` and `completed/effective-options.json`; the inherited
labels do not identify the engine used for these new results.

Raw evidence: `~/sircus-mjlab/output/neighborhood-v2`; execution code58c1a9f,
adapterv2; contract fingerprint
`12b2952c68b38d6bb392e3256d140cb7445ea97c73b090a6a39c5dfc4a4aa680`.
The supervisor completed native execution and CPU audit and published the matching
derived viewer report. Execution PIDs below are historical, not liveness evidence.

## Startup status checked November 3, 2026 — historical

**No new flight results yet.** The original two-hour queue expired before any
physics. A single unchanged resubmission acquired the4090 but its option guard
stopped initialization: native tolerance1e-8 is converted by MuJoCo Warp3.11 to
float32 tolerance1e-6. Both attempts have zero native releases and remain saved.

The [CPU conversion proof](cpu-warp-option-conversion.json) verifies this vendor
rule and shows the parent/default configuration has the same effective values.
The fresh v2 adapter now checks both values exactly, without mutating them or
changing contact/scoring.42 focused tests passed in the pinned Ubuntu environment;
the complete local mjlab suite passed454 tests with41 dependency/native skips.

Corrected execution source: `~/sircus-mjlab/studies/neighborhood-v2`, code commit
`58c1a9f11ab289ff3d81f54cfda3dd44e79bf960`; planned evidence
`~/sircus-mjlab/output/neighborhood-v2`. Supervisor1383189 is a historical launch
identifier. The job is queued cooperatively behind another4090 mjlab study,
with a four-hour queue limit and four-hour execution limit. Its supervisor will
audit and update the existing viewer automatically; no server restart or active
policy change.4070 and Isaac remain untouched.

The interface now displays the earlier audited references separately:

| Board-center height mm | Tilt from horizontal | Gap mm | Ball-center release height mm | Nominal | Fresh gap-only noise | Validated substeps |
|---|---|---|---|---|---|---|
|397.388|36.1548°|321.219|756.107|3/3|20/20|64 only|
|500.000|36.1548°|238.531|776.031|3/3|20/20|64 only|
|440.000|32.0000°|285.781|763.281|3/3|20/20|64 only|

All use the same fixed472.826 mm horizontal board-to-bucket distance and all
captures were rim-assisted. Fresh noise varied gap only with sigma2 mm; this
does not establish robustness to board-height/tilt noise or32/128 substeps.
No setting has passed this study's paired32/64/128 checks yet. The first two
references are prior TRAIN-window confirmations; the440 mm reference is the
old frozen TEST result, now explicitly a known DEVELOPMENT anchor.

## Original launch checkpoint — November 2

Status at launch checkpoint: **queued for the shared4090 allocation**. No new
ball-flight result is claimed yet. The existing35-position study retains its
lock and is not interrupted. The4070 workload and Isaac setup remain untouched.

- [Prespecified method](../../sircus/mjlab/NEIGHBORHOOD_STUDY.md):75 local board
  height/tilt contexts ×17 gaps, captured-context refinement, up to12 selected
  recipes ×23 joint-noise requests paired at32/64/128 substeps. At most2,580
  requested attempts including27 initial reference controls.
- [CPU native preflight](cpu-preflight.json):72/75 board contexts resolved;
  all three exact source joint pairs preserved; gap and local perturbation,
  query identity and out-of-bounds checks passed. This is kinematics/collision
  evidence only; physics and paired timestep outcomes remain pending.
- Execution snapshot: `~/sircus-mjlab/studies/neighborhood-v1`.
  Planned raw evidence: `~/sircus-mjlab/output/neighborhood-v1`.
  Supervisor1198357 /queued child1198358; these are historical identifiers, not
  proof of current liveness. Queue timeout is two hours; execution is capped at
  four hours, with512 MiB output cap and512 MiB free-space floor.
- The finite supervisor will run the independent CPU audit and update the derived
  viewer after native completion. `#heightTiltStudy` is read-only; active inference
  and previous models remain unchanged. Existing viewer process186578 is retained.
- A separate bounded CPU queue-display helper1209570 shows the waiting state until
  the study creates its first real report, then exits. Source is
  `studies/neighborhood-queue-display-v1`; it never acquires a GPU or writes into
  the original evidence folder. No existing services are restarted.

All75 contexts are DEVELOPMENT, including the previously tested440 mm reference.
This task does not train a new model or provide an unseen-layout evaluation.
Historical studies retain their original results/definitions. Final numeric
results and curated evidence require native completion and a matching passed
audit. Raw observations/logs stay outside Git; never substitute queued placeholders
for measurements.
