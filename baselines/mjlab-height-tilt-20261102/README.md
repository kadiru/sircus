# Local height/tilt/timestep study

## Status checked November 3, 2026

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
