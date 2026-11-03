# Local height/tilt/timestep study

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
