# Local height/tilt search and paired timestep sensitivity

This opt-in DEVELOPMENT study expands around three previously confirmed shots.
It does not train or deploy a policy. All old scenes, results, physics adapters,
models and the active viewer solver remain unchanged.

Board XY and bucket XY are fixed to the previous study. The board faces the
bucket, 472.826 mm away horizontally. Board dimensions are measured; mass,
mount and contact response remain provisional. The bucket retains its reported
175 mm height and unresolved interpretation, modeled provisionally as 175 mm
interior depth plus an 8 mm base.

## Prespecified search

The reference centers (board height / tilt from horizontal / gap) are:

- 397.388 mm /36.1548° /321.219 mm.
- 500 mm /36.1548° /238.531 mm.
- 440 mm /32° /285.781 mm.

Each contributes five heights at ±40/20/0 mm and five tilts at ±4/2/0°:75
distinct contexts. Each gets17 gaps at 10 mm spacing within ±80 mm of its
reference gap:1,275 coarse requests. Source-family proposals and geometry seeds
are distinct: geometry chooses the nearest verified H/tilt anchor deterministically
before outcomes, freezes a board branch and independently solves dropper gaps.
Strict5 mm clearance, full flange FK, self/cross-arm, tool, ball, board, table,
bucket and floor checks remain. Unresolved requests stay visible as failures.

At64 substeps, contexts with a coarse capture receive six transition/broad-interval
midpoint refinements (at most450). Selection keeps all three known references
and up to nine additional distinct contexts, diversified in height/tilt. Each
new recipe uses an **actually captured sample** near the center of the widest
sampled hit run, rather than an untested midpoint. These are adaptive search
results, not policy evaluations or certified continuous success intervals.

The former frozen TEST shot at440 mm is explicitly a known DEVELOPMENT anchor
now. Its historical test result remains unchanged. Future training needs fresh
validation/test contexts; this search provides no new unseen-layout score.

## Repeatability and timestep intervention

Freeze3 nominal +20 fresh joint perturbations per selected recipe:
height sigma1 mm, tilt sigma0.25°, gap sigma2 mm; seed20261205. Re-solve each local
pose from its exact nominal branch, requiring no more than0.10 rad deviation in
either arm. Bounds/IK/continuity failures are retained, without clipping/retry.

Reuse these exact requests **and identical audited joint arrays** at32,64 and128
native substeps per240 Hz observation (7,680 /15,360 /30,720 Hz). An isolated
fresh-allocation adapter inherits the existing flight/scorer; no allocated model
or old environment is modified. Actual native and Warp options are saved and
checked, including float32 device representations. Geometry/material/scoring and
gravity compensation are unchanged. Each shot has a four-second native horizon.

The separate finite engineering gate requires3/3 nominal and≥18/20 fresh captures
at both64/128,≥21/23 paired capture-status agreements and≤5 mm maximum nominal
position disagreement at350 ms. Rates at32 are also reported. This does not prove
mathematical timestep convergence or real contact calibration. Geometric transit
and actuator tracking are outside this held-endpoint experiment.

Initial native controls repeat all three references three times at each timestep
(27 requests). Only64-step controls must reproduce3/3 before search. Coarser/finer
control disagreement is a sensitivity result, not silently corrected. Maximum
total requests:27+1,275+450+828=2,580. Every native release is independently
rescored from saved240 Hz positions, velocities and categorized contact forces.

## Execution and evidence

Use the pinned Ubuntu mjlab Python3.10 environment, read-only existing assets,
4090 UUID and cooperative `~/sircus-isaac/scene.lock`. Keep4070 and Isaac processes
untouched. The finite runner caps its output at512 MiB, requires512 MiB free and
stops after four hours; the supervisor bounds queue+execution and then audits on
CPU. Code/source/package/asset/parent evidence hashes precede new native labels.

```bash
export CUDA_VISIBLE_DEVICES=GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$HOME/sircus-mjlab/venv/bin/python" -m sircus.mjlab.neighborhood_pipeline \
  --comparison-study "$HOME/sircus-mjlab/output/window-sac-v1" \
  --confirmation-study "$HOME/sircus-mjlab/output/window-confirmation-v1" \
  --assets "$HOME/sircus-isaac/assets" \
  --output "$HOME/sircus-mjlab/output/neighborhood-NEW" \
  --gpu-lock "$HOME/sircus-isaac/scene.lock" \
  --display "$HOME/sircus-mjlab/interface-report.json"
```

The existing viewer's `#heightTiltStudy` section uses only derived display data.
Static files/derived reports can be updated without server restart or policy
activation. All75 contexts, requested/native denominators, exact selected shots,
paired timestep outcomes and saved native trajectories remain distinguishable.
Raw observations/logs stay outside Git; only curated synthetic metrics/provenance
are published. Never write derived display files into an audited evidence folder.

While waiting for the shared allocation, the supervisor publishes an explicitly
queued placeholder with no invented measurements. The optional standalone
`neighborhood_queue_display --study PENDING_DIR --display VIEWER_REPORT` helper
supports older launch snapshots; it is CPU-only, bounded, writes derived display
data only and exits when the real study report appears. It never steals a lock,
stops another job, or writes into an audited evidence directory.
