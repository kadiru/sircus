# Expanded 35-table sweep study

This development experiment searches the same 7 × 5 distinct tabletop bucket
positions, using corrected native MuJoCo/mjlab physics. It learns no policy and
does not establish continuous-table coverage. The interface displays one selected
workcell and the complete per-table evidence table.

## Completed result — November 3, 2026

[Audited compact evidence and full-precision settings](../../baselines/mjlab-grid35-expanded-20261103/README.md)
cover all 35 targets: 17 selected recipes, 12 unresolved and 6 clearance-admission
blockers. Parent and follow-up audits passed separately. The parent froze 17
recipes (51/51 nominal, 495/510 perturbed); one 22/30 recipe was replaced by a new
bearing/joint configuration scoring 3/3 and 28/30. Sixteen exact parent configurations
retain their original frozen rates. Current per-recipe recorded rates are 28–30/30,
not a new common-bank test or learned-policy success rate.

The diversified pass audited 13,312 proposals at 13 clear targets. All 101 eligible
broad/inverse poses reached native physics, out of 6,656 proposed; another 1,563
native search shots were local proposals. No additional target qualified. Finite
geometry/path/flight limits remain explicit. Both finite workers exited; the
existing solver catalogue, policy routing and other services were preserved.

## Search and selection

Each target retains the original scene, materials, ball, bucket dimensions,
landscape board mount, strict endpoint checks and fixed common home. A bucket
that blocks the common home remains explicitly blocked. In particular, a fixed
base/bucket overlap cannot be cured by another arm configuration. Finite failure
is not a proof that no shot exists with other staging or shot families.

The action is board X/Y/Z, bucket-facing tilt and the vertical release gap. The
ball is released above the board center. Heading faces the bucket; horizontal
release velocity and lateral impact offsets are outside this family.

| Parameter | Search bounds |
|---|---|
| Board center X | 300–1,040 mm |
| Board center Y | −500–350 mm |
| Board center Z above table | 120–600 mm |
| Tilt from horizontal | 8–60° |
| Ball-bottom release gap above board center | 50–650 mm |

Measured native board centers can differ slightly from requested centers because
of numerical FK. Release Z means the **ball center**, and is nominally board Z
plus gap plus the 37.5 mm ball radius. Neither value is the gripper flange height.

1. Save the 12 qualified historical target recipes as explicitly old evidence.
   Freshly retest those exact recipes rather than counting their previous repeats
   as this study's work.
2. Baseline search at other targets: up to 64 CPU proposals and 16 native search
   releases. Historical captured geometries from the same target lead, followed
   by translated/inverse-ballistic and broad deterministic proposals. All poses
   pass current geometry checks and all labels come from new native simulation.
3. Expand unresolved targets: up to 1,024 new CPU proposals and 128 native search
   releases. Previously checked values are excluded. Exact historical poses,
   interleaved local refinements and seeded broad proposals cover additional
   height, gap, tilt, radius and bearing combinations. Local warm starts include
   measured baseline captures and near misses; inherited success is never used.
4. Up to three baseline or six expanded captured candidates receive sampled
   common-home path admission and 3 nominal plus 20 perturbed repeats. Qualification
   requires 3/3 and at least 18/20 captures. Each cohort is bound to its exact
   values and joint branches; the selected rates are never totals from several
   configurations. Other attempted cohorts remain in the raw work ledger.

The historical hint input is hash-checked: 349 captured search geometries across
22 targets, plus the qualified catalogue. Only geometry, joints and source
identity are retained. This is outcome-informed development search, not a test
of unseen-layout policy inference.

Local refinements interleave all selected warm starts:

- Gap: ±1, 2, 4, 8, 16 and 32 mm.
- Board Z: ±5, 15 and 30 mm.
- Tilt: ±0.5, 1, 2 and 4°.
- Board/bucket radius: ±10, 25 and 50 mm.
- Bearing around the bucket: ±3, 7.5 and 15°.
- Joint parameter perturbations and the existing broad/inverse-ballistic pool
  fill the remaining budget. Out-of-bounds proposals are omitted, never clipped.

## Independent frozen repeats

After search finishes, freeze every selected setting in `frozen-selection.json`.
Run **3 nominal plus 30 fresh perturbed requests** for each frozen setting, using
new per-cell seeds. Never choose a replacement from these outcomes. The final
rates remain visible even if they are worse than qualification.

Both qualification and final variations use independent Gaussian board X/Y
noise of 1 mm, board Z and gap noise of 2 mm, and tilt noise of 0.35°. Rejected
endpoint or local-path requests remain in the requested-repeat denominator.
Saved-reference, qualification, search and frozen-validation outcomes are kept
separate. Clean versus rim-assisted quality is tagged with its recorded cohort.

Success still requires a board rebound, whole-ball entry and one continuous
second inside without external assistance. Rim-assisted captures count. Neither
an ODD event nor a geometric or analytic trajectory is a capture label here.

## Additive diversified follow-up

The first completed expanded cell exposed an acquisition bias: although 1,024
poses were planned, the native budget was exhausted after 160 CPU checks and
128 local shots. No broad proposal reached physics. The actually tested gap
range was only 81–195 mm. Three entries involved a table rebound and correctly
failed the intended board-to-bucket route. Planned ranges must not be described
as tested coverage.

`grid_diverse_study` is a separate, versioned follow-up, added after that finding.
It preserves the original run and requires its completed, independently passed
audit before starting. All 35 old outcomes are archived as saved evidence, with
exact original request/result sidecars, IDs, provenance and parent hashes. None
of those trials count as new work.

Targets are fixed before new outcomes: every unresolved position and each
previously selected recipe whose final set was missing/blocked or below 3/3
nominal or 27/30 perturbed captures. These previous final findings now inform a
new development round; their original scores remain unchanged.

For each clear target, audit **all 1,024 distinct proposals first**. At least half
of the pool comes from broad/inverse proposals. Joint local sweeps include gaps
60–640 mm and tilts 12–60°, jointly with the board positions/heights. Select up to
128 eligible native attempts with a 64-slot broad/inverse quota when enough are
eligible, then normalized farthest-point geometric diversity. Flight outcomes
do not determine this selection. Save planned, checked, eligible and actually
executed family counts and parameter ranges separately.

Current qualification remains 3 nominal + 20 perturbed requests. Compare up to
six captured candidates and choose a qualified recipe by most fresh captures,
then most clean fresh captures, then native search rank. Stop early only for
3/3 nominal and 20/20 fresh with all 20 clean. This rule was strengthened before
follow-up trials started after one parent recipe scored only 22/30 in its frozen
check; the parent's selection and results remain unchanged. Freeze every
newly selected recipe before a separate new 3 + 30 repeat set. Unchanged parent
recipes retain their previous frozen rates, explicitly labeled as previous-study
evidence; they do not inflate current work or current validation counts. A failed
follow-up can leave a fragile saved fallback visible with its actual old rates.
The interface verifies cell, bucket and configuration identity before showing any
inherited rate.

```bash
"$HOME/sircus-mjlab/venv/bin/python" -u -m sircus.mjlab.grid_diverse_study \
  --reference "$HOME/sircus-mjlab/output/corrected-grid-v1" \
  --assets "$HOME/sircus-isaac/assets" \
  --gpu-lock "$HOME/sircus-isaac/scene.lock" \
  --parent-study "$HOME/sircus-mjlab/output/planner-grid35-v1" \
  --parent-audit "$HOME/sircus-mjlab/audits/planner-grid35-v1.json" \
  --parent-source "$HOME/sircus-mjlab/studies/planner-grid35-v1" \
  --output "$HOME/sircus-mjlab/output/YOUR_NEW_DIVERSE_STUDY"
```

Use a frozen source snapshot and the same pinned environment variables as below.
`--prepare-only` validates and archives parent evidence without GPU work. Missing
or altered parent evidence fails closed. The optional results-view
`--parent-study-dir` defaults to the exact compact copy inside the follow-up.

## Reproduction

Use the pinned mjlab environment and generated assets/reference described in
[SWEEP_PLANNER.md](SWEEP_PLANNER.md). The runner binds all native dependencies,
its own source, the report contract, assets, packages, hints, grid and protocol.
Changing those identities requires a separate output directory.

```bash
export PYTHONPATH="$HOME/sircus-mjlab/studies/YOUR_FROZEN_SNAPSHOT"
export CUDA_VISIBLE_DEVICES=GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
"$HOME/sircus-mjlab/venv/bin/python" -u -m sircus.mjlab.grid_study \
  --reference "$HOME/sircus-mjlab/output/corrected-grid-v1" \
  --assets "$HOME/sircus-isaac/assets" \
  --gpu-lock "$HOME/sircus-isaac/scene.lock" \
  --output "$HOME/sircus-mjlab/output/YOUR_NEW_STUDY"
```

`--prepare-only` checks input identities and publishes the complete initial
35-row snapshot without new physics. An identical interrupted run can resume
completed stages from its ledger. Unique output directories preserve old runs.
Do not edit an active snapshot, overwrite historical outputs or bypass the GPU
lock. This runner selects only the 4090 and never signals other workloads.

The separate `grid_study_server` adds `--study-dir` to the sweep server's launch
arguments. It serves `planner_grid_study` inside `/api/study`. Read-only
`GET /api/planner-grid/result/<cell-id>` verifies report/source/configuration and
result identity before replay; it cannot start a search. The ordinary explicit
search button retains its original separately gated behavior.

## Evidence and limits

The run keeps its request/result ledger, exact proposals, endpoint rejects,
candidate-specific nominal/fresh IDs, all local-path proofs, native arrays and
checksums outside Git. Compact verified settings, counts and audit summaries may
be published as a dated baseline. Neither cached evidence nor absent trials are
presented as new successful drops.

Native arm endpoints are prescribed before every substep. Geometric setup paths
are sampled checks; this does not demonstrate physical grasping, actuator
tracking or realistic impact compliance. Contact/mount properties and the
reported 175 mm bucket-height interpretation remain provisional. Isaac, physical
Viam resources, sound and unrelated workloads are independent of this experiment.
