# Seeded bucket-distance development study

This opt-in study starts from the eight configurations that captured 23/23 at
each of 32/64/128 substeps in the audited height/tilt parent. It searches shot
parameters and tests repeatability. It does not train or activate a policy.

The parent scene, table, robot bases, board dimensions, bucket contact/clearance
geometry, material/mount assumptions, scoring and native physics remain unchanged.
The bucket's 175 mm reported height is still unresolved; the existing provisional
175 mm interior depth plus 8 mm base is retained.

## Placements and geometry

Board XY stays fixed at the parent's measured/provisional coordinates. Two
bucket rays are reported separately; their results are not pooled as a pure
distance-only experiment:

- Original direction: 350, 450, 472.826 (exact parent XY), 550, 600, 640 and 650 mm.
- Diagonal toward the supported lower-left table corner: 550, 650, 700, 750,
  770, 800 and 950 mm. Its unit direction is computed before outcomes from the
  fixed board XY to the bucket-center corner allowed by the full outer radius.

The upright bucket's largest outer radius plus a declared 5 mm table-edge margin
must fit inside the current provisional tabletop. This limits the original ray
to about 648.8 mm and the diagonal to 772.8 mm. Original 650 and diagonal 800/950
are retained requested geometry failures, never silently moved or simulated as
floating buckets. This conservative footprint limit is distinct from robot reach
and from a failed ball flight.

Original-bearing seed contexts retain exact source board joints. A changed
bearing derives a new board branch from geometry-only continuation hints,
independently verifies full flange pose and strict 5 mm common-park clearance,
then freezes it before gap queries. Every release independently passes existing
workspace, full-pose, self/cross-arm, tool, carried-ball, board and bucket checks.
Endpoint admission does not establish a swept trajectory or actuator tracking.

## Prespecified finite search

1. Reproduce all eight exact source configurations three times at 64 substeps
   (24 controls), including the exact source joints. Stop if any fails 3/3.
2. At each of 14 requested bucket placements, test each of the eight source H/T
   settings with 17 gaps: source gap + offsets from −80 to +240 mm in 20 mm steps
   (1,904 requested attempts, including unsupported placements).
3. For supported targets at least 650 mm away with fewer than two coarse-captured
   contexts, add eight-source continuation contexts with H/T offsets
   (−80 mm,−12°), (0 mm,−12°), (+40 mm,−8°). Each gets 14 absolute gaps from
   120 to 640 mm. At most 96 contexts / 1,344 requested attempts.
4. Each captured coarse/extension context receives four transition/promising
   gap-interval midpoints, chosen before refinement outcomes. At most 832.
5. Select an actually captured sample near the center of a widest sampled hit
   run. Keep one context per nearer target, up to two diverse contexts for targets
   at least 650 mm away. At most 15 recipes. No untested midpoint or continuously
   successful interval is inferred from adjacent samples.

Maximum total: **5,589 requested attempts**, including repeated checks below.
Every rejected request stays visible. This is adaptive DEVELOPMENT selection,
not frozen-policy accuracy, held-out generalization, or exhaustive state coverage.

## Paired repeatability

Freeze 3 nominal and 30 fresh Gaussian joint H/tilt/gap perturbations per recipe:
sigma 1 mm height, 0.25° tilt, 2 mm gap; seed 20261206. No clipping or retries.
Local IK must stay within 0.10 rad of the nominal branch on both arms. Reuse the
exact same requests and audited joint arrays at 32/64/128 native substeps per
240 Hz observation. A recipe's 99 requests are not 99 independent randomized
scenes; three nominal repeats share one geometry and resolutions reuse all noise.

Prespecified fine-pair gate: 3/3 nominal and at least 27/30 fresh captures at both
64 and 128, at least 30/33 matching capture outcomes, and at most 5 mm nominal
position disagreement at 350 ms. The all-resolution repeat gate additionally
requires 3/3 nominal and at least 27/30 fresh captures at 32. It adds no coarse
positional-convergence claim. Neither finite gate proves numerical convergence,
physical calibration or future success.

The fresh-allocation v2 adapter records native solver tolerance 1e-8 and the
pinned Warp float32 floor 1e-6 explicitly. No allocated options or parent physics
are overwritten. Board rebound, whole-ball entry, and one continuous second of
retention still define success; rim assistance counts, external assistance fails.
An independent CPU audit rescores every saved native flight and checks all source,
asset/package/option/parent hashes, requests, selection, pairings and gates.

## Runtime and evidence

Use the pinned Ubuntu mjlab environment and RTX4090 UUID only, with the existing
cooperative `~/sircus-isaac/scene.lock`. The 4070 and Isaac installation/workloads
are not changed. The native study is bounded to four hours, output to 256 MiB,
and free storage must stay above 512 MiB; it never deletes another task's files.
The supervisor bounds its own queue/execution and runs the final audit on CPU.

```bash
export CUDA_VISIBLE_DEVICES=GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$HOME/sircus-mjlab/venv/bin/python" -m sircus.mjlab.distance_pipeline \
  --parent-study "$HOME/sircus-mjlab/output/neighborhood-v2" \
  --assets "$HOME/sircus-isaac/assets" \
  --output "$HOME/sircus-mjlab/output/distance-NEW" \
  --gpu-lock "$HOME/sircus-isaac/scene.lock" \
  --display "$HOME/sircus-mjlab/interface-report.json"
```

The existing viewer's `#distanceStudy` section is read-only. It separates bucket
support, board IK, paired-arm endpoints, native flight outcomes and repeatability,
with actual bucket-specific saved trajectories. Static files/derived reports can
be updated without restarting the viewer or activating a checkpoint. Never write
derived files into an audited raw evidence directory. Raw observations, queues,
logs and runtime configuration stay outside Git; curate small metrics and hashes.

## Completed execution and native examples

[Audited results and exact settings](../../baselines/mjlab-distance-20261103/README.md)
record the completed 4090 run at execution commit 9ed20bc: 3,891 requests,
1,387 native releases, and 9,962 independent checks passed. The current solver and
policies were not changed. Farther targets produced tabletop-assisted entries,
which remain failures under the original route contract.

After the study and its full matching audit complete, the display adapter can
append up to four explicitly post-hoc far-miss examples:

```bash
python -m sircus.mjlab.prepare_distance_report \
  --base-display /absolute/current/interface-report.json \
  --study /absolute/completed/distance-study \
  --output /absolute/separate/distance-display.json \
  --include-far-misses
```

This CPU-only transformation checks source record/cache hashes and exact
record-column/geometry associations, reads saved arrays without pickle, and
changes no score, gate or original evidence. It selects one recorded miss per
supported target at least 650 mm away, ranked by whole entry, retention, closest
opening distance and stable ID. Such examples explain failure; they are not an
additional evaluation or an unbiased performance sample. The default is off.
The selected-recipe trace limit retains all 30 possible 64/128 examples; optional
post-hoc misses are additional. Publish a derived report only when the owning
writer has completed, preserving every historical panel and the running viewer.
