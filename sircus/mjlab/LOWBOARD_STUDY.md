# Lower board, moderate tilt and far buckets

This separate DEVELOPMENT study keeps the completed distance study, existing
solver, policies, physics, scoring and scene geometry unchanged. It tests the
same supported corner-ray bucket centers at 650, 700, 750 and 770 mm from fixed
board XY. The board faces each bucket. No new RL or arm-transit learning occurs.

## Prespecified search

- Board-center heights: 220–360 mm above tabletop, every 20 mm.
- Tilt from horizontal: 28, 32, 36 and 40 degrees, independently crossed with
  height. This avoids coupling every lower board to a lower tilt.
- Five ball-center release levels are derived from the unchanged dropper outer
  workspace prior: 5, 15, 30, 50 and 80 mm below its ceiling. For each height,
  gap = release-center height − board-center height − ball radius. No clipping.
- The cheap-prior ceiling is approximately 819.324 mm at the existing board XY:
  `0.364 + sqrt(0.842^2 - X^2 - Y^2) - 0.095 - 0.09`, in metres.
  This is not native IK admission, an actual maximum safe height, or a physical
  calibration. Every requested shot separately passes the original native full
  pose, self/cross-arm, tool/ball, board/bucket and 5 mm clearance checks.
- A geometry-only nearest original seed (H/40 mm, tilt/4 degrees, stable ID tie)
  is assigned before outcomes. A new bucket-facing board branch is verified and
  frozen. Original joints are numerical hints for a changed target.

The four targets receive 128 H/tilt contexts and 640 requested gaps. All rejects
remain in the denominator. Eight additional exact source contexts reproduce
24 parent control drops (three each at 64 substeps); any mismatch stops the study.
The control bucket is separate from the four far-target search tables.

Each context with an actual capture gets four tested-interval midpoint queries.
At most two diverse, actually captured settings per far target are then selected
for repeats. Finite maximum: **1,968 requests** =24 controls +640 coarse
+512 possible refinements +792 possible repeat requests. No hidden retries,
resampling or untested midpoint is promoted as successful.

## Repeats and scoring

Each selected setting receives three nominal plus 30 fresh joint H/tilt/gap
variations: sigma 1 mm /0.25 degrees /2 mm, seed 20261207. The exact same requested
poses and audited joints are reused at 32/64/128 native substeps per 240 Hz saved
sample. Nominal repeats share one geometry; paired timesteps are not independent
randomized scenes. Variations cannot leave the selected joint branch by more
than 0.10 rad and are never clipped or retried.

The unchanged fine-pair gate requires 3/3 nominal and ≥27/30 fresh at 64 and 128,
≥30/33 matching outcomes, and ≤5 mm nominal position difference at 350 ms.
The all-resolution gate adds the same capture thresholds at 32. Neither gate
proves convergence or real-world behavior. If no capture is found, no recipe is
invented and no 32/128 dynamics are claimed from unused engine allocations.

Success still requires board rebound, whole-ball entry and one continuous second
inside. Rim-assisted captures count; table/floor/robot-assisted entries fail.
The provisional 175 mm interior plus 8 mm bucket base is unchanged; whether the
reported 175 mm means overall height or interior depth remains unconfirmed.

## Provenance and execution

Parent: completed `distance-v1`, fingerprint
`6d94edc505db7201f603456116303cd14acd9dde26e364ea295b1739c78f8499`.
Parent evidence hashes and all exact controls are checked. All parent source
hashes must match except one exact, versioned old→new hash pair for the read-only
distance display adapter published after its execution; physics/IK/scoring have
no exception. Assets, packages, effective native/Warp options, requests, saved
caches, selection, pairing and every flight receive an independent CPU audit.

Use the pinned Ubuntu environment, RTX4090 UUID and shared cooperative lock.
The finite supervisor bounds its own queue/runtime and storage; it never kills
another task, deletes its files, or touches the 4070 or Isaac installation.

```bash
export CUDA_VISIBLE_DEVICES=GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$HOME/sircus-mjlab/venv/bin/python" -m sircus.mjlab.lowboard_pipeline \
  --parent-study "$HOME/sircus-mjlab/output/distance-v1" \
  --assets "$HOME/sircus-isaac/assets" \
  --output "$HOME/sircus-mjlab/output/lowboard-NEW" \
  --gpu-lock "$HOME/sircus-isaac/scene.lock" \
  --display "$HOME/sircus-mjlab/interface-report.json"
```

The separate read-only `#lowBoardStudy` panel retains the original distance and
height/tilt results. It shows requested/rejected endpoints, native outcomes,
exact settings, repeated gates and actual saved native examples. Post-hoc
development illustrations are labeled and do not change counts or selection.
Updating static/derived display files needs no service restart or policy switch.
Do not write derived reports into raw evidence. Credentials, raw observations,
joint arrays, logs and runtime configs remain outside Git.
