# Exact shallow-board candidate at four far targets

This new DEVELOPMENT experiment tests the user's exact **380 mm board-center
height, 28° tilt from horizontal and 386.8 mm release gap** at the existing
650/700/750/770 mm corner-ray targets. It preserves the completed lower-board
study and its zero-capture results. No physics, scoring, material, scene or
active policy is changed. The bucket's reported 175 mm height interpretation
remains unresolved; its provisional interior-depth model is unchanged.

The gap is vertical clearance from the released ball bottom to board-center
height. With ball radius37.5 mm, ball-center release is804.3 mm above tabletop.
The exact rounded386.8 mm request differs by0.024338 mm from the previous
CPU-only386.824338 mm proposal. Native full-pose IK/clearance admission is required
again; a nearby CPU result cannot establish a successful ball flight.

## Prespecified bounded experiment

- Keep shared board XY470.875779/−277.893401 mm and the four exact parent bucket
  centers unchanged. Board heading faces each bucket. Only distance changes
  across the four discovery requests.
- Reproduce24 exact parent controls first, three per seed, at64 substeps. Stop on
  any mismatch. Their captures are separate from far-target outcomes.
- Test the exact candidate once per far target at64 substeps. Retain rejects and
  failures; do not clip, refine, resample or search other parameters.
- Only an actual captured discovery shot is selected for repeats, at most one
  per target. Each receives3 nominal plus30 fresh joint height/tilt/gap variations
  (sigma1 mm/0.25°/2 mm, seed20261208). The same requested poses and audited joint
  branches are reused at32/64/128 substeps. Rejected variants remain failures.
- Maximum424 requests:24 controls+4 discovery+4×33×3 repeats. If all four miss,
  only28 requests and64-substep native evidence exist. A one-off miss is not a
  probability estimate or proof the setting can never capture.

The unchanged fine-pair gate requires3/3 nominal and≥27/30 fresh captures at64
and128,≥30/33 matched outcomes, and≤5 mm nominal displacement difference at350 ms.
The all-resolution gate additionally requires the capture thresholds at32.
Capture agreement does not imply full trajectory convergence or physical validity.

Strict capture requires a board rebound, whole-ball entry and one continuous
second retained. Rim-assisted captures count. Table/floor/robot-assisted entries
fail. Clean versus assisted/marginal capture quality is separate from capture;
an ODD impact or an attractive trajectory does not determine success.

## Reproduction and boundaries

Use the pinned mjlab environment and exact audited `distance-v1` parent
fingerprint6d94edc505db7201f603456116303cd14acd9dde26e364ea295b1739c78f8499.
The immutable source snapshot, packages/assets/options, complete requests and
native caches receive an independent CPU audit. The runner reuses the same
strict native geometry and ball-flight engine as the earlier lower-board study.

```bash
export CUDA_VISIBLE_DEVICES=GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$HOME/sircus-mjlab/venv/bin/python" -m sircus.mjlab.shallow_pipeline \
  --parent-study "$HOME/sircus-mjlab/output/distance-v1" \
  --assets "$HOME/sircus-isaac/assets" \
  --output "$HOME/sircus-mjlab/output/shallow-NEW" \
  --gpu-lock "$HOME/sircus-isaac/scene.lock" \
  --display "$HOME/sircus-mjlab/interface-report.json"
```

The cooperative lock preserves other4090 work. Never bypass it, signal4070 jobs,
or alter Isaac installation/runtime. The existing viewer gains a separate
read-only`#shallowBoardStudy` without service restart, solver switch or policy
activation. Raw joints, sampled observations, logs and runtime config stay out
of Git; only compact synthetic results and provenance hashes are curated.

## Where this sits in learning

Current work is feasibility and numerical repeatability, not a new RL update.
Prior learned selectors and their held-out evaluations remain historical,
versioned evidence. At closer targets, repeated working shots exist; at650–770 mm,
the completed220–360 mm grid had no captured native release. This experiment
tests one geometry-admitted shallower family before committing more RL compute.

If it works: measure local gap/tilt windows and fresh robustness, then expand
distance/direction and collision-aware arm paths. If it misses: use saved flight
and contact diagnostics to choose the next bounded grid. Only after a useful
feasible bank exists should a scene-conditioned policy be trained and compared
with a uniform/analytic baseline on disjoint, frozen bucket positions. Camera
perception, arm transit/tracking and arbitrary operator constraints need their
own tests; privileged endpoint success does not establish those capabilities.
