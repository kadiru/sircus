# IK audit and controlled distance sweep

This is a bounded diagnosis of the completed privileged PPO study, not another
training run. No checkpoint weights change. Numerical IK endpoints are not
collision-safe paths, physical grasping, or hardware validation.

## Questions and method

1. **Was Arm14 actually unreachable?** Revisit all 62 board-arm failures from the
   two old test evaluations and 16 solver-success controls (first eight valid
   cases from each run). This is retrospective analysis of already-used test
   scenes, not a new held-out score. Use the same flange position plus board-normal
   residual and strict combined residual norm < 1e-4. Preserve free in-plane spin.
   First reproduce the original two starting guesses and 80 evaluations each;
   then try the same guesses with 300 evaluations, then a midpoint and seven
   deterministic random guesses with 200 each. Seed: 20261101. Unresolved means
   no acceptable solution was found, not proven geometric infeasibility.
2. **Do good shots exist within the current action range?** Keep board center
   `(0.670, -0.100, 0.400)` m fixed. Place the bucket 0.20, 0.25, 0.30, 0.35 and
   0.40 m away along negative Y. On the same RTX4090, run both frozen selected
   policies once with original IK and once with extended board IK at each layout.
   Also run the unchanged warm fixed action once per layout.
3. Search an explicitly labeled diagnostic grid: tilt 12/16/20/24/28/36/45 degrees
   × release gap 60/100/150/200/250 mm; heading offset zero (faces the bucket).
   Select by capture, then shaping reward, and run three fresh repeats of the
   selected action. This is **per-layout search, not learned inference**. The grid
   has 35 actions per distance; the entire sweep has 215 shots. No noise or layout
   jitter is added to the repeats, so three repeats are a consistency check, not
   a robustness estimate.
4. The 250 mm gap failed Arm10 IK in the sweep. A separate offline audit of all
   five gap targets retains full flange orientation and tests the original
   100-evaluation solve, a 400-evaluation solve, then eight additional starts with
   300 evaluations. It does not change or rerun the recorded grid results.

Success remains the original versioned PhysX criterion: measured board contact
and rebound, whole ball entry, and one continuous second inside the tapered
bucket. Invalid IK remains in denominators. Bucket motion uses the same kinematic
compound body as contract v3. Geometry is unchanged: the reported 175 mm bucket
height still has unconfirmed overall/interior interpretation; depth, bottom,
mount offset and dynamics remain provisional. Original training and evaluation
records are preserved.

## Reproduction

Use the existing pinned Isaac Sim 5.1 / Isaac Lab 2.3.2 environment and generated
manufacturer assets. The pure IK audits require numpy and scipy, not a GPU.
Paths below are templates; no credentials are required. Set `SIRCUS_ISAAC_SOURCE`
to an immutable copy of this directory. Do not launch a second simulation while
another owns the same scene/output workspace.

```bash
python.sh ik_diagnostics.py RUN_1000 RUN_100 \
  --assets /workspace/assets --output /workspace/output/ik-audit
python.sh dropper_diagnostics.py \
  --assets /workspace/assets --output /workspace/output/ik-audit/dropper-audit.json
python.sh launch_scene.py --headless --enable_cameras --device cuda:0 \
  --distance-sweep --sweep-policy RUN_1000/selected.pt \
  --sweep-policy RUN_100/selected.pt \
  --ik-audit /workspace/output/ik-audit/ik-audit.json
python3 diagnostic_report.py SWEEP_RUN \
  --output /workspace/output/ik-distance.json --conclusion 'Interpret the verified results here'
```

The actual run used a scene lock and 30-minute timeout. The report finalizer rejects
an incomplete run or missing grid cells. Source and checkpoint SHA256 digests are
saved in each run contract. `shots.jsonl`, source snapshots, logs, joint targets
and model files remain host-local; only aggregates are published to Git.

## Interface and remaining boundary

The read-only preview's `#ik-distance` panel shows solver recovery, both frozen
policies under both solver budgets, per-distance search outcomes, selected fresh
repeats, and tilt/height heatmaps. It cannot issue robot or training commands.
The preview fetches `/ik-distance.json`; existing studies remain below it.

See [completed results](../../baselines/isaac-ik-distance-20261031/README.md) for
findings and the next bounded implementation step. Neither the training solver
nor a physical controller is silently upgraded by this diagnostic. Quasi-perception
and another long training run remain deferred.
