# Workspace feasibility before longer RL training

This is a finite, privileged-state **development search**, not policy training or a held-out generalization score. It asks whether two articulated UF850s can hold useful endpoint configurations and produce a board-to-bucket capture across a wider tabletop grid. `feasibility_contract.py` defines a new fingerprinted task. Historical parallel-world, PPO, distance and rectangular-container results remain unchanged.

## Geometry and actions

There are 32 table-local bucket centers: X = −100, 100, 300, 500, 700, 900, 1100, 1400 mm; Y = −720, −470, −220, 230 mm. Every workcell has two arms, one attached board, one bucket and one dynamic ball. Buckets remain fixed throughout the study. The table is 1800 × 1250 mm, and all bucket outer footprints fit its bounds. Sampling these points does not cover all intervening positions.

The original board/release line is (670, −100) mm. Expanded search adds sixteen board XY positions from X = 520/620/720/820 mm and Y = −300/−150/0/150 mm. Configurations with board-to-bucket horizontal center separation below 280 mm are excluded as a conservative footprint guard. This is not arm-path collision certification.

Board heading is prescribed toward the bucket; the five searched parameters are board X, board Y, board-center height, vertical release gap, and tilt from horizontal. The ball starts vertically above board center, at zero velocity. Gap means **ball-bottom release Z minus board-center Z**, not distance along the surface normal. Ball-center Z adds the provisional 37.5 mm radius. Downward flange target adds a further provisional 95 mm tool offset. The arms are initialized at IK endpoints, then allowed to settle; grasping, pickup and transit are not learned or validated.

The 200 × 300 × 5 mm landscape board and 1337.31 mm base spacing are preserved. Bucket inner top/bottom diameters are 177/150 mm, outer diameters 200/158 mm. Its **175 mm reported height still has unresolved overall-height versus interior-depth interpretation**; this simulator assumes 175 mm interior depth plus an 8 mm bottom. Rigid mounting, friction, restitution, mass and gripper geometry are provisional. No physical performance claim follows from these results.

## Finite protocol

1. Run six measured attribution/isolation probes: bucket floor, rim, table, floor, robot and foreign-cell collision isolation. A new table-probe point avoids the wider bucket grid; older experiments retain their original default probe positions.
2. Audit direct-drop endpoint IK at ball-center heights 300/450/600 mm. Separately derive a conservative horizontal outer bound from URDF link-origin lengths plus the 95 mm ball/tool offset. Only this geometric bound can establish out-of-reach XY in the model; failed numerical IK is unresolved. This does not certify a collision-free direct-drop path.
3. Fixed-XY baseline: 28 settings per target, 896 attempts.
4. Expanded XY: eight proposals per admissible board point, 3,912 attempts. Heights 300/350/400/450 mm are balanced across low gap 80–140 mm and high gap 160–280 mm. Low-gap strata use a frictionless restitution/ballistic range estimate only to rank proposals; high-gap strata sample a seeded broad tilt/gap grid (tilts 10–42°). The approximation supplies no success labels. It cannot establish impossibility or eliminate the broad half of the portfolio.
5. At most 27 local refinements per target around its best valid candidate: height ±25 mm, gap ±20 mm, tilt ±2°. Unsuccessful valid candidates may seed refinement. Maximum 864 attempts.
6. Select by valid board-only capture, single board bounce, clean entry, retention and legacy shaping reward. Save and hash settings **before** all repeats. No result from confirmation changes the selected action.
7. Three same-setting repeats, three paired repeats with the same heights/tilt but original board XY, and 30 seeded local perturbations for each selected target. Perturb board X/Y by ±3 mm, height/gap ±2 mm, tilt ±0.5°; heading is recomputed toward the fixed bucket. Maximum 1,152 attempts. Endpoint rejects remain in denominators.

The overall cap is **6,824 attempts**; targets with no selected setting have no confirmation trials. Fixed-XY and expanded searches have unequal budgets, so raw success totals do not isolate the effect of extra actions. The frozen old-XY ablation is a separate small paired check, not a reoptimized baseline.

Capture uses the existing versioned scorer: board rebound, whole sphere entry, and one continuous second retained. Partial exits reset retention. `sircus-board-route-v1` additionally rejects detected table/floor/robot/unattributed contacts during flight. Rim-assisted entry counts; single-board-bounce and clean entry are reported separately. Per-shot endpoint gates remain ≤2 mm center error, ≤0.5° board-normal error, joint speed ≤0.005 rad/s and measured stability over 250 ms. Self-collision checking is disabled in the inherited articulation asset: successful endpoints are **not** collision-certified motion plans.

## Reproduce on the isolated Ubuntu GPU

Use the pinned Isaac Sim 5.1 / Isaac Lab 2.3.2 environment and generated UF850 assets described in [README.md](README.md). The host's generic Python is not a substitute for the bundled interpreter. Preserve the existing runtime and use an immutable source snapshot; make only the intended GPU visible to its container. The current Sircus container exposes the 4090, not the unrelated 4070 workload.

```bash
export SIRCUS_ISAAC_SOURCE=/workspace/studies/your-feasibility-snapshot
timeout 900 flock -n /workspace/scene.lock \
  "$SIRCUS_ISAAC_SOURCE/python.sh" "$SIRCUS_ISAAC_SOURCE/feasibility_map.py" \
  --headless --device cuda:0 --smoke

timeout 7200 flock -n /workspace/scene.lock \
  "$SIRCUS_ISAAC_SOURCE/python.sh" "$SIRCUS_ISAAC_SOURCE/feasibility_map.py" \
  --headless --device cuda:0
```

Run as the container user that owns/writes the study output, following that installation's established launcher. The script refuses to begin or continue batches below 3 GiB free space. Search is headless without RGB rendering, avoiding raw image queues. Scene lock plus finite timeout prevents concurrent scene jobs. Source snapshots, raw JSONL, generated USD, logs and host paths stay outside Git.

The run produces `contract.json`, raw `trials.jsonl`, `summary.json`, `frozen-settings.json`, and hashes. Top-level `feasibility-map.json` is progress, not verified completion. After successful completion:

```bash
python feasibility_report.py --source /path/to/completed-run \
  --executed-source /path/to/immutable-source --dest /path/to/compact-export
```

Use a NumPy/SciPy environment for the offline exporter. It checks source/artifact hashes, exact action schedules, frozen selection, repeat seeds, scoring/settling invariants, and every raw row and replay trajectory before writing `report.json` and `settings.csv`. Publish those atomically as `feasibility-map.json` and `feasibility-settings.csv` in the read-only preview output directory. Keep all previous studies.

## Interface and interpretation

`/feasibility.html` provides a clickable tabletop map, exact selected geometry, separate fixed/expanded/refinement/confirmation counts, direct-reach audit, confidence intervals and CSV. Green means 30/30 fresh perturbations; amber indicates failures in a selected setting's repeats; gray means pending or unresolved. A 30/30 result has a Wilson lower 95% bound of about 88.6%, not guaranteed reliability. Those trials vary only the listed local parameters, not uncalibrated contact properties.

The replay uses actual saved PhysX ball samples in top and flight-plane projections at half speed. It does not fabricate robot motion or substitute for the native Isaac videos. No physical, cloud, audio or command API is exposed.

Before another long PPO run: identify useful action regions and remaining endpoint failures, create fresh spatial splits, and train a scene-conditioned model with board XY/height/tilt and drop gap actions. Keep model inference separate from per-scene optimization, and retain a simple searched or analytic reference. Grasping, transit, randomized contact properties and quasi-perception remain separate validation stages.
