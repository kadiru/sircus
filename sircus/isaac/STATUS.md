# Compact farm recording completed — November 2, 2026

[Verified replay](../../baselines/isaac-dark-farm-20261102/README.md) and
[capture instructions](DARK_FARM_FILM.md) describe the native 35-table film,
dark stage, driven arm transitions and edited impact views. The replay produced
162 released board rebounds and 28 captures from 350 attempted recorded actions.
Seven close-ups include two captures and three measured floor arrivals. The
60-second full tour and three shorter playback versions decoded and matched
their checksums on the Mac. This is a replay of training actions with scripted
transitions; full grasp/path learning and physical transfer remain open.

## Workspace feasibility completed — November 1, 2026 (local date)

[Verified results](../../baselines/isaac-feasibility-map-20261101/README.md) and
[protocol](FEASIBILITY_MAP.md): run `feasibility-20261102T012536Z-n32` finished
6,212 attempts  / 3,043 accepted releases on the 4090. Fifteen selected targets;
six passed 30/30 fresh perturbations. Target 23 is outside the direct-reach bound
but captured only 6/30, all non-single-bounce assisted captures. No new RL policy,
continuous workspace coverage or physical validation is claimed.

All raw/summary records, source/evidence hashes and 80 asset hashes/mtimes matched.
The job exited 0. The read-only preview serves `/feasibility.html` from the separate
`feasibility-preview-v1` snapshot on Ubuntu port 18860, forwarded to Mac port 18780. Preview
PID 2672808 remained active at this checkpoint; 4070 process 2653439 was unchanged.
These are point-in-time observations, not deployment instructions. No physical,
Viam or audio changes. Prior sections below are historical process observations.

# Current checkpoint — parallel world completed, 2026-11-01 UTC

Run `parallel-world-20261101T205548Z-n32` used one isolated RTX 4090 PhysX scene
with 32 tables, 64 arms and distinct fixed table-local bucket positions. It finished
1, 090 attempts in 197.3 s: 64 controls, 896 searches and 130 fresh confirmations.
Thirteen targets had selected settings; six passed 10/10 perturbed repeats each.
Confirmation 107/130 captured, all single-board-bounce; 17 dropper IK rejects and 6
released misses. No setting was found beyond 440 mm. This is parameter search,
not new RL training or physical validation.

All 32 fixed-target checks, five per-cell contact probes and the deliberate
foreign-cell isolation test passed. All 1, 090 raw/compact rows, three artifact
hashes and 48 source hashes matched. One-/four-cell reference runs completed;
one borderline control changes outcome in 32 cells, so strict clone equivalence
is not established. Geometry/provisional height status and older baselines are
preserved. See [full evidence](../../baselines/isaac-parallel-world-20261101/README.md)
and [method](PARALLEL_WORLD.md).

Execution source: `~/sircus-isaac/studies/parallel-world-v4`.
Raw outputs: `~/sircus-isaac/output/parallel-world-20261101T205548Z-n32`.
One-cell reference: `parallel-world-20261101T205913Z-n1`.
Four-cell smoke: `parallel-world-20261101T205442Z-n4` (v3; same physics, before
render-history flush). Earlier v1/v2 camera dtype failures remain documented in
local logs; they are not successful trials.

The finite 4090 processes exited. At final inspection 4070 PID 976650 remained
unchanged; no 4090 compute process remained. Read-only preview PID 2262680 serves
port 18860 from `~/sircus-isaac/studies/parallel-viewer-v1/preview.py`, forwarded to
Mac 18780. `/parallel-world.html` shows verified JSON/CSV, actual world renders and
a 121-frame, 30 fps video. These are point-in-time process observations, not permission
to signal these PIDs without checking. No hardware/Viam/audio services were changed.

Next: feasibility-aware batched policy adapter and disjoint spatial evaluation;
retain the full trajectory/pickup and perception boundaries.

---

# Historical checkpoint — spatial grid completed, 2026-11-01 UTC

RTX4090 run `bucket-grid-20261101T193317Z` completed 689 parameter attempts in
21.5 minutes. Three near cells passed 30/30 fresh perturbed shots each; two middle
cells scored 18/30 and 6/30 board-only captures. Four cells had no qualifying
setting in this finite search. This is development search, not new RL training.
See [all settings, uncertainty and verified evidence](../../baselines/isaac-bucket-grid-20261101/README.md).

The additive `sircus-board-route-v1` preserves historical labels and separates
board-only retained captures from exactly-one-board-bounce captures. Five contact
probes, two controls, 689 raw/compact matches, three artifact hashes and 43 source
snapshot hashes passed. Source snapshot: `~/sircus-isaac/studies/bucket-grid-v3`.
Raw evidence remains under the run directory outside Git.

The finite 4090 job exited naturally; no 4090 compute process remained at final
verification. The 4070 compute process976650 remained untouched. Read-only preview
PID2125076 serves port18860, forwarded to Mac18780, from
`~/sircus-isaac/studies/grid-viewer-v1/preview.py`. Final `#bucket-grid` JSON/CSV
match the verified exports. These are point-in-time observations, not PIDs to
signal without checking. No hardware, Viam, audio or physical-dashboard changes.
Source synchronization is not runtime deployment.

Next: resolve far-shot feasibility and validate parallel shot execution before
another long policy run. See [method and compute examples](BUCKET_GRID.md) and
[project state](../../docs/PROJECT_STATE.md).

# Previous checkpoint — 2026-11-01 UTC

The [bucket-distance catalogue](../../baselines/isaac-distance-catalogue-20261101/README.md)
completed 2,081 attempts at six distances, including 50 fresh perturbations per
frozen setting. At 300/400 mm both settings captured 50/50. Farther settings
captured 37–49/50, but many saved paths suggest intermediate table/other rebounds.
The original scorer permits these; the separate warning audit does not change
scores or certify direct paths. No new RL policy was trained.

After completion, the finite 4090 process had exited and no 4090 compute process
remained. The existing 4070 process (PID 976650) remained unchanged. The read-only
preview uses Ubuntu loopback port 18860, forwarded to Mac port 18780. These are
point-in-time observations: recheck actual processes before future deployment.
The immutable executed source is `~/sircus-isaac/studies/distance-catalogue-v1`;
raw evidence remains under `~/sircus-isaac/output/distance-catalogue-20261101T125810Z`.
No physical dashboard, Viam configuration, hardware or audio service was changed.

Next: table/robot contact instrumentation and a versioned direct-path criterion
before further learning. Existing perception remains a separate provisional
pipeline; no perception-driven control was tested here. See [project state](../../docs/PROJECT_STATE.md)
for intervening learning/IK studies. All runtime statements below are historical.

# Isaac foundation checkpoint — 2026-10-30 Pacific

This is synthetic simulation evidence, not physical validation or a trained policy.
Observed runtime state below must be rechecked before future changes.

## Installed and running

- Ubuntu workspace: `~/sircus-isaac`. Dedicated container: `sircus_isaac_4090`.
- Isaac Lab `v2.3.2` / commit `37ddf626871758333d6ed89cf64ad702aef127d0`.
- Actual mounted binary: Isaac Sim `5.1.0-rc.19+release.26219.9c81211b.gl`.
  The reused Docker image tag says 6.0.1; the read-only 5.1 mount determines the actual runtime.
- Private Python3.11 venv; bundled torch2.7.0+cu128. No global Python/driver update.
- Only the RTX4090 is exposed to the Sircus container. Limit: 12 CPU cores, 28GiB RAM.
- Existing `aiworker_pouring_4070` retained its original GPU assignment and start
  time (`2026-10-29T07:16:02Z`). No restart, reconfiguration or workload signal.
- Read-only preview: Ubuntu loopback `http://127.0.0.1:18860`; Mac SSH forward
  `http://127.0.0.1:18780`. Viewer has no robot/simulator command API.
- Continuous scene holds starting joint targets and renders the fixed RGB-D view.
  It does not autonomously pick, drop, bounce or train. Nothing auto-starts at boot.

## Completed verification

Final finite capture: `output/20261031T023110Z/` on Ubuntu (not committed).

| Check | Result |
|---|---|
| Geometry unit tests | 4 passed on Mac and Ubuntu |
| Existing project regression suite | `scripts/test.sh` passed, including physical mocks, numerical/scoring contracts, study UI and the new dependency-light geometry tests |
| Manufacturer URDF → USD import | Both arms imported; 6 arm joints each, plus 2 gripper-proxy joints on arm10 |
| Simulation stepping | 180 rendered frames / 6 simulated seconds at 240Hz physics, 30Hz render |
| Wrist command response | Requested +0.05rad; measured arm10 +0.0499992, arm14 +0.0499654 |
| Attached board | Center (0.670115, -0.099995, 0.396280)m; rigid wrist offset 0.09999996m |
| Board facing | Normal (-0.005913, -0.422787, 0.906210); faces negative-Y bucket, landscape long edge along X |
| Bucket opening/floor | Ball fell through opening; settled center Z=0.0455000m, matching 8mm base + 37.5mm radius |
| RGB-D | 640×480 RGB and axial metre depth; 149,557 valid depth points |
| Camera transform | 35,650 reconstructed points within 2mm of known synthetic tabletop |
| Saved observation | RGB PNG, depth NPY, colored optical-frame PLY, intrinsics/world transform, joint/body states and provenance |

The board's approximately 3.7mm gravity sag reflects this provisional actuator
configuration. Command response is not trajectory collision validation. The
bucket probe is not a complete board-bounce-and-retention success measurement.
Camera points are from rendered visible surfaces; the tabletop test uses known
synthetic geometry and does not claim learned object detection.

The live browser preview was verified with advancing frames and about149,390
valid depth pixels. The final scene holds a ball on the loading area; the finite
smoke placed a separate test ball over the bucket to test its opening/floor.

## Storage and dependencies

At validation, the new workspace used approximately 2.4GB. Root had about 29GB
free; `/data` about 6.3GB free. No existing datasets, caches or logs were deleted.
Reused the large existing simulator/image; no second full simulator download.
Live preview replaces bounded latest files, and timestamped captures are opt-in.
The scene stops without deleting anything if free output storage drops below 3GiB.

`pip check` is **not globally clean**: the existing bundled optional
`nvidia-srl-usd[-to-urdf]` packages advertise missing `usd-core`, an `lxml<5`
constraint versus bundled5.4, and bundled `msal` advertises missing `pyjwt`.
These optional export/cloud paths were not used. Kit's own USD, URDF import,
articulations, camera, Torch and saved-capture inspection were exercised successfully.
We did not install another USD build into Kit to silence dependency metadata.

Startup can take a few minutes on a cold shader cache. Sim5.1's stage/Replicator
cleanup hung on earlier smoke runs; after synchronous capture writes the script
uses the documented `SimulationApp.close(skip_cleanup=True)` exit path. The final
smoke completed and exited cleanly. A lock prevents concurrent Sircus scene runs.

## Still missing

- RL task reset/observations/actions/reward/termination and PPO training.
- Board-bounce + whole-ball entry + continuous-one-second retention scoring in PhysX.
- Collision-checked cooperative trajectories; self-collision currently disabled.
- Grasp/release validation; current gripper is a geometric proxy.
- Compliant suction/board mount, ball parameter calibration and timestep convergence.
- Depth noise, segmentation/localization and held-out perception-driven evaluation.

Bucket diameter measurements and unresolved175mm height interpretation are
preserved. Table, placements, tool offsets, masses and contact parameters remain
provisional. No old simulation score or policy is claimed valid in this backend.

## Dense shot catalogue checkpoint

See [completed settings and robustness results](../../baselines/isaac-shot-catalogue-20261031/README.md). Fixed400 mm distance;579 searched settings,240 initial confirmation trials,100 independent follow-up trials. Recommended starting point H400 mm/gap130 mm/tilt21° passed140/140 clean captures across the tested nominal and narrow perturbation conditions. Other settings were substantially less robust. Simulation search, not new RL or physical calibration. Finite4090 jobs completed; preview read-only;4070 and hardware untouched.
