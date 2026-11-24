# Sircus in mjlab

November 24: [expanded deduplicated library and supervised selector](../../baselines/mjlab-expanded-selector-20261124/README.md) completed **44/50 captures**, matching the current hybrid planner's **44/50** on the same fresh positions. Two gains/two losses; mean decision time **10.62 vs 11.06 seconds**. All 118 native64 drops, controls and the 139,349-check audit completed. The library contains 26,789 distinct native settings; training also includes setup-only evidence and balances bucket positions. **Keep the current live planner.** [Protocol](EXPANDED_SELECTOR_STUDY.md) · UI `#expandedSelectorStudy`. No service restart, 4070 or Isaac change; about 3 GB of verified redundant copies reclaimed, with unique evidence preserved.

November 24: [board-first scoring and corrected-label comparison](../../baselines/mjlab-board-first-20261124/README.md) permits any later ball contacts after the first confirmed board rebound, while retaining whole-ball one-second capture and setup/path checks. Original labels remain intact. Original neural-corpus re-score: **3,331 → 3,344 captures (+13, zero lost)** across 5,150 saved drops. Same-model refit tested on 50 fresh positions: **original 38/50; retrained 37/50**, 83 test drops plus 18 controls, 36,090 checks passed. Keep the live planner. [Scoring contract](SCORING_BOARD_FIRST.md) · [study protocol](BOARD_FIRST_STUDY.md) · UI `#boardFirstStudy`. The separate 60-source historical pass accepts **19,620 → 19,985** of 37,111 saved native drops (+365, including 242 later-robot-contact cases and 123 older table-only cases; zero lost); no physical, driven-arm or perception validation, service restart, 4070 or Isaac change.


November 23: [higher-gap and tilt test at one arm-free missed position](../../baselines/mjlab-clear-shot-range-20261123/README.md) found **0/11 new captures** at bucket (1193.0, 147.4) mm. Eight flights hit a robot; three cleared the arms but missed. Of all 49 checked settings, 30 failed the exact full-pose wrist-reach screen and eight failed endpoint-clearance or pose checks; zero cells timed out. Unchanged source: **0/3**, without robot contact. All **9/9 preservation captures**, **18 controls** and **10,993 audit checks** passed across 41 native64 releases. UI: `#clearShotRange`. [Protocol](CLEAR_SHOT_RANGE.md). The second unresolved bucket was not tested. No retraining, live planner promotion, restart, 4070 or Isaac change; Saved-flight diagnosis puts the three clear shots **175–193 mm short at rim height**; all eight contacts matched board-arm link 3. The proposed next change is board position/height while keeping release reachable; it remains untested.

November 23: [adapting nearby successful shots](../../baselines/mjlab-nearby-shot-transfer-20261123/README.md) recovered **three of five known missed bucket positions**. Of 20 adjusted settings at each target, **16 / 0 / 10 / 16 / 0** captured; all 42 passed **3/3 fresh repeats (126/126)**. All 15 unchanged originals missed. The other two targets still had robot contact in every search drop. All **259 releases**, both control blocks and the **56,065-check independent audit** completed at 64 substeps. UI: `#nearbyShotTransfer`; [protocol](NEARBY_SHOT_TRANSFER.md). This is development recovery, with no retraining, live planner promotion, service restart, 4070 or Isaac change.

November 23: [concurrent board-pose search](../../baselines/mjlab-parallel-board-20261123/README.md) completed. **All five prior misses stayed 0/3 with each method; the successful control stayed 3/3.** The alternate worker checked 69 setups and admitted 23 arm configurations, but none predicted sufficient ball-to-arm clearance. No shot was replaced. Mean full request increased **28.19 → 31.52 seconds**. All 54 native drops and 45,635 audit checks completed. Keep the current planner; next, try nearby verified successful settings as additional starting candidates. UI: `#parallelBoardStudy`. No retraining, restart, 4070 or Isaac change.

November 23: [guarded board fallback correction](GUARDED_BOARD_FALLBACK_STUDY.md) completed. **Both planners captured at the same 15/20 fresh positions** (12 direct, three table-assisted); separately, each captured **3/3** at the previously regressed target. Guarded fallback activated once and replaced no shots. Mean full planning request increased **12.94 → 15.05 seconds**. All 64 native releases completed; 71,175 audit checks passed. Audited results: `#guardedBoardFallback`. Keep the existing live planner; this correction showed no capture gain. No retraining, restart, 4070 or Isaac change.

November 22: [bounded board-position fallback comparison](BOARD_FALLBACK_STUDY.md) completed on 50 fresh positions. **Current planner 41/50; fallback 40/50**, with zero gains and one loss. Mean full planning time rose **12.87 → 14.39 seconds**. All 98 test drops, 18 controls and 152,638 audit checks completed at 64 substeps. The separate remaining-target heading sweep captured 0/13. Audited results: `#boardFallbackStudy`. Keep [the checked hybrid planner](STALLED_REFERENCE_STUDY.md) as the live default. No retraining, service restart, 4070 workload or Isaac setup change.

Earlier November 21: [live fast-planner results](INTERACTIVE_FAST_RESULTS.md) show **7/10 captures** versus **8/10** for the original planner on ten new bucket positions, with mean HTTP planning waits of **21.16 versus 46.12 seconds**. All 20 attempts and the 17-trace independent native64 audit completed. Fast declined three setups; original released at all ten. [Fast interactive planning](INTERACTIVE_FAST.md) was the default at that checkpoint, with the original available as an explicit longer-wait fallback. [Acceptance protocol](INTERACTIVE_FAST_ACCEPTANCE.md). No model retraining, contact/scoring, 4070 or Isaac change.

Completed: [30-second planning-budget comparison](PLANNING_BUDGET_RESULTS.md) on **100 fresh bucket positions**. The frozen learned selector captured **75/100** versus the current planner's **69/100**, gaining ten positions and losing four. All 172 test drops, 18 controls and 87,307 audit checks completed at 64 substeps. Median decision time was 30 seconds for both; worker lifecycle was about 31.2 seconds and excludes parent evidence handling. Results: `#planningBudgetStudy`. Next: an explicit bounded-selector UI trial; no automatic planner switch, service restart, 4070 or Isaac change. [Protocol](PLANNING_BUDGET_STUDY.md).

Earlier: [clearance-triggered wider search](../../baselines/mjlab-clearance-expansion-20261116/README.md) captured **42/50** fresh positions versus **40/50** for the current planner. It gained two captures and lost none; robot-contact shots fell from seven to six. All 100 drops, 18 controls and the 40,635-check audit completed. The UI shows results at `#clearanceExpansionStudy`; no model retraining or active-planner switch. [Protocol](CLEARANCE_EXPANSION.md).

[Remaining-six protocol and learning workflow](FIXED35_SIX.md): bounded alternative-pose search, compiled-collider audit, preserved recipes and independent native64 verification. No UI publication or training in this follow-up.

Latest: [farther bucket-facing shot tilt/drop-gap results](../../baselines/mjlab-yaw-tilt-gap-20261109/README.md).
At fixed 562.434 mm distance and 449.818 mm board height, ten recipes captured
all three fresh repeats. Keeping original 30.3307° tilt and adding 20 mm gap
recovered 3/3; rotated baseline stayed 0/3. All 49 proposals, 30 confirmations,
nine references and 18 controls are accounted for: 106 requests / 92 releases,
14 finite IK declines, 50,615 checks passed. No new grid/repeat robot contact.
[Protocol](YAW_TILT_GAP.md) · [handoff](../../docs/MJLAB_YAW_TILT_GAP_20261109.md).
UI destination: `#yawTiltGap`; audited results and a 55.8-second recorded movie
are verified, including full decode and browser playback. Historical files and
the viewer are preserved; the handoff records an externally changed fountain
process identity at the final check. Development tuning only; no policy retraining
or promotion.

Earlier: [bucket-facing gripper rotation results](../../baselines/mjlab-gripper-yaw-20261109/README.md).
On two inspected collision cases, yaw reduced robot-contact shots from 6/6 to
0/6 and improved capture from 0/6 to 3/6. The nearer case captured 3/3; the
farther case still missed. All 36 requests and 18 controls completed; 18,330
independent checks passed. Matched videos and exact poses: `#gripperYaw`.
[Protocol](GRIPPER_YAW.md). No new policy score or default behavior change.

Earlier: [completed seven-hour-budget comparison](../../baselines/mjlab-coverage-training-20261109/README.md).
New imitation and selected PPO both captured 28/70 (40%); no net PPO gain.
All 2,757 requests completed in 235.1 worker minutes. No policy promotion.
[Protocol](COVERAGE_TRAINING7H.md).
Fresh search on105 TRAIN positions, imitation and two PPO seeds, validation on35
positions, then four frozen policies on70 new TEST positions. Up to2,757 native
requests with a seven-hour pipeline deadline,64 substeps and unchanged v3 scoring.
All methods share19 geometric starting poses, including two audited Case00 poses
reclassified as development evidence. The previous policies retain their weights.
Audited results and all 70 position outcomes: `#coverageTraining7h`.
This is a new experiment, not evidence of improved capture until it completes.


Earlier: [Case 00 tilt/drop-gap results](../../baselines/mjlab-case00-tilt-gap-20261109/README.md).
70 proposals / 42 admitted / two discovery captures. At board height 460 mm,
**26° and 32° tilt with 251.819 mm gap each captured 3/3 fresh repeats**;
both used rim assistance without robot contact. The original 28° setting
captured 0/3. Independent audit 27,547 checks and all 18 controls passed.
Full matrix, exact settings and saved native64 replay: `#case00TiltGapStudy`.
[Frozen protocol](CASE00_TILT_GAP.md). No new training, policy activation,
128-substep/noise claim, 4070 or Isaac changes.

Earlier: [joint height/tilt admission search and native64 drops](../../baselines/mjlab-table-height-tilt-20261108/README.md).
Of 120 fixed-grid candidates for cases 00/01/18/26, 58 passed the unchanged
admission checks. All 58 bounced the board but none entered or captured; 42
had robot contact. All 18 controls and 36,371 independent checks passed.
No candidate qualified for fresh repeats. The read-only `#tableHeightTiltStudy`
panel preserves every declined candidate and exact admitted configuration.
[Frozen protocol](TABLE_HEIGHT_TILT.md). No new training or policy activation.

Earlier: [table-assisted PPO failure diagnosis and tilt-only correction](../../baselines/mjlab-table-training-diagnosis-20261108/README.md).
All 31 released misses lacked whole-ball entry; first post-board contacts were
table 16 / bucket 8 / robot 7. Four setup declines split into two finite IK searches
and two fixed-base clearance checks. The bounded 42-request test captured 0/12
original requests and 0/12 changed requests: all 32° corrections were rejected
by board-arm IK. Both control gates and 12,557 independent checks passed.
UI: `#tableTrainingDiagnosis`. Next is admitted height/tilt configurations,
not another PPO run or a relaxed clearance rule. [Protocol](TABLE_TRAINING_DIAGNOSIS.md).

Latest task definition: [table-assisted scoring v3](SCORING_TABLE_ASSISTED.md)
permits table contact and awards the normal capture bonus after one continuous
second inside. The old calibrated containment kernel and physics guards are
unchanged. New scorer/reward modules and a native environment adapter are
explicit opt-ins; historical experiments, Isaac and active policies are preserved.
[Saved-shot rescoring completed](../../baselines/mjlab-table-assisted-scoring-20261108/README.md):
8,037 native trajectories, 4,409 → 4,532 accepted captures (+123), 64,883 checks.
UI: `#tableAssistedScoring`. No retraining or new native trials.

Latest: [case 01 horizon-only replay](CLEARANCE_HORIZON.md) extends one predicted
clearance window while keeping the exact shot, joints, forecasts and native64
scoring unchanged. [Completed results](../../baselines/mjlab-clearance-horizon-20261108/README.md):
short window missed 3/3 with robot contact; long window withheld all three. All
12 controls and 44,237 independent audit checks passed. This improves detection
of a known bad candidate, not capture. No training or policy promotion. A separate
[historical example](../../docs/MJLAB_TABLE_ASSISTED_EXAMPLE_20261108.md) confirms
board → table → bucket retention, excluded by the unchanged strict scorer.
UI: `#clearanceHorizon`.

Latest: [50-position failure diagnosis](../../docs/MJLAB_GAP_FLIGHT_DIAGNOSIS_20261107.md)
separates 18 flight failures (robot9 / bucket6 / table3 first subsequent contacts)
from eight setup rejections. Four retained balls failed the complete route after
robot contact. The new `#gapFlightDiagnosis` panel shows concrete examples and
all 50 requests. The [prespecified +60 mm gap correction](../../baselines/mjlab-gap-flight-correction-20261107/README.md)
completed all 30 native requests: original **0/9** complete captures versus
corrected **9/9**, all with recorded rim-band/bucket contact and none clean;
all 12 controls behaved as expected. Robot
contact after board rebound fell from 9/9 to 0/9. Independent audit passed 46,625
checks. These three post-hoc development cases do not change the original 24/50
result or establish generalization. No new training or policy activation;
the 4070 and Isaac remain untouched.

Latest: [flight-miss diagnosis](FLIGHT_RESIDUAL_DIAGNOSIS.md) of the completed
flight-consistent selector comparison: 17/30 complete captures, nine native
misses and four setup rejections. Eight misses never entered; one was retained
after a disallowed robot contact. All nine failed layouts had negative predicted
opening margins across their 16 proposals. A [paired gap-coverage ablation](GAP_COVERAGE.md)
compared unchanged versus two wider gap proposals at 50 fresh layouts, with no
model fitting or automatic policy switch. Both captured **24/50 (48%)**;
clean captures were 13 original versus 11 wider-gap. The separate v2 audit passed
**1,441,075 checks**. [Results and preserved audit history](../../baselines/mjlab-gap-coverage-20261107/README.md).
UI: `#flightResidualDiagnosis`.

Latest: [PPO versus imitation diagnosis](POLICY64_DIAGNOSIS.md), with
[audited paired examples and 96-slot intervention results](../../baselines/mjlab-policy64-diagnosis-20261106/README.md).
Five of seven lost captures were setup rejections; board-height-only restoration
recovered all five once. The original two native misses repeated at 64. No new
training or policy activation. UI: `#policy64Diagnosis`.

Completed [fixed-64 success-bootstrap/PPO study](POLICY64.md): historical TRAIN-only
successes seeded an actor, followed by 1,024 fresh PPO requests and frozen comparison
at 50 new bucket positions. [Results and checkpoints](../../baselines/mjlab-policy64-20261105/README.md):
prior 19/50 captures, nearest success 18/50, imitation 20/50, selected PPO 18/50.
PPO did not improve capture rate. All 1,284 requests passed the independent audit.
Historical 128 results and compatibility guards remain unchanged; the existing
interactive solver has not switched to the new checkpoint.

Latest diagnostic: the [bounded 650 mm impact-point study](IMPACT650.md) completed
189 requests / 146 native releases in 544.3 worker-seconds. Discovery produced five
rim captures and zero clean; all five selected recipes captured 3/3 at 64 but 0/3 at 128.
No recipe qualified. Both control gates and 54,829 independent checks passed.
[Audited geometry/results](../../baselines/mjlab-impact650-20261105/README.md) and
[read-only UI](http://127.0.0.1:18781/#impact650Study) remain separate from historical
policy benchmarks. No new PPO/noise or physical work was started.

The [matched current-ball sweep/PPO benchmark](SWEEP_RL_BENCHMARK.md) completed
all six workers. Each method/seed spent 1,024 TRAIN requests from the same
17-recipe prior; frozen formal test scores are sweep **67/216 (31.0%)** and PPO
**58/216 (26.9%)**. The strict 1 µm wall predicate is contact-sensitive, so these
results do not establish an ordinary capture or adaptation winner. The
[checksummed evidence and six selected artifacts](../../baselines/mjlab-sweep-rl-benchmark-20261103/README.md)
passed the corrected 902,252-check audit: 7,440 total requests / 3,738 native
releases. Setup + TRAIN wall averaged about 35 minutes per worker for either
method; full reservation averaged about 52 minutes and sampled VRAM 762 MiB.
The new [read-only result view](http://127.0.0.1:18789/#sweepRlBenchmark) is separate
from existing services; no policy was activated. Contact calibration and a
versioned scorer/reward/contract are required before the next comparative study.

Earlier opt-in [conditional gap-window/SAC experiment](WINDOW_STUDY.md): one fixed
development bucket, 24/6/8 disjoint height/tilt contexts and equal1,008-request
training budgets. It compares adaptive search plus a supervised capture predictor
with terminal continuous SAC, preserving rejected requests and frozen evaluation.
This does not replace the existing viewer policy or claim all-table coverage.
Completed: predictor23/184 frozen tests versus SAC0/184; only3/8 test board
contexts resolved. Two separately repeated wider-window centers captured3/3
nominal +20/20 gap-noise trials. [Audited results and exact settings](../../baselines/mjlab-window-sac-20261102/README.md).

The [corrected-contact spatial map](CORRECTED_GRID.md) completed: 35 DEV layouts,
460 native search releases, 87 nominal repeats and 580 fresh requests. Seventeen
settings at 12 positions passed fixed-home paths and the 3/3 plus ≥18/20 gate.
Both CPU audits passed. This supports preparation of a limited corrected-physics
selector pilot; existing trainers still need a matching adapter. No new PPO or
policy switch occurred. [Final evidence and exact settings](../../baselines/mjlab-corrected-grid-20261102/README.md).

The [IK-stability follow-up](../../baselines/mjlab-ik-stability-20261102/README.md)
fixes query-order dependence and preserves all 16 saved-source branches in its
audit. A larger 16-start cold budget also recovers 16/16. Another 10,240 native
throws at the ten remaining positions found direct captures at five, but none
passed independent final reliability qualification. No additional PPO was run.
[Stable solver protocol](IK_STABILITY.md).

The [targeted bank expansion](../../baselines/mjlab-expanded-bank-20261102/README.md)
adds one 9/10-confirmed template after 15,360 search throws, followed by 4,096 PPO
throws. On 100 fresh positions × 3: learned 188/300 versus uniform 182/300; the small
capture gain is uncertain, while clean captures improve 145 versus 76. Shared
original candidates retain identical cached IK branches. [Protocol](EXPANSION.md).

The [interactive learning interface](web/README.md) and [two completed learning
studies](../../baselines/mjlab-learning-20261102/README.md) extend this contact
comparison. Bucket dragging, positional locks, frozen prediction, explicit search
and native trajectory verification are available. [Method and limits](LEARNING.md).

This comparison runs the two-arm bounce task with mjlab's MuJoCo Warp backend and scores the recorded ball/contact traces using the existing Isaac-independent Python scoring code. The starting scene and selected shots come from the completed Isaac cylinder-farm study. This is a contact-model and endpoint replay comparison. The existing Isaac setup provides the reference implementation.

## Runtime

Pin `mjlab==1.6.0` in a separate Ubuntu environment. The tagged release accepts Python 3.10–3.13 and PyTorch 2.7 or newer, and declares MuJoCo and MuJoCo Warp 3.11 dependencies. It differs from the moving `main` branch. The experiment uses only the RTX 4090 selected by its GPU UUID. Runtime packages, generated models, logs and full observations belong outside Git. [Tagged dependencies](https://github.com/mujocolab/mjlab/blob/v1.6.0/pyproject.toml)

The GPU simulation is created with `mjlab.sim.Simulation` and `SimulationCfg`, using a compiled MuJoCo model. The checked run uses1920 Hz physics and240 Hz shared scoring. Reset state by writing the existing `sim.data.qpos[:]` and `qvel[:]` buffers and calling `sim.forward()`. Keep buffer identity stable because CUDA graphs retain their addresses. [Simulation source](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/sim/sim.py)

Bucket placement uses a world-child mocap body and per-world `sim.data.mocap_pos`/`mocap_quat` values. MuJoCo Warp caches the collision transforms of static geoms when data is created. Changing an expanded `model.body_pos` field can therefore move the reported body pose while leaving its static collision geometry at the template location. Check the bucket base's `geom_xpos` against its requested position and run a direct-drop contact probe before comparing shots. [Static-geometry update implementation](https://github.com/google-deepmind/mujoco_warp/blob/main/mujoco_warp/_src/smooth.py)

## Matched task

Both engines use the measured 300 × 200 × 5 mm board, the same provisional ball properties, and the same straight-sided tapered bucket. The bucket retains the reported 175 mm height and its unresolved interpretation. The current model assumes 175 mm interior depth plus an 8 mm bottom, putting the rim at 183 mm above the table. Bucket walls remain 64 separate convex wedges surrounding an open cavity.

The fresh paired comparison preserves selected bucket XY coordinates, frozen paired arm configurations, board geometry, release position, initial zero ball velocity and physics timestep. Source joint states were measured at the end of the original throws, so the original outcome labels only stratify the diagnostic sample. Fresh paired engine results use the same prescribed initialization in both engines. It reuses `ShotScore`, `EntryQuality` and `BoardRoute` from `sircus/isaac`. A successful route requires a measured board rebound, whole-ball entry and one continuous second inside. Rim-assisted captures count. A partial exit resets retention, and intervening table, floor, robot or unidentified contact disqualifies a direct route.

MuJoCo and PhysX have different contact solvers. MuJoCo's `solref`/`solimp` contact response is not numerically interchangeable with PhysX restitution and contact-offset settings. The report therefore needs the exact contact settings, contact-probe results and trajectories alongside capture counts. Compare first impact time, rebound velocity, landing/entry position and retention. Any tuned contact setting must be labeled with its calibration trials; evaluate it on separate shots.

Timing has two useful measures. Physics-only throughput excludes installation, compilation, CUDA graph warmup, rendering and Python scoring. End-to-end trial time includes reset, stepping and the shared scorer. Report both, with environment count, device, package versions and repeated measurements. GPU parallelism is a potential benefit; speed remains an experimental result for this task.

## Perception and the interactive showcase

mjlab provides RGB-D cameras and pinhole raycasting. Depth images can be back-projected into point clouds using their camera intrinsics and extrinsics. Segmentation can supply simulator object IDs for the planned quasi-perception stage, while the actual perception pipeline learns to estimate the ball, board and bucket from sensor data. Ground-truth measurements remain available for evaluation. [Camera documentation](https://mujocolab.github.io/mjlab/v1.6.0/source/sensors/rgbd_camera.html)

A future interactive task should accept bucket XY and optional constraints on either arm. The state contains the bucket pose, arm joint states, board geometry and any fixed constraints. An action proposes a board center/orientation and release position/height. A feasibility stage masks unreachable or conflicting actions and passes valid ones through IK and measured settling before a simulated throw. The result should show the predicted configuration, observed trajectory and score. If no feasible configuration is found, show that outcome and allow the user to relax a constraint.

Start by varying bucket positions with both arms free. Then introduce constraints during training, including fixing board center, restricting board height or tilt, or fixing the dropper's XY/height. Supply the constraint mask and values to the policy so it can condition its answer. Use disjoint training, validation and test layouts and frozen-policy evaluation. Finite training grids establish performance at those layouts; success across most table positions needs held-out continuous-position tests and a coverage map.

The current goal is entirely simulated. ExPloy's export pipeline is optional if portable inference artifacts become useful. The earlier UF850/Viam adapter comment concerned executing policies on physical robots; that adapter is outside this showcase's requirements.

## Reproduce the checked comparison

Run from the repository root on the Ubuntu machine. The native assets in `$HOME/sircus-isaac/assets` and the referenced `vendor/xarm_ros2` meshes are required. Install into an isolated Python 3.10 environment using `sircus/mjlab/requirements.txt`. GPU index 0 was verified as the RTX 4090 on this host. The runner refuses a visible set containing multiple GPUs or a device other than the 4090.

```bash
python3 -m venv "$HOME/sircus-mjlab/venv"
"$HOME/sircus-mjlab/venv/bin/python" -m pip install -r sircus/mjlab/requirements.txt
CUDA_VISIBLE_DEVICES=0 "$HOME/sircus-mjlab/venv/bin/python" -m sircus.mjlab.run \
  --manifest baselines/mjlab-comparison-20261102/benchmark.json \
  --assets "$HOME/sircus-isaac/assets" \
  --output "$HOME/sircus-mjlab/output/a-new-run" \
  --substeps 8 --solref .004 .15
```

The checked run uses 1920 Hz physics with eight substeps per 240 Hz scoring interval. Forces are aggregated by category over the eight substeps. Both engines apply the same recorded joint positions and zero joint velocities before every inner step. Rendering is excluded from timing. Full scoring/state reads are included; physics-kernel-only speed was not measured.

`calibrate.py --fine` reproduces the independent normal-drop calibration scan. Its fixed 5 mm board, train drops at 100/300/600 mm and held-out drops at 200/400 mm do not use bucket success outcomes. The selected `solref=(.004,.15)` meets the 2 mm penetration limit at 1920 Hz. Restitution still varies materially with impact speed/phase. [Calibration and checked results](../../baselines/mjlab-comparison-20261102/README.md)

The fresh Isaac counterpart is `isaac_pair.py --headless --device cuda:0 --substeps 8 --benchmark BENCHMARK --output NEW_RESULT --assets /workspace/assets`. Copy it and `contract.py` beside the existing pure Isaac source in a new snapshot, then launch with the existing `python.sh` inside the isolated 4090 container. The checked counterpart is `sircus-isaac/output/mjlab-paired-fine-v1.json`; its source snapshot is `studies/mjlab-paired-v1`. Original source, runtime configuration and study outputs retain their paths.

```bash
"$HOME/sircus-mjlab/venv/bin/python" -m sircus.mjlab.compare \
  --manifest baselines/mjlab-comparison-20261102/benchmark.json \
  --first ISAAC_RESULT --second MJLAB_RESULT \
  --output comparison.json --markdown comparison.md
```

`render_replay.py` creates a common native MuJoCo rendering of both engines' recorded ball trajectories at the fixed arm configurations. The video is a physics-trajectory replay; it does not show an Isaac application viewport.

## Table-assisted training follow-up

See [the explicit v3 short-training protocol](TABLE_ASSISTED_TRAINING.md) for the
review of Isaac feasibility findings, TRAIN-only relabeling, bounded PPO and
frozen old/new policy comparison. Historical v2 contracts remain unchanged.
