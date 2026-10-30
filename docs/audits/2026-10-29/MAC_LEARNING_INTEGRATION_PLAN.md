# Sircus learning integration plan — proposed, not implemented

> Repository edition of the October 29, 2026 Mac audit. Runtime findings and source references describe that audit, not current live status. The source was subsequently published to `kadiru/sircus` at `6e1fbf1`, and [Ubuntu CI passed](https://github.com/kadiru/sircus/actions/runs/41006601692). Credentials, raw observations/trials, configuration dumps, screenshots and exact host calibration records remain local. Local evidence paths below identify provenance; they are not files shipped in this repository.

Audit date: 2026-10-29/30 UTC. This report proposes the smallest next implementation milestone from the inspected Mac code; it does not start training, install a simulator, deploy Viam configuration, load a policy or command either physical arm. Detailed current algorithm/source evidence is in the detailed learning appendix in `MAC_SIRCUS_SYSTEM_AUDIT.md`; UI contracts are in `MAC_UI_API_CONTRACTS.md`.

## Accepted experiment

The operator supplies a board center and bucket position. For each shot, arm14 rotates the board about that fixed center and holds still; arm10 releases a ball. A scene-conditioned learned policy chooses board tilt direction/amount, release height and optionally board-local impactU/V. Training and evaluation must cover randomized scenes with whole scenes held out. All policy proposals remain simulation-only until explicit hardware integration and calibration.

The existing code is a useful seed: it already has finite board contact, compliant support, an open tapered bucket, model-limit failures, two-arm endpoint IK, guarded fake-arm Viam replay, immutable trial evidence and a functional physical operator state machine. It does not yet implement the accepted randomized-scene learning problem. The current dashboard Learn release searches one fixed board pose'sU/V/gap; the newer CLI searches one scene'sroll/pitch/gap at fixed world releaseXY. Neither is learned inference over many board/bucket placements.

## Recommended bounded next milestone

**Implement a versioned, headless one-shot learning environment around the current contact model, with a fixed scene dataset/split, a shared outcome evaluator, and offline policy-inference preview in the Mac dashboard.** Keep the physical panel separate. Use this milestone to establish a reproducible baseline before changing the physics backend or selecting a large RL stack.

Acceptance criteria:

1. A seeded reset samples a feasible board center/bucket position from declared bounds and emits an explicit observation. Training, validation and test use disjoint scene IDs; fixed seeds are recorded. Domain parameters and release noise have their own seeded streams. The full scenario set is serialized, not inferred from GUI state.
2. An action selects board normal (tilt direction/amount), height and optionalU/V; board center and bucket remain fixed for that shot. Every action reconstructs an unambiguous world release pose. No hidden baseline action or optimizer answer leaks into policy observations.
3. A complete rollout returns board/contact events and whole-sphere opening entry/exit; classifies miss, clean/rim-assisted capture and bounce-out; evaluates **one-second retention**, separately from the old0.2s flag. Old evidence retains its original outcome definition and version.
4. A baseline fixed action, the existing per-scene numerical oracle and one scene-conditioned predictor are evaluated on exactly the same held-out scenes. Report feasibility, invalid/model-limit rate, success fraction with sample count, error distribution and contact class; include the user-reported lateral-miss case only after its exact scene/artifacts are imported.
5. A checkpoint plus manifest can be loaded for **simulation preview** in the Mac UI; observation/action dimensions, units, physics/outcome version, scene-distribution schema and source fingerprint are checked. The UI displays the proposed pose/action and compatibility errors. Hardware execution is not part of this milestone.
6. One deterministic rollout matches after export/import to Ubuntu within declared numerical tolerances. Independent workers write to unique shard directories. No training worker connects to sfhack10 or reuses a live cloud machine identity.

The first deliverable should be this experiment contract and benchmark, not a promise that a larger simulator produces physically correct bounces. Contact parameters still need measured calibration.

## Proposed observation/action/outcome contract

| Area | Minimum fields and conventions | Existing reusable source | Missing work |
|---|---|---|---|
| Observation | boardcenter/rotation and surface normal; bucket openingcenter/normal/inner radii/depth; ballradius/mass; each tool pose and arm joint state; availability/confidence/source frame; optional RGB-D detections | `scene.json`, FK, physical snapshot, stationary perception outputs | Explicit schema, transforms and masks; general bucket orientation; avoid substituting assumed values for observations |
| Action | unit board normal or azimuth+tilt; vertical gap with exact reference plane; optional board-localU/V; chosen yaw convention; downstream desired board/tool/release poses | `train_orientation_height.candidate`, `board_target`, `release_point` | IndependentU/V mode; fixed-center rotation invariant; configurable tilt bounds; action scaling/validation |
| Feasibility | board/dropper endpointIK, clearance rejection, model-limit reason; native path validation state separate | `two_arm_kinematics.feasibility`; Viam replay | Batch/cached checks and standardized invalid action result; do not mark endpoint feasibility as path safety |
| Rollout | trajectory times/positions/velocities; board state; every board/bucket contact; opening entry/exit; outcome and retention duration | `physics.simulate`, result contacts/frames | Currently frames omit velocity; event records need stable types/IDs; clean/rim-assisted/bounce-out evaluator and1s dwell |
| Metadata | trial/session/sceneID; observation/action/outcome schema; physics backend/version; policy ID/version; split; RNG streams; full contact parameters; source hashes; data origin; calibration ID | `TrialStore`, config/source fingerprint, CLI experiment JSON | Unified schemas between CLI, Viam and physical records; full dependency/tool geometry hash coverage; machine-independent canonical representation |

Keep a bucket-opening pose in the future API even if the first implementation restricts the bucket to upright. Return an explicit unsupported-orientation error until the physics backend supports tilt. Current `bin` usesXY plus floorZ, and its analytic axisymmetric contact assumes the symmetry axis is worldZ.

For the first study, use one action per shot (a contextual decision problem). A Gym-style `step(action)` can execute an entire shot and terminate. Continuous feedback control, moving-board interception and joint torque policies require a different temporal model and should not be implied by that wrapper.

## Minimum code work by dependency

| Order | Proposed change | Reuse / purpose | Preconditions and boundaries |
|---|---|---|---|
|1|Version the experiment schema, units, outcome evaluator and source/dependency manifest|Existing scene and trial schema provide starting structure|Clarify reported175mm height versus interior depth; keep unconfirmed values provisional; retain historical scene/result versions|
|2|Expose a pure headless rollout/reset interface and scene sampler|Wrap `physics.simulate`; separate frontend/native Viam from training|No imports with runtime side effects; don't mutate shared `scene.json`; board center fixed per shot; feasibility bounds documented|
|3|Add1s retention and event taxonomy behind a new outcome version|Reuse bucket contact and whole-sphere containment|Extend horizon after entry; distinguish timeout from failure; numerical convergence and rim/edge cases before training|
|4|Create fixed train/validation/test scene manifests and benchmark harness|Existing seed/noise/freeze-before-test discipline|Hold out whole scene/parameter regions; ensure test scenes are not selected after results; distinct namespaces from older rectangle/bucket archives|
|5|Train/evaluate one small scene-conditioned policy|Per-scene optimizer can generate teacher action labels; legacy affine learner is a simple baseline idea|Start with supervised action regression or outcome/dynamics regression plus bounded search; RL is optional, not required for one-shot action selection|
|6|Add a simulation inference/preview API and UI model selector|Existing viewer, scene form, results and replay|Checkpoint version/checksum checks; display source/calibration/compatibility; no physical inference route by default|
|7|Optional worker sharding on Ubuntu and checkpoint return|Pure simulation, uniqueTrialStore IDs and JSON manifests|Record platform/versions, assign seeds/shards, no sharedmutablefiles/cloudidentity; exact GPU model/VRAM/driver not yet established|
|8|Later physical integration using measured calibration|Existing physical load/drop state machine, pose teaching, recording, Stop|World calibration, ball-in-tool offset, support parameters, board/bucket localization, board orientation planner and operator-approved motion envelope required|

## Backend options and distributed training assessment

| Option | Present status / proposed adaptation | Practical assessment |
|---|---|---|
|Current numerical model|Implemented pure-Python shot function plus NumPy/SciPy IK; noGPU/vector API|Smallest baseline path. Multiple independent processes can run isolated rollouts after unique output/seed handling. Python threads are not a reliable CPU-speedup strategy for its Python inner loop; Viam controller currently permits one task. Benchmark before infrastructure work. Two GPUs do not accelerate this code automatically.|
|Gymnasium|Not installed/integrated in current dependency spec; noEnv class|Wrap shot-levelreset/step and explicit spaces. Gymnasium's API separates `terminated` and `truncated`; that distinction fits capture/miss versus simulation limit/timeout. [Official Env API](https://gymnasium.farama.org/api/env/).|
|MuJoCo|NoMJCF assets, model converter, dynamic joints/contact adapter or validated contact settings in repo|A plausible next backend for a compact headless articulated scene. Convert850 geometry/frames, give ball/free joint and board support dynamics, calibrate contact, compare outcomes against the baseline. Its official Python package provides native bindings and viewer; the macOS passive viewer has its own launcher requirement. Do not assume JSON Viam geometry is directly loadable. [Official Python documentation](https://mujoco.readthedocs.io/en/stable/python.html).|
|Isaac Sim / Isaac Lab|NoUSD assets, Isaac task, GPU rollout pipeline or chosen installed stack|Optional Ubuntu study after hardware/software compatibility and rendering needs are known. Current official Isaac Lab docs distinguish backend choices and optional Isaac Sim; pin a tested release/backend rather than equating Isaac Lab with one fixed simulator. GPU/driver/VRAM suitability must be checked on that desktop. [Official installation/backend requirements](https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/installation.html).|
|CustomRL / learned dynamics|No current neural model, optimizer state, replaybuffer or policy inference service; legacyxArm6 has a small affine fit|Use common records and split manifests so supervised forward-dynamics/outcome models and RL can be compared. A scene→action model can be trained from optimizer-generated labels, but validate on unseen scenes and preserve teacher failures/censoring; do not label every returned optimizer action successful.|
|Ubuntu twoGPUs|User-provided capacity, not inventoried by this audit|Start independent CPU shards for current code. Add distributedGPU training only after selecting a framework and benchmarking bottlenecks. Exchange code commit, immutable scene manifest, worker results and checkpoints; keep Mac hardware credentials and Viam part ownership separate.|
|Mac checkpoint inference|No current arbitrary model loading; only saved fixed-scene JSON actions and CLI replay|Add model artifact descriptor, normalization, observation schema, supported distribution, checksums and inference implementation. UI can preview proposed action, then optional native fake replay. One physical button does not become learned closed-loop control automatically.|

Changing the physics backend does not calibrate the real ball/board interaction. Fit or bracket spring/damping/friction/rebound parameters from controlled observed drops and quantify sensitivity. Preserve a backend abstraction so identical scenes/actions can be evaluated in both engines; expect contact outcomes to differ and report those differences.

## Viam data, sequences and artifacts

Reuse `TrialStore`'s immutable local archive plus disposable upload copies, train/test/native phases and synthetic-origin labels. Standardize orientation-CLI files into the same envelope rather than introducing another hidden format. Keep raw trial evidence separate from model artifacts; checkpoint manifests should cite input dataset IDs, split manifests, simulator/outcome versions and hash of weights.

The physical adapter already has local image/joint recording, optional `capture-control` when discovered, direct DataClient uploads on Save and `CreateSequence` after timestamped image/joint uploads (`physical_trials/viam_adapter.py:313`,`:420`). Upload acknowledgment is distinct from independently verified cloud availability. Capture/exposure/state RPCs are not hardware-synchronized. These are useful foundations, but not a validated automatic perception reward or a unified simulator-to-hardware learning dataset.

There is no implemented training-cluster job scheduler or model-registry publish/load workflow in the audited code. Viam remains useful for robot API access, selected-path validation, scene display and trial artifact transport; the current contact integrator and optimizer execute in Python on the host, not automatically in Viam Cloud. Do not claim a configured Data service means an RL learner is running.

## Issues to resolve before trusting study results

- **Current0.2s success is not future1s retention.** Version the evaluator and never relabel earlier10/10 as1s captures.
- **Signed-zero fingerprint false invalidation:** saved bucket scene roll-0.0 versus current+0.0 changes hash although sources and numeric scene match. Normalize finite numeric values under a new fingerprint version and test equivalence; retain old hashes/artifacts and explicit migration evidence. Do not bypass guards globally. Include all behavior-changing geometry and versions in future contract hashes.
- Current fixedworldXY orientation search cannot correct every lateral miss; enableU/V or let releaseXY derive from chosen impact coordinates. The separate Ubuntu0/100 is a reason to expose the missing action freedom, not evidence to pool with10/10 fixedscene captures.
- NativeViam replay remains required for path feasibility; endpointIK alone cannot clear both arms/tools/board/ball/table/bucket.
- Current rectangular and tapered engines, legacyxArm6 scripts, and physical state machine have duplicated logic. Pick one canonical numerical engine for the new study; preserve legacy code/results rather than silently altering them.
- Scene truth, kinematics-derived poses, manual measurements and RGB-D estimates need explicit provenance/confidence/calibration IDs. Neither a plausible 3D overlay nor one stationary image establishes camera-to-world calibration.
- No present model captures board bending, spin or complete robot compliance. Simulated peak forces are not validated hardware loads. The AS1200 hold test demonstrates observed attachment in those tests, not identified contact dynamics.

## Planning summary

Reuse the Mac UI/operator recorder, current scene/assets, reduced-order physics, endpointIK, fake-onlyViam replay and immutable evidence. Add a scene-conditioned experiment contract, randomizedscene sampler,1s outcome evaluator, Gym-style shot wrapper, dataset splits and inference-preview UI. Benchmark a small predictor against fixedaction/per-scene-search baselines. Only then choose whether MuJoCo/Isaac orGPU rollout work is justified. No such changes were implemented by this audit.
