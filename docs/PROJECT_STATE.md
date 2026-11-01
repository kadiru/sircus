# Project state

## Completed dense fixed-plane catalogue — October 31, 2026

- [Settings and evidence](../baselines/isaac-shot-catalogue-20261031/README.md):495 coarse +84 refined settings at400 mm bucket distance.39 clean captures,89 assisted/marginal captures,354 released misses,97 endpoint rejections. Six frozen settings each received20 nominal and20 perturbed repeats.
- Identical repeats all passed, but perturbation results varied from1/20 to20/20. Strongest: board-center H400 mm, gap130 mm, tilt21°, release ball-center Z567.5 mm (bottom530 mm; provisional flange662.5 mm). It then passed100/100 fresh independent perturbed trials, all clean. Total140/140 for this setting across nominal and narrow H/gap±2 mm, tilt±0.5° variations. No guarantee or physical calibration.
- New additive quality2/1/0 distinguishes clean/assisted/missed shots while retaining board-bounce + whole-ball +1second capture. Invalids remain separate. Bucket contact filter passed floor/rim/outside probes. Clean entry does not require exactly one board contact.
- `#shot-catalogue` shows exact heights, all six confirmation sets, a separately matched100-trial check, heatmaps and downloadable579-row CSV. Historical studies remain intact.919 total attempts,822 releases,97 rejections; source/artifact hashes verified.
- Both finite4090 jobs completed;4070 untouched. No RL training, hardware, audio or cloud changes.67 Isaac tests,21 JS tests and full mock/numerical regression passed. Next: broader perturbation/contact validation around the strong setting, then a separately defined distance sweep.


## Completed fixed-plane feasibility sweep — October 31, 2026

- [Results and limits](../baselines/isaac-plane-sweep-20261031/README.md): fixed 400 mm bucket distance, aligned release/board centers, 175 H/gap/tilt combinations on RTX 4090. 25 captured, 115 missed, 35 were rejected by dropper IK. All 140 released grid shots passed measured settling.
- Three selected settings (H/G/tilt): 450/80 mm/32°, 350/120 mm/22°, 400/240 mm/27°. Each captured 5/5 fresh same-condition repeats and its separate rendered replay. This is development search and repeatability, not RL or generalization.
- `#plane-sweep` shows the actual two-arm scene, selectable measured trajectory animations, per-height heatmaps and repeat results. Raw observations remain outside Git; aggregate evidence and hashes are committed. Prior baselines remain unchanged.
- The finite job exited in about six minutes. 4070, hardware and Viam untouched; no training across distances started. Next is review of feasible regions before a separately defined distance-conditioned study. 58 Isaac tests, five diagnostic/pilot/plane DOM tests and full regression suite passed.

## Completed repaired-IK balanced pilot — October 31, 2026

- [Results and decision](../baselines/isaac-balanced-pilot-20261031/README.md): all 62 historical IK failures recovered, 78/78 regression cases passed without relaxing acceptance. New opt-in controller uses gravity compensation and measured pose/velocity/drift gates before release.
- RTX 4090 completed 384 balanced PPO throws on 96 layouts. Validation selected episode 384 (13/18). On 36 fresh paired test layouts: pilot 21/36; existing 100-layout and 1,000-layout policies each 22/36; fixed action 28/36. All 144 test shots passed IK/settling. The pilot lost far-distance competence (3/12 versus its starting policy’s 9/12), despite better validation.
- Do not promote the pilot or begin another long study. Next: development-only feature/action-conditioning and distance/reward diagnostics, then a targeted correction with larger fresh validation and test sets. One-seed small-sample results do not establish superiority; quasi-perception remains deferred.
- New read-only `#balanced-pilot` panel shows the completed comparison, validation curve, distance bars, measured settling and limitations. Old results/contracts remain intact. Finite GPU job exited; 4070 untouched. No hardware/Viam/audio changes. 52 Isaac tests, 3 DOM tests and full regression suite passed.

## Completed IK audit and distance sweep — October 31, 2026

- [Results and limits](../baselines/isaac-ik-distance-20261031/README.md): 78 offline board IK cases; 54/62 historical failures recovered (29 larger budgets, 25 other seeds), eight unresolved. Strict tolerance preserved; 16 valid controls unchanged.
- RTX4090 completed 215 controlled shots at five distances, 200–400 mm. Grid-selected actions captured 3/3 fresh repeats at every distance. At400 mm only3/35 grid candidates captured; each distance includes seven Arm10 IK failures at250 mm gap. Frozen policies miss feasible cases despite valid IK; their weak conditioning remains a separate limitation.
- New read-only `#ik-distance` panel shows solver recovery, frozen-policy/fixed comparisons and tilt/height heatmaps. Original studies remain intact. Extended solver is diagnostic-only; no new training, quasi-perception, hardware or cloud changes.4070 stayed idle.
- Next: versioned solver/settled-pose validation and dropper feasibility checks, then a short distance-balanced learning sanity run before another long study. See [method](../sircus/isaac/IK_DISTANCE_AUDIT.md). Full regression suite and42 Isaac tests pass; interface HTTP/DOM checked, browser visual inspection unavailable.

## Completed privileged study comparison — October 31, 2026

- Both4,096-throw jobs completed; GPUs are idle. [Frozen results and diagnosis](../baselines/isaac-privileged-study-20261031/README.md):100-layout policy142/256 (55.5%);1000-layout policy119/256 (46.5%). Fixed baselines48–50%. Same test layouts/contracts verified; selected checkpoints hashed.
- The100-layout policy reached82.3% medium and66.7% far, but only21.1% near. All26/36 endpoint IK failures were on the board arm. Small action variation suggests weak conditioning; this does not prove that more scenes hurt learning (one seed/different GPUs).
- Quasi-perception training remains deferred by post-study engineering judgment. Next: board IK diagnostics and distance-balanced privileged learning, then same-device/multiple-seed confirmation on a fresh final test set. No new training or hardware work started in this comparison.
- Interface now opens with completed comparison tables, validation curves, distance bars, confidence intervals and failure/action details. Historical sensor frames remain marked stale.

## Privileged cooperative PPO study — October 31, 2026, 20:50 UTC

- [Methodology and reproduction](../sircus/isaac/LEARNING_STUDY.md): centralized scene-conditioned PPO chooses board tilt, bucket-facing heading offset and release gap. Endpoint initialization remains separate from grasping and path learning.
- Two bounded jobs are **running, not completed**:4090 uses1,000 training layouts (`general-rl-20261031T205020Z-n1000-seed42`);4070 uses100 (`general-rl-20261031T205024Z-n100-seed42`). Each gets4,096 training throws, with128 validation and256 test layouts, frozen evaluation and two fixed-action baselines. Final results are pending. Both GPUs are authorized for Sircus; the unrelated pouring workload remains stopped.
- Before launch, physical probes caught a static-USD/PhysX bucket mismatch. Contractv3 now moves a kinematic compound bucket through its root state; three floor probes passed on both GPUs. Old v1/v2 generalized diagnostics are invalid for performance claims and preserved separately. Eight repeated development inputs had zero capture disagreements.
- Corrected16-throw smoke on each GPU gave matching tiny development checks:6/8 validation,5/8 test,4/8 fixed warm action,2/8 midpoint. These verify execution only, not superiority/generalization. Final test pools were untouched.
- Live preview reads both study progress files. Jobs use separate runtime/output directories, immutable source snapshots, scene locks,6-hour timeout and3GiB free-space floor. No hardware/cloud configuration changed.30 Isaac tests and the full project mock/numerical suite passed.
- Next gate: review frozen results, failures by distance/reachability and data-volume curve before larger/replicated studies. Quasi-perception training has **not started**. It will crop actual rendered visible depth using privileged padded object volumes, with missing/occluded points preserved and no hidden ground-truth fallback in policy/controller inputs.

## Randomized RGB-D validation — October 31, 2026

- [Study contract, results and reproduction](../sircus/isaac/PERCEPTION_VALIDATION.md): frozen detector evaluated on 24 nearby board/bucket layouts × six paired lighting/noise conditions. All 144 observations completed; zero exclusions/timeouts. Ground truth and visibility labels remained evaluation-only.
- Clean: table/board/bucket detected 24/24; all 18 visible balls localized within 10mm. Board detection fell to 0/24 under changed lighting or 2mm noise. At 8mm noise, table/board/bucket were all missing; only 1/18 visible balls localized within 10mm. Combined challenge had a 113mm ball outlier. This baseline is not ready for perception-driven control.
- A renderer-lighting confound in an earlier diagnostic pass was corrected by reusing exact control captures for depth noise; all 72 paired RGB/calibration/truth checks passed in final contract v3. Earlier results were not relabeled.
- Preview shows per-condition availability/error table and a run-matched chart. Raw data/checksums remain on Ubuntu outside Git. The finite 4090 study completed; CPU observation workers/preview remain. The 4070 stays stopped; no hardware/Viam changes.
- 25 Isaac tests and full project regression suite passed. Next: robust board/plane detection on development scenes, followed by fresh held-out validation; no policy changes in this task.

## Isaac RGB-D perception — October 30, 2026

- [Perception architecture, validation and launch instructions](../sircus/isaac/PERCEPTION.md): independent CPU geometry estimates for table, pink ball, board and bucket from rendered RGB-D plus synthetic camera calibration. No simulator object poses or instance masks enter the estimator. Truth is evaluated separately on exact frame IDs.
- Preview shows overlays, centers/normals, missing/stale states and a separate error panel. Color/shape priors are explicit; this is not generalized learned segmentation. Visual capture scoring and perception-driven control remain unimplemented.
- The 256-episode RL run completed: initial 0/2, heuristic 2/2, learned frozen 8/8 fixed-scene captures. Original results/checkpoint preserved; perception is currently observed during frozen replay, not new training.
- A 30-second window produced 179 matched frames: table/board/bucket available 179/179, ball 63/179. Conditional mean errors were 1.99mm table-plane,2.18mm board-center,3.33mm bucket-center and 4.48mm ball-center; worst ball error 21.91mm. Availability is not recall. No physical/generalization claim.
- 20 Isaac unit tests and the full regression suite passed.4090 replay and independent CPU workers only;4070 remains stopped.

## Isaac fixed-scene RL task — October 30, 2026

- [Fixed-scene task and launch contract](../sircus/isaac/RL_FIXED_SCENE.md): shared one-shot policy-gradient actor chooses board tilt/heading and drop height in articulated PhysX. Positions stay fixed; no per-scene optimizer labels or search at inference.
- Actual ball-to-board contact forces and subsequent rebound gate success; whole-sphere containment uses the bucket's64 collision wall planes and one continuous second of retention. The new versioned physics/scoring contract does not relabel historical results.
- A16-episode pipeline smoke changed the saved policy and passed8/8 frozen repeats, while the midpoint heuristic passed2/2. This is functionality evidence, not evidence that RL beats the heuristic. That separate256-episode study has now completed; see the newer perception checkpoint above.
- Scope remains endpoint-initialized shot learning with known scene state. Physical grasping, joint trajectories, compliance, camera-driven observations and random-position generalization are not trained. The4070 workload is now stopped by explicit user request; only4090 is used.

## Isaac foundation — October 30, 2026

- A separate [Isaac Lab backend](../sircus/isaac/README.md) now runs on the Ubuntu desktop's dedicated RTX4090, reusing its read-only Isaac Sim5.1 binary with pinned Lab2.3.2. The existing RTX4070 workload was not restarted or reconfigured. See [verified status and limits](../sircus/isaac/STATUS.md).
- Two imported articulated UF850s, a wrist-attached landscape board, open tapered bucket, ball, table and rendered RGB-D camera are implemented. This is a new PhysX contract; historical scene fingerprints, scoring definitions and results are unchanged.
- The finite smoke test passed simulated arm command response, board attachment/orientation, bucket opening/floor contact and depth-to-world tabletop consistency. Synthetic RGB/depth/point clouds and joint/body state are saved locally. No hardware, Viam configuration or physical calibration is involved.
- Next work: resettable cooperative RL task and contact/event scoring, validated grasp/release and collision constraints, compliant board mount and calibrated bounce parameters, then perception-derived policy observations. This historical foundation statement is superseded by the RL checkpoint above.

## Integration and learning checkpoint — October 30, 2026

- PR #2 was reviewed and merged at `6f00ac3faa4e34d508b005891e62d9911ead81cf`; see [review evidence](PR2_INTEGRATION_REVIEW.md). Scoring requires a board bounce, whole-ball entry and one continuous second inside, including rim-assisted captures. Earlier result definitions are preserved.
- New scoring is opt-in. The existing Viam runtime and running dashboard have not automatically switched. The v2 native adapter is still missing and compatibility guards still block replay; endpoint IK is not path validation.
- [Scene-conditioned study](SCENE_CONDITIONED_STUDY.md): 24 training, 6 validation and 8 disjoint test layouts; real ridge regression from training-only search labels; five explicit actions; frozen direct inference. The [published synthetic checkpoint](../baselines/scene-conditioned-20261030/README.md) scored 8/8 versus 6/8 fixed action and 8/8 separately labeled per-scene search. This narrow, eight-layout test does not establish broad generalization or physical performance.
- A separate numerical-preview API/UI loads the checkpoint and replays its trajectory in the existing viewer. It was tested on an isolated offline dashboard with physical controls disabled. No deployment or restart of existing services, hardware connection, Viam changes or audio merge accompanies this checkpoint.
- Next boundaries: independent Ubuntu reproduction and broader prespecified studies; a matching fake-only v2 native adapter and collision-aware replay; calibrated physical trials only under separate authorization. Audio remains on draft PR #1.

## Historical initialization snapshot — October 29, 2026

- Mac: physical sfhack10 hosts arm10/dropper and arm14/board holder. Operator dashboard on localhost8765. Practice cycles use taught poses,2s loading,2s settled release,2s post-release recording. Completed practice records are retained locally unlabeled. Successful panel Open commands can skip redundant loading Open; unknown/failed states do not. Saved hardware poses/credentials are deliberately not in Git.
- Ubuntu: isolated fake-only server8083 and local dashboard8765. New orientation/height scripts from the tested Varun transfer are now canonical in sircus/two-arm. Original transfer archives were not changed.
- Geometry: board200x300x5mm, landscape90° mount; base spacing1337.31mm. Tapered bucket inner177/150mm, outer200/158mm.175mm reported height remains unresolved; provisional simulation depth175mm and8mm base. Mount offsets, masses, support compliance and bucket placement remain provisional.
- Learning: old U/V/gap search has one fixed board pose. New CLI searches roll/pitch/gap at fixed release XY and fixed board/bucket centers. This is not a generalized policy. New schema/fingerprint protects replay; selected candidates require native path validation.
- Perception: camera on arm10 tool bracket; stationary capture works. Camera-to-world calibration and reliable ball/board/bucket localization remain incomplete. No current scene overlay is proof of calibration.
- Tests before repo initialization:34 physical-controller/adapter mock tests;46 two-arm baseline regressions plus orientation contract checks. New training smoke4 candidates/2 synthetic tests and8 successful native fake-arm motion checks on Mac. Linux CI is the next portability check; native Linux Viam runtime remains independently unverified.
- Native cloud simulation runtime previously on the Mac is no longer assumed running. Do not reuse its cloud identity without coordinating ownership. Local8083 needs no credentials.

Current detailed physical/perception STATUS files include historical PIDs and timestamps. Verify live state before acting. Read docs/UBUNTU_SETUP.md for repository startup; old transfer/local absolute paths are historical.
