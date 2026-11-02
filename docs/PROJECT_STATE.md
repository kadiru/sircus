# Project state

## Expanded farm: all 35 sites released — November 2, 2026

- [Method and reproduction](../sircus/isaac/EXPANDED_FARM_POSES.md), [results and evidence](../baselines/isaac-expanded-farm-20261102/README.md): the completed v4 run made 350/350 releases, ten at each of the 35 unchanged bucket sites, with zero invalid trials. Five PPO updates produced 111 qualified captures, including 59 clean and 101 single-board-bounce captures. These are training outcomes, not a held-out performance estimate.
- Native qualification supplied 24 configurations per table, 840 total, with variation in all five action dimensions. The menu is branch-local around measured holds. Nine sites nevertheless repeated one projected configuration in their ten sampled policy choices; available diversity does not prove that the policy explored it.
- The bounded joint-readiness wait can finish with 34/35 cells ready. Independent final measured settling/release checks subsequently admitted all 350 releases; this does not mean every transit met the earlier readiness gate or certify collision-free paths. Earlier failed/partial diagnostics remain separate.
- The CPU checkpoint/likelihood audit passed for the immutable v6 source. The original camera bytes, distinct bucket placements, bases, geometry uncertainty and scoring remain unchanged. Inputs are five privileged geometric features; the actor chooses five shot parameters. Transit and ball loading remain scripted. No held-out/generalization, perception-driven control or physical readiness claim follows.
- Reproduction uses `--expanded-options-per-cell 24`; nearby-menu refinement recommends `--append 24`, followed by fresh native qualification. No hardware, Viam or audio deployment accompanies this work.

## Fixed-camera actual training recording — November 2, 2026

- [Training and action-only film](../sircus/isaac/ACTION_TRAINING_FILM.md), [checkpoint and verification](../baselines/isaac-action-training-20261102/README.md): ten fresh 35-slot batches feed five PPO updates, all recorded at the exact selected camera. The edit retains every batch's setup/drop/outcome and removes quiet tails: 23.33 s versus 52.33 s for all captured frames, both native 1080p/30 fps. No camera tracking or outcome-selected replay.
- Seventeen geometry-supported sites, eighteen parked sites. Of 350 sampled slots, 170 are enabled and 156 release; 30 qualified captures, 28 clean. Foreground: ten board rebounds, two captures, five table outcomes and three measured floor arrivals. All slots, including invalid penalties, remain in the original PPO buffer. This short filming pass is not held-out policy evaluation or evidence of solved workspace coverage.
- All 350 sampled latent likelihoods were recomputed from their saved policies on CPU (max error 9.54e-7), with five update links and final weights bound. Native contact/rim/isolation, every camera/frame, full-rate foreground scoring, cut timing, video decode and Mac transfer hashes passed; all 88 source hashes match. Local suite: 182 tests with seven Torch skips; six new audit tests passed on Ubuntu CPU.
- The partial v2 stopped before release eight due to a camera progress tracker at a nearly unchanged pose. V3 records a fresh full run using planned-time RGB sampling and actual measured joints; joint readiness, final release guards and physics scoring are unchanged. No partial v2 footage/outcomes were reused. Scripted transit, initialized balls and privileged inputs remain explicit; there is no learned grasp/path or physical validation.
- The finite 4090 job exited; the existing 4070 workload and all services were preserved. The first capture queued behind another learning job's shared lock rather than interrupting it. No hardware/Viam/audio work. Previous films and studies retain their original definitions.

## User-selected farm film completed — November 2, 2026

- [Native one-minute recording](../sircus/isaac/CHOSEN_FARM_FILM.md): 35 tables / 70 articulated arms, selected silver-gray look, 14-second orbit, 2-second approach and 44-second fixed front-row view. The user's original camera JSON is preserved byte-for-byte and every native camera frame checked; no ball-following. [Compact evidence](../baselines/isaac-chosen-farm-20261102/README.md).
- Eight foreground board rebounds: two retained captures, four measured floor arrivals and two bucket/table misses, with visible configuration changes and background releases. Full film is 60 s; front-row-only file is 44 s, both 1080p/30 fps. Native contact/isolation probes, rim rays, source lineage, frame/chapter counts, decode and Mac transfer digests passed. Selected native release/contact/outcome samples for all eight trials were inspected.
- Replays selected PPO actions with scripted joint transitions and initialized balls; no new optimizer updates or held-out performance claim. All selected/repeated batches total 420 attempts, 198 releases, 29 captures. Source case `r1c2` stays at display slot `r1c7` as in the scout; this does not solve the old rightmost target. Geometry, reported-height uncertainty, historical training and previous videos remain intact.
- Isaac regression suite completed 158 tests with one Torch-dependent skip on Mac; Black and diff checks passed. The finite 4090 job exited; read-only final GPU check found it idle and the existing 4070 workload active. Existing camera scout/services were preserved. No physical/Viam/audio changes.

## Interactive camera scout — November 2, 2026

- [Camera-only browser window](../sircus/isaac/CAMERA_SCOUT.md) uses original UF850 meshes and a frozen native Isaac snapshot: 35 tables, 70 arms, measured body/object transforms and the selected right-front layout. All 490 arm-mesh matrices match the native export. Browser lighting/material appearance is approximate; final renders remain in Isaac.
- Orbit, Pan, zoom, numeric coordinates, exact reset, browser autosave and full-precision camera JSON download were verified. 146 Isaac tests passed with one Torch-dependent skip. [Compact verification](../baselines/isaac-camera-scout-20261102/README.md). Raw scene/meshes/screenshots remain outside Git.
- The finite 4090 export completed. A loopback-only GET server remains running on the Mac at `http://127.0.0.1:18784/` for the user. No simulator/hardware commands are exposed, and no existing services or 4070 workload were changed. Native remote Isaac streaming was not installed or enabled.

## Right-front camera choices — November 2, 2026

- [Ten native silver-gray previews](../baselines/isaac-camera-options-20261102/README.md) compare fixed views of the same 35-table scene. Whole source cases `r1c2` and `r1c7` exchange origins so a previously demonstrated case occupies the right-front corner; all targets/objects stay with their source cases. This does not solve the old rightmost target.
- Native image/readback, source/transfer hashes and frozen-state checks passed; all ten final renders were visually inspected. The selected lighting is preserved and exposed sky is gray. 141 Isaac tests passed with one Torch-dependent skip; gallery layout/navigation checks passed. The finite 4090 job exited; 4070 and existing services were untouched.
- View 06 was selected and [revised by shifting 0.30 m camera-right and raising Z to 2.0 m](../baselines/isaac-view06-z2-20261102/README.md), with its aim unchanged. The native still passed source/image hash and camera readback checks; 143 Isaac tests passed with one Torch-dependent skip. It waited for the existing learning run's GPU lock and did not interrupt either GPU workload. Review of this still precedes the [next one-minute orbit and fixed close-up film](../sircus/isaac/CAMERA_OPTIONS.md). These stills add no training updates, rollouts, new performance claims or physical validation.

## Silver-gray fixed-camera Isaac recording — November 2, 2026

- [Completed recording and evidence](../baselines/isaac-silver-station-20261102/README.md): selected silver-gray tabletops, medium-gray floor, slightly brighter neutral fill and softer shadows. The camera stays fixed for twelve front-row releases with concurrent background drops. All twelve rebounded; three captured, six reached the floor and three missed after bucket/table contact.
- The 57.33-second readable edit removes waits/tails. The 20.67-second fast file samples the full 206.4-second recorded source at 10×; all twelve events retain descent and rebound frames. Its release phase is approximately 2.5× physical speed because the source release phase plays at quarter speed. Native pixels, contacts, encoded frame/chapter counts, resume hashes and Mac transfer digests were checked.
- This replays selected PPO actions with scripted joint transitions and initialized ball loading; it is not a new training or held-out evaluation result. Across all batches: 420 attempts, 197 accepted releases, 23 captures. Historical baselines, geometry/provisional heights and final release guards are preserved. Local regression passed, including 137 Isaac tests with one Torch-dependent skip. The finite 4090 job exited; 4070 and existing services were untouched. No physical/Viam/audio changes.

## Compact dark-floor Isaac recording — November 2, 2026

- The [new film](../sircus/isaac/DARK_FARM_FILM.md) shows 35 tables and 70 arms at 2.50 × 2.10 m spacing, with a continuous black floor and 9 mm charcoal seams. Seven close-ups show seven measured board bounces, two qualified captures and three real floor arrivals. Four consecutive attempts at one front-row table use different configurations and include measured joint movement.
- [Verified replay evidence](../baselines/isaac-dark-farm-20261102/README.md) records 350 previously sampled PPO actions, 162 releases and 28 captures. This camera-directed replay includes repeated source batches. It uses scripted arm transitions, initialization of the ball at release, and the original contact/capture scoring; the historical training results remain identified separately.
- The one-minute 1080p video, 19-second impact edit, normal station edit and 10× station edit passed decode and checksum checks. Recording resumed at a camera cut from saved joint poses after one neighboring station's transition failed; the reused RGB and shot records were verified byte-for-byte. Local tests passed, including 128 Isaac checks with one Torch-dependent skip on Mac. Videos and raw logs remain outside Git.

## Completed workspace feasibility map — November 1, 2026

- [Verified settings and evidence](../baselines/isaac-feasibility-map-20261101/README.md): actual articulated Isaac Lab/PhysX search at 32 tabletop targets, with 17 board/release XY candidates and height/gap/tilt variation. Completed 6,212 attempts (3,043 releases; 3,169 endpoint/clearance rejects) in 25.96 minutes on RTX 4090. No new RL policy.
- Fifteen targets had frozen settings and passed 3/3 nominal repeats each. Fresh perturbations: 364/450 captures, 338 single-bounce and 276 clean; six targets passed 30/30 each. Seventeen targets remain unresolved. This does not establish continuous coverage or physical reliability.
- Target 23 is outside the conservative dropper reach bound but scored only 6/30 fresh repeats, all multiple-board-contact/assisted captures. Nominal 3/3 is not robust far-shot performance. Fixed-original-XY ablation scored 18/45 versus 45/45 selected nominal; it is a small selected development comparison.
- All 6,212 records, 63 source hashes, four evidence hashes and 80 unchanged assets checked. Six contact/isolation probes passed. The portable verifier preserves the exact original fingerprint; four intentional corruptions were rejected. No raw histories, credentials or generated assets enter Git.
- Read-only `/feasibility.html` shows the verified map, exact settings, failure reasons, recorded trajectory projections and CSV. The finite 4090 job exited; preview remains live; unrelated 4070 work, physical hardware, Viam and audio untouched. Next: robust far-target/trajectory diagnostics with reachability-aware proposals before another long PPO study. See [protocol](../sircus/isaac/FEASIBILITY_MAP.md).

## Direct native Isaac application recording — November 1, 2026

- Low close-up: `--view close-up --focus-cell 2 --playback-rate 0.5` frames one near-target table with both arms, board and bucket mouth clearly visible. Verified 58.634-second native video; foreground shots were one miss then three captures. This repeats the same bounded demo, not a new independent learning study. [Camera, contact and media evidence](../baselines/isaac-native-viewport-20261101/closeup-verification.json).
- Perspective follow-up: `--view perspective` adds an elevated diagonal pinhole view with side panels hidden; camera parameters are saved in the run metadata. A separate 58.634-second native recording was verified and archived, preserving the orthographic clip. Same bounded seed/contract, not a new independent policy study. See [perspective evidence](../baselines/isaac-native-viewport-20261101/perspective-verification.json).
- [Native capture and reproduction](../sircus/isaac/NATIVE_VIEWPORT.md); [verification](../baselines/isaac-native-viewport-20261101/README.md). Direct X11 capture shows all 32 tables / 64 arms in the actual Isaac Sim 5.1 window, including menus and scene tree. No composited panels, generated replacement imagery or added trails. 58.1 seconds, 2400 × 1000, 30 fps.
- A separate bounded demonstration completed 128 PhysX throws and two PPO updates (seed 20261104): 27 training captures, seven clean. These are not held-out test results or evidence of policy improvement. All contact/isolation probes, batch counts, update hashes, source hashes, full movie decode and sample-frame inspection passed.
- Current training still covers a small tabletop grid with prescribed heading and fixed release/board XY. Actions are height/gap/tilt; endpoints are initialized, and pickup/transit are not learned. Arbitrary bucket positions are not covered. The existing selected policy's 6/32 test result (fixed baseline 9/32; far 0/14) remains unchanged.
- The temporary GUI container was removed after recording. Existing preview/simulation services and the unrelated 4070 workload were preserved. Raw media, generated assets, logs and demonstration policy remain outside Git. No physical/Viam/audio actions.

## Actual 32-table PPO training farm — November 1, 2026

- [Method and video reproduction](../sircus/isaac/FARM_PPO.md); [completed pilot and checkpoint evidence](../baselines/isaac-farm-ppo-20261101/README.md). A shared neural actor/critic trained on 2,048 PhysX throws across 32 simultaneous tables / 64 UF850 arms. The movie records actual on-policy batches at iterations 1, 8 and 16, with a wide orthographic view and synchronized close-ups. It is separate from the prior search and frozen replay.
- Validation selected update 12 (14/32 versus initial 8/32). Frozen test: selected 6/32, final 7/32, initial 7/32, fixed action 9/32. All 128 test endpoints passed IK/settling; all policies scored 0/14 on the far test subset. The policy is not promoted. This single-seed, small pilot demonstrates training/data/video integration, not a solved task or broad generalization.
- Policy actions are board height, drop gap and tilt, with bucket-facing heading prescribed. Inputs are privileged geometry. Endpoint initialization is separate from pickup, arm paths, gripper control and perception. Historical physics/scoring results and provisional measurements are preserved.
- All 2,048 training rows, filmed-row membership, weight-update chain, saved test actions, artifact/source hashes and contact probes were verified. The finite 4090 run completed; the 4070 workload and existing preview were unchanged. Raw frames/video remain outside Git; small checkpoints and compact evidence are published. No physical/Viam/audio changes.
- Next: diagnose far-shot feasibility and flat external-contact rewards on development scenes before another long run, then use a fresh held-out test. Existing dashboard pages have not automatically switched to this study; progress has its own `farm-training.json` file.

## Close-up simulation replay — November 1, 2026

- [Orthographic video and reproduction](../sircus/isaac/OVERHEAD_REPLAY.md) uses four selected cells from the existing parallel-world baseline. A fresh bounded 4090 replay passed contact/foreign-cell probes and endpoint settling; all four shots had one board bounce and clean capture with one-second retention. This is visualization of frozen settings, not a new training study.
- Six synchronized cameras recorded 252 frames each at 120 Hz. The 18.6-second, 1920 × 1280 movie shows four tables at 0.5×, then cell-00 in overhead/front views at 0.25× with measured simulator trails. Arms start at solved poses; pickup and transit are not shown. Numerical baselines and the existing dashboard remain unchanged.
- The finite GPU job exited. The existing 4070 workload, physical services, Viam and audio were untouched. Video/raw frames are outside Git; compact evidence and reproduction code are published.

## Completed parallel world — November 1, 2026

- [World, launch and limits](../sircus/isaac/PARALLEL_WORLD.md); [exact settings/evidence](../baselines/isaac-parallel-world-20261101/README.md). One RTX 4090 PhysX world contains 32 tables and 64 articulated UF850 arms. Each table has one board, one ball and one bucket; 32 distinct table-local bucket targets stay fixed throughout all height/gap/tilt trials. Distances span300–632 mm between cells, constant within each cell.
- Completed 1, 090 attempts in 197.3 s after application launch: 64 controls, 896 search attempts, 130 fresh perturbations. Thirteen targets had qualifying searched settings; six passed 10/10 fresh repeats each. Confirmation totals107/130 captures (all 107 single-board-bounce; 67 clean), 17 dropper-IK rejects and 6 released misses. Farthest selected target440 mm; no setting found farther away in this finite search. This is not a new RL policy or held-out generalization evidence.
- Bucket drift stayed below0.000469 mm. Five attribution probes passed in every cell, plus foreign-cell collision isolation. All 1, 090 raw/compact rows, three artifact hashes and 48 source hashes matched. One-/four-cell reference controls passed, but a borderline shot flipped in 32 cells; strict clone equivalence remains unproven. Older definitions/results are preserved.
- Read-only `/parallel-world.html` shows the actual multi-arm world, cell details, valid/invalid results, verified CSV and a 4.03-second matched-camera video of simultaneous exploration. Full mock/headless regression and browser checks passed. The finite 4090 jobs exited; 4070 workload, physical hardware, Viam and audio were untouched.
- Next: address far-shot feasibility and reachability-aware action bounds, then implement a batched policy adapter with disjoint spatial splits and frozen-checkpoint evaluation. The current environment initializes joint endpoints; pickup, arm paths and cloud-conditioned policy learning remain separate.

## Completed spatial bucket grid — November 1, 2026

- [Verified settings and evidence](../baselines/isaac-bucket-grid-20261101/README.md): nine XY targets, 689 attempts in 21.5 minutes on RTX4090. Three near targets (300–361 mm) each passed 30/30 fresh perturbations, all single-board-bounce captures. Center/right middle targets scored 18/30 and 6/30 board-only captures; the latter had no single-bounce captures. No qualifying setting was found for four other cells at 492–632 mm. Finite search failure is not impossibility.
- H/gap varied ±2 mm and tilt ±0.5° after each selected setting was frozen. These are searched development cells, not held-out generalization or a new RL policy. New additive `sircus-board-route-v1` excludes detected table/floor/robot/other contacts while preserving historical scores. Exactly-one-board-bounce is a separate metric.
- Five contact probes and two controls passed. The old 400 mm control uses two board contacts; the old 600 mm control measurably hits the table. All 689 raw/compact records, three evidence hashes and 43 source-snapshot hashes matched. Maximum released-trial settling errors: 0.848 mm and 0.104°. Geometry remains measured/provisional as documented.
- Final data, per-cell settings, confidence intervals and CSV are live at `#bucket-grid`. The finite GPU job exited; 4070 workload, hardware, Viam and audio untouched. Raw observations remain outside Git. See [protocol and published RL compute examples](../sircus/isaac/BUCKET_GRID.md).
- Next: investigate feasible far shots, then validate a vectorized shot environment before a larger scene-conditioned PPO study with disjoint spatial evaluation. Do not relabel this search as RL or promote an existing policy on this evidence.

## Completed bucket-distance catalogue — November 1, 2026

- [Exact settings and verified evidence](../baselines/isaac-distance-catalogue-20261101/README.md): 2,081 attempts on RTX 4090, including 300 fresh confirmation trials. Frozen settings captured 50/50, 50/50, 44/50, 41/50, 49/50 and 37/50 at 300, 400, 500, 550, 600 and 650 mm respectively. Independent variations: board height/gap ±2 mm, tilt ±0.5°. The 400 mm setting is a prior-result control, not a new broad search.
- Important qualification: the original capture scorer permits intermediate contacts. A separate sampled-path audit flags low external rebounds in 0/50, 0/50, 7/44, 39/41, 49/49 and 36/37 captured trajectories respectively. These suggest table/other contact; no flag does not certify direct travel. Preserve original scores and historical definitions. Farther high capture rates are not proof of the intended direct board-to-bucket shot.
- Source snapshot `distance-catalogue-v1`; run `distance-catalogue-20261101T125810Z` completed naturally in 72.4 minutes. All 26 artifact hashes, 37 executed source hashes, the contract fingerprint and all 2,081 raw/compact trial records matched. 75 Isaac tests, four distance-panel DOM tests and full mock/numerical regression passed.
- Read-only `#distance-catalogue` shows all results, warnings, selected nominal geometry and a validated CSV. The 4070 workload was untouched. No physical/Viam changes or new RL training. Next: instrument table/robot contacts and define a separate versioned direct-path criterion before another learning study. See [method](../sircus/isaac/DISTANCE_CATALOGUE.md).

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
