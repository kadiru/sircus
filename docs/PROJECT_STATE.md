# Project state

## Calibrated 650 mm replay — November 4, 2026

- [Frozen finite protocol](../sircus/mjlab/REPLAY650.md) and
  [geometry-only preflight](../baselines/mjlab-calibrated650-20261104/README.md):
  replay the exact 380 mm board / 28° tilt / 386.8 mm gap shot at 650 mm distance,
  with matched positive/negative controls and three nominal copies at64/128.
  All eight initial/conditional endpoint groups passed fresh5mm CPU clearance.
- The five-gap window is conditional on both-resolution capture/control, valid
  physics, independent scoring and ≤5mm350ms drift gates. Maximum48 requests;
  this is held-endpoint engineering verification, not new RL or fresh-noise
  reliability. Historical shallow misses are preserved.
- Fresh18 native releases completed: positive control3/3 and negative0/3 at
  both resolutions; candidate3/3 at64 but0/3 at128, which never achieved whole
  entry. Candidate350ms drift1.734mm passes the positional check, but outcome
  agreement fails. Window remains closed; none of its30 conditional requests ran.
  The64 captures are rim-assisted; finer integration deflects the ball outside
  after rim contact. Final independent evidence audit is being completed.
- Additive read-only interface key `calibrated650_replay`; existing viewer,
  4070 workload and Isaac configuration remain separate and unchanged.

## Calibrated benchmark v2 — completed November 4, 2026 UTC

- [Protocol, results and reproduction](../sircus/mjlab/CALIBRATED_BENCHMARK.md),
  [checksummed evidence and six selected artifacts](../baselines/mjlab-calibrated-benchmark-20261104/README.md),
  [draft PR62](https://github.com/kadiru/sircus/pull/62), stacked on PR60. No merge
  or checkpoint activation. Historical v1 and earlier scene results remain
  unchanged below; their scoring, setup and evaluation cohorts differ.
- **Frozen TEST: sweep78/216 (36.1%), PPO97/216 (44.9%)**, including setup rejects
  in every denominator. The8.8 percentage-point PPO advantage is descriptive:
  three seeds share24 held-out layouts with three paired trials per layout.
  Per-seed captures are31/22/25 versus28/33/36 out of72; PPO lost the first seed.
  Clean captures are16 versus26. Both methods still miss more than half of
  requested shots; this does not establish universal coverage or convergence.
- Both methods spent3,072 TRAIN requests. Native TRAIN releases were2,036 sweep
  and2,344 PPO. Full totals:6,144 TRAIN +864 validation +432 TEST =7,440 requests;
  native releases4,380/704/383 by phase. Ten native TRAIN numerical failures count
  as failed attempts. All six workers completed; the independent full audit
  **passed1,204,473 checks** with no failures and exact source/control/artifact
  bindings. Raw trajectories remain local; the compact bundle references their
  original manifests rather than bundling raw observations.
- Validation selected budgets0/1,024/1,024 for sweep and1,024/1,024/256 for PPO
  (seeds20261411/12/13). Selection uses captures, mean shaped reward, then earliest
  checkpoint; a capture tie does not imply identical artifacts. Every worker
  still paid all1,024 TRAIN requests. The third PPO seed's earlier checkpoint
  beat its final one on validation; TEST never selected or replaced checkpoints.
- Current-ball simulated contact calibration passed3,444 fit and15,011
  verification checks. Canonical64, fine128 and reversed64 controls passed the
  frozen wall3.563906mm/floor2.943836mm contact envelope without refitting. These
  allowances are separate from2µm geometry precision and are not physical
  material calibration. Capture requires measured board rebound, downward
  open-mouth whole-ball entry and one continuous observed second inside;
  partial exits reset retention, rim assistance counts, external contacts do not.
- Geometry-only pre-policy resolution plus exact-action admission preserves5mm
  clearance,>150mm board–bucket distance and serialized forward/reverse paths
  sampled at≤0.5°. A new canonical simulator park removes the old nominal-pose
  veto; this is not a proven path from the old park. CPU controls admitted69/71
  old layouts, retaining two genuine fixed-base/bucket conflicts. Finite IK and
  requested-action collisions can still reject new actions; no projection or
  flight-label fallback masks them. Verified-prefix reuse reduces repeated path
  checks without changing actions or the frozen clearance/sampling limits.
- Mean setup + TRAIN wall per worker: sweep4,018.9s / PPO4,076.6s; CPU6,257.2 /
  6,325.0s. Validation + TEST wall1,439.3 /1,070.2s; full GPU reservation5,459.1 /
  5,147.7s. Lifetime peak RSS means1,878.5 /2,002.6MiB, sampled VRAM1,092 /
  1,216MiB. Costs include evidence I/O; memory includes simulator/evaluation.
  Geometry-only source resolution still costs roughly300s per24-layout cohort;
  actor prediction timing is not end-to-end planning latency.
- The separate [read-only interface on18791](http://127.0.0.1:18791/#sweepRlBenchmark)
  verifies both audit dependencies and displays measured curves, admission
  reasons, resource costs and unsealed final TEST records. Its reader revision
  has separate provenance from immutable experiment producer C. All prior
  services, including18790, and the4070 workload were preserved. No physical,
  Viam, audio, camera or cloud configuration work occurred.
- [Calibration/setup/native engineering controls](../baselines/mjlab-calibrated-controls-20261103/README.md)
  are separate:17/17 source captures,50 requests/48 native releases, a discarded
  actual PPO update and9,167 independent checks. The complete synthetic CPU
  six-worker/controller/auditor/exporter gate and native reset/constructor
  regressions also passed. The old71 proof audit checks recorded hashes and
  structure; it does not recompute every collision sample.
- This remains current fixed-ball, privileged-geometry five-parameter shot
  selection with17 shared historical recipes (known4,234 prior native releases,
  earlier acquisition cost unknown) and held articulated endpoints in mjlab /
  MuJoCo Warp. No learned grasp/transit/compliance, perception, physical
  validation or new-ball adaptation claim. Reported175mm bucket-height ambiguity
  is preserved. Next: a validation-only failure breakdown before another change.

## Historical v1 matched sweep/PPO benchmark — November 3, 2026

- [Frozen design and reproduction](../sircus/mjlab/SWEEP_RL_BENCHMARK.md): guided
  sweep-built library transfer versus continuous residual PPO, using the same 17
  audited shared source recipes, corrected 64-contact and common-home/local gates.
- Completed 3 seeds × 1,024 spent training decisions per method, with
  0/256/512/1,024 validation milestones, 12 fresh validation and 24 final-test
  layouts, three paired nominal/perturbed requests each: 6,144 TRAIN + 864
  validation + 432 test = 7,440 requests. Rejects consume budget. Native releases
  are 3,026 TRAIN, 496 validation and 216 test = 3,738; TRAIN native counts are
  sweep 1,385/3,072 and PPO 1,641/3,072. Shared 17-recipe historical cost is separate.
- [Checksummed frozen evidence and six selected artifacts](../baselines/mjlab-sweep-rl-benchmark-20261103/README.md):
  sweep **67/216 (31.0%)**, PPO **58/216 (26.9%)**, under the frozen formal
  strict-containment metric only. Validation selected budgets 0/0/0 for sweep
  and 256/1,024/0 for PPO (seeds 20261311/12/13); all workers still spent 1,024
  TRAIN decisions. No checkpoint was activated. Mock/headless and pinned Ubuntu
  CPU/PPO checks passed.
- The corrected independent full audit passed **902,252 checks**. The first
  attempt failed while composing a summary with a duplicate native-count keyword;
  an auditor-only fix in a separate source snapshot produced a new full audit.
  Executed workers, frozen report, scores, evidence and selections were unchanged.
  [Native controls](../baselines/mjlab-sweep-rl-controls-20261103/README.md) passed:
  17/17 exact source captures, 50 requests / 48 native releases, a deliberately
  invalid request rejected, and a discarded real PPO actor update. Independent
  controls audit passed 6,881 checks; these controls are separate from the 7,440.
- Per-worker three-seed mean setup + TRAIN wall: sweep 2,097.0 s, PPO 2,099.6 s;
  process CPU 3,111.6/3,123.6 s and full GPU reservation 3,134.6/3,129.5 s.
  Lifetime RSS means are 1,628.4/1,738.4 MiB, sampled process VRAM 762 MiB for both.
  Timings include implementation/evidence I/O; memory includes simulator and
  evaluation. Full costs and artifact sizes are in the frozen CSVs. `query_s`
  totals cohort prediction/preparation, not per-query or end-to-end latency.
- New read-only results server [18789](http://127.0.0.1:18789/#sweepRlBenchmark)
  binds the passed corrected audit. The earlier 18788 process with its failed
  original-audit binding is untouched; this was a separate server, not a restart.
- A [validation-only trace diagnostic](../baselines/mjlab-sweep-rl-numerics-20261103/README.md)
  found a material metric limitation: tiny wall
  overlaps can reset the 1 µm strict containment predicate even for early-entry,
  end-inside balls. The completed frozen study's formal labels do not establish
  ordinary capture/adaptation rankings. Independent contact
  calibration and a versioned scorer are required before that stronger claim.
- Original 35-position results/services and 4070 workload are preserved. Current
  ball/materials and reported 175 mm bucket-height uncertainty remain unchanged.

## Completed expanded 35-position planner study — November 3, 2026

- [Final settings, source-matched rates and compact evidence](../baselines/mjlab-grid35-expanded-20261103/README.md):
  **17 selected targets, 12 unresolved, 6 clearance-admission blockers**. Parent
  and diversified follow-up both completed and passed independent audits
  (66,373 and 74,791 checks; overlapping check sets are not additive evidence).
- The parent selected 17 recipes; separate frozen repeats scored 51/51 nominal and
  495/510 perturbed. One recipe scored 22/30. The follow-up changed that target's
  board XY/bearing and joint branches; its selected replacement scored 3/3 and
  28/30, all clean. Height, tilt and gap were essentially unchanged, and seeds
  differ: this is an observed new-configuration result, not causal attribution.
- The current bank retains 16 exact parent recipes plus the replacement. Its
  individual recorded frozen rates are 28–30/30; ten are 30/30. These source-matched
  stored cohorts are not a fresh common-bank evaluation or learned-policy rate.
- The follow-up checked all 13,312 proposals at 13 unblocked targets. Only 101 of
  6,656 broad/inverse proposals passed endpoint/clearance admission; all101 reached
  native physics. Another 1,563 native search requests were local proposals. The
  prefix bias was corrected; finite reach, clearance, path and flight coverage
  remain limits. Six blockers record unproven clearance, not negative penetration.
- Together the distinct stages requested 4,243 shots and released 4,234 natively:
  2,944 search + 705 qualification + 594 frozen requests. Nine rejected qualification
  requests remain in denominators. Do not pool those roles into a performance rate.
- Both workers and the continuation exited. The separate 18787 results view
  provides all 35 rows, full-precision CSV and read-only saved-shot replay. Original
  18786/18781 services and the 4070 workload were preserved. No physical/Viam/audio
  action, new RL, automatic recipe-catalogue update or policy deployment occurred.

### Preserved method and integration boundary

- [Protocol and reproduction](../sircus/mjlab/GRID_STUDY.md): run all 35 distinct
  bucket targets, freshly confirm saved settings, then expand unresolved cells
  with local gap/height/tilt/radius/bearing sweeps and broad proposals. Historical
  captured geometries are warm starts only; current native checks supply labels.
- Baseline budgets are 64 CPU proposals / 16 native search requests, expanded to
  1,024 / 128 where unresolved. Up to 3 / 6 captured candidates receive the
  unchanged strict geometry, common-home paths and 3 + 20 qualification trials.
- Freeze all selected recipes before a separate 3 nominal + 30 perturbed repeat
  set per selected cell. No fallback selection from final validation outcomes.
  Qualification, final repeats and saved historical evidence remain separate.
- A separate loopback results view on 18787 shows all table settings, counts,
  explicit missing/blocked states, CSV and read-only native-data replay. The
  original planner on 18786 and other simulation services are preserved. The
  completed 4090 results are bound to their source/scene identities and audits.
  No new RL or physical deployment.
- An independent sampling audit found the first expanded cell spent all 128
  native slots on local refinements before any broad sample was executed.
  The [additive diversified follow-up](../sircus/mjlab/GRID_STUDY.md) corrects this:
  audit all 1,024 proposals before choosing up to 128 eligible native attempts,
  with a 64-slot broad/inverse quota and normalized geometric diversity. It
  requires a completed, audited parent; archived outcomes never count as new
  trials. The original parent remains unchanged. Only new selections get
  new frozen repeats; inherited rates remain separately labeled.

## Sweep planner implementation checkpoint — November 2, 2026

- [Independent simulation sweep planner](../sircus/mjlab/SWEEP_PLANNER.md) accepts
  tabletop bucket XY and optional positional locks without requiring an RL
  checkpoint. It preserves learned-policy guards and legacy result identities.
- Exact corrected-catalogue recipes at12 grid positions replay saved hash-bound
  native evidence. New positions use source-relative/analytic/broad proposals,
  strict geometry, corrected native contact,3/3nominal and>=18/20fresh captures,
  plus sampled common-home geometric admission before a ready result.
- The viewer is one workcell plus a selectable7x5 farm-target grid, not a live
  Isaac farm rendering. Endpoint-held dynamics are not validated arm compliance.
  Raw evidence/cache stays local. [Native verification and runtime identities](../baselines/mjlab-sweep-planner-20261102/README.md)
  show two new development targets, each3/3nominal and20/20fresh, with independently
  rescored native data and supplemental CPU geometry/path checks. Both replay in
  the new loopback18786 UI. This is search, not an unseen-target RL result.

## Conditional gap-window/SAC experiment — November 2, 2026

- [Prespecified method](../sircus/mjlab/WINDOW_STUDY.md): user-approved fixed-bucket
  sweep followed by a binary success predictor and matched-budget terminal SAC.
  Context is board height/tilt; action is release gap only. This is not learned
  joint control or unseen-bucket generalization. No active policy switch.
- New strict CPU adapter freezes one q14 branch per context before native labels,
  independently solves dropper gaps, verifies both full flange poses and keeps
  rejected requests visible. Ubuntu CPU preflight:23 tests passed; exact anchor
  and ±10 mm gaps retain q14 and pass native FK/clearance. The original comparison
  completed on4090 in29.1 min; [evidence/checkpoints](../baselines/mjlab-window-sac-20261102/README.md)
  passed886 independent checks, including rescoring all397 native releases.
- Both methods receive504 shared seed requests plus504 exclusive requests. Four
  equal data milestones, quarantined validation selection and frozen final
  8-context ×23-repeat tests include the same bounded-uniform baseline. Conditional
  sampled windows, objectives, noise and unknown transit limits are explicit.
- Frozen tests: supervised predictor23/184, SAC0/184, bounded uniform0/184.
  Both learned methods released69/184 natively; uniform's eight frozen random
  base gaps all failed geometry. Every predictor capture came from H440 mm,
  tilt32°, gap285.781 mm:3/3 nominal and20/20 gap-noise repeats, all rim-assisted.
  Its specified common-home geometric path passed separately. This supports one
  narrow transfer case. No checkpoint was activated.
- All2,211 attempts remain:1,512 unique train,144 validation,552 test and three
  quarantined controls;1,814 rejected geometry. Train/validation/test board IK
  resolution was6/24,2/6,3/8 under current budget/staging. The adaptive sweep
  found four separated sampled capture runs at H397/500 mm, tilt36.1548°;
  a separate92-request center-repeat follow-up completed and passed39 audit checks.
  Two wider-run centers scored3/3 nominal +20/20 fresh: H397.388/tilt36.1548°/
  gap321.219 mm and H500/tilt36.1548°/gap238.531 mm. The two narrow patches scored
  only5/20 fresh each; one midpoint captured0/3 nominal despite captured neighbors.
  All59 confirmation captures were rim-assisted; gap-only robustness is limited.
- Only4090 native simulation is authorized;4070/Isaac/runtime services remain
  unchanged. Both finite jobs completed. Next: expand reachable height/tilt TRAIN
  neighborhoods and freeze fresh validation/test contexts before another learning
  run. Current evaluated contexts are known; no broad coverage claim or long run.

## Completed corrected-contact spatial study — November 2, 2026

- [Final results](../baselines/mjlab-corrected-grid-20261102/README.md) and [protocol](../sircus/mjlab/CORRECTED_GRID.md): 35 development bucket positions, 11,760 initial geometry requests, 460 strictly admitted endpoints at 27 positions. The corrected articulated/passive-gravity 64-substep contract is frozen; historical definitions and policies are preserved.
- Completed 1,127 shot requests / 1,114 native releases: search 349/460 captures, nominal 83/87, fresh perturbations 495/580 (13 geometry rejections remain failures). These are selected development trials, not frozen-policy success rates. Seventeen sources at 12/35 positions meet common-start paths, 3/3 nominal and ≥18/20 fresh captures. Five positions have multiple qualified sources; 22 had any search capture, 17 a checked common-start connection, 29 a clear starting posture.
- Both contract-matched audits pass: 3,115 original checks and 3,165 supplemental checks, each independently rescoring the same 1,114 releases. Check sets overlap. Jobs and finite watchers exited normally; the existing preview server/policy routing remains unchanged. Final `#correctedGrid` displays both audits, phase denominators and training readiness; only static/derived display files changed.
- Eight positions lack an admitted endpoint, five had no search capture, five captured but lacked the prescribed common-start path, and five connected but failed confirmation; these categories plus 12 qualified sum to all 35. All ten positions in the X=1.05/1.24 m columns had zero captures. Four layouts obstruct the frozen starting posture; two others overlap modeled fixed bases, with supplementary evidence preserved separately from conservative original categories. Failed finite searches do not establish universal impossibility.
- Evidence supports preparation of a bounded shot-selector pilot, not broad all-table training. Existing trainers/inference still use legacy physics/admission. Next: deduplicate relative geometry prototypes while retaining branch provenance, implement and smoke-test matching corrected-64/common-home admission, then fresh-init 2,048–4,096 terminal-PPO throws versus identical uniform feasible proposals. All 17 records are distinct absolute joint configurations but only 12 relative prototypes. The frozen 35/30/100 train/validation/test coordinates remain unchanged and future validation/test have not received outcome search.
- No new policy training or deployment occurred. The 4070 workload and Isaac setup were untouched. Prescribed endpoints and sampled geometric transit do not establish actuator tracking, grasping, continuous collision avoidance, perception or physical validation. Reported 175 mm bucket height remains unresolved; provisional 175 mm interior plus 8 mm bottom is retained.

## Contact repeatability and low/high transit — November 2, 2026

- [Verified 1,708-throw results and exact configurations](../baselines/mjlab-contact-transit-20261102/README.md): the previous 13/20 used perturbed poses. Duplicate gravity compensation is a verified control bug; the dominant capture improvement comes from finer integration, with unchanged geometry/material/scorers. Corrected articulated 32/64 each capture 100/100 unique fresh jitters on both resets, plus 4/4 singleton checks. All corrected fresh captures are rim-assisted.32→64 still differs by up to 4.873 mm at 350 ms; no exact physics or physical calibration claim.
- Independent audits pass 415+414 checks and rescore all 1,708 throws. Saved policy and original 17-template bank remain intact; no PPO. Historical results keep their version. The new read-only `#contactRepeatability` panel adds exact CSV and native-FK geometric path previews; existing inference routing stays unchanged.
- Stricter CPU collision checks include non-adjacent self-collision and a carried ball. Nominal 21+20 jitter cases pass full serialized high-dropper/low-board cycles (21/25,11,650 samples,≥5.042769 mm modeled lower bound). Three others have separate modest alternative routes; target 39 is rejected for self-penetration. Sampled geometry is separate from actuator dynamics or universal reachability. See [protocol](../sircus/mjlab/CONTACT_REPEATABILITY.md).
- All finite native/CPU jobs completed.4090 only/shared lock used;4070 workload and Isaac setup untouched. Simulation preview process retained; static/derived display updated and stale local SSH forward restored. Next study must explicitly bind corrected contact and stricter collision/transit gates before training; no automatic deployment of old policies into new physics.

## Deterministic IK and ten-target feasibility study — November 2, 2026

- [Results and evidence](../baselines/mjlab-ik-stability-20261102/README.md): new opt-in target-hashed starts, independent branch convergence, source-request-aware continuation and collision-tested alternatives fix order/batch instability. All 16 saved-source rows preserve exact joint pairs; stable order/partition differences are at most 1.8e-15 rad. Original solver/physics/scorers remain unchanged. Cold 8/120 accepts 12/16 versus original 14/16 on this audit; explicitly increasing to 16 starts recovers 16/16 in a separate CPU diagnostic. Determinism is not universal reachability.
- The exact remaining indices 8,9,10,12,21,23,26,35,39,41 received 1,024 native shots each. Direct captures occur at 10,21,23,26,39, including new hits at 23. Four hundred screening attempts and 20 independent final attempts produced no qualified additions: target 21 passed 8/10 screening but only 13/20 final. All 20 final endpoints were valid; remaining failures concern ball flight/entry. The original 17-template prefix and selected checkpoint remain intact; no PPO or model deployment occurred.
- The 156-check independent audit passed. `#ikStability` reports endpoint invariance, native controls, direct/assisted findings and failed confirmations. Previous tables, solve routing and the running preview are preserved. Only 4090 native work was used; this task did not change 4070/Isaac/hardware/Viam/audio. Raw data and immutable source snapshots remain outside Git.
- Before another training run: use validated broader cold starts or exact-source continuation, then investigate airborne range and release/contact sensitivity. Do not promote table-assisted entries, lower the reliability gate or treat these development coordinates as untouched tests.

## Targeted mjlab expansion and fair frozen comparison — November 2, 2026

- [Published evidence/checkpoint](../baselines/mjlab-expanded-bank-20261102/README.md): the 12 previously unresolved test positions became development. 15,360 native search throws plus 1,000 confirmation attempts added one template qualified 9/10; original 16 remain an unchanged prefix. Six farther targets produced many table-assisted entries but zero qualified direct routes in 9,216 additional throws. No impossibility claim or scoring relaxation.
- A 132-attempt diagnostic found saved-branch 27/30 versus fresh-IK 20/30 captures, with batch-slot-dependent multistart failures. Before retraining, v2 reused exact shared-prefix masks/joints/failure reasons by stable candidate ID, including strict-to-wide overlap. The general IK initializer still needs deterministic per-query starts and branch preservation.
- 4,096 new PPO throws, 175 declared training contexts (125 eligible), 30 validation and 100 fresh test positions × 3. Same-split captures: old uniform 175/300, old learned 183/300, expanded uniform 182/300, retrained 188/300. The +2.0-point learned-versus-uniform gain has a descriptive 95% layout interval −3.3 to +7.7: no established capture advantage. Clean captures 145 versus 76 are a separate quality gain. Both banks still lack endpoints at 25/100 fresh positions.
- Converted 12-coordinate rerun: old bank 1/12 eligible/captured, expanded 2/12; ten remain unresolved. This is development, not new held-out evidence. Independent audits pass 94 search, 41 refinement, 80 far, 60 join and 428 training checks; original results are preserved. New selected checkpoint SHA `768d21bdbe104f64c87eeec96bda81395f9ffea9af14f211baa53c70f7a898bd`.
- Existing localhost preview adds read-only `#expandedStudy` results; previous inference controls remain active. New checkpoint requires its matching future adapter, with guards intact. All new native work used 4090 and the shared lock; this task did not change Isaac, 4070, Viam, hardware or audio. Raw native data/snapshots remain outside Git. Next is deterministic IK and airborne far-shot coverage, before another long PPO run.

## mjlab interactive policies and audited learning — November 2, 2026

- [New interface](../sircus/mjlab/web/README.md): drag a bucket, fix board-center or released-ball XYZ, request a frozen prediction or separately labeled native search, and inspect the actual pose/ball trace. Native kinematics drive simplified arm surfaces; this is not an Isaac viewport. Manual controls stayed gated until both learning studies and independent audits completed.
- [Published evidence and checkpoints](../baselines/mjlab-learning-20261102/README.md):8,507 continuous-policy training attempts and2,048 categorical bank-policy attempts on RTX4090. Original continuous test65/150 captures. On a fresh50-position×3 comparison: continuous75/150, uniform feasible bank94/150, learned selector93/150. The learned selector did not outperform uniform selection; proposal geometry contributes the demonstrated gain. Twelve fresh positions lack a candidate in the finite bank, not proof of impossibility.
- Both policies use privileged geometry and fixed/held joint endpoints. No learned grasp, arm transit, camera pipeline or physical deployment. Exact one-second scoring, provisional bucket-height interpretation and historical results remain unchanged.407 continuous-study and170 bank-study audit checks passed. The new localhost preview uses finite4090 child jobs; it owns no GPU between requests. Existing Isaac setup, its preview and the4070 workload were not changed by this task.
- Source snapshots and raw evidence remain under the isolated mjlab runtime on Ubuntu. Code and compact synthetic evidence are on the task branch; starting or pulling that branch does not launch a server. Next research step: expand the candidate family/endpoint feasibility for unresolved edge positions, then test a learned selector against its geometric baseline on another untouched spatial split. Do not assume that more PPO updates alone solve empty action masks.

## Simulation-only mjlab comparison — November 2, 2026

- The [isolated mjlab port](../sircus/mjlab/README.md) compiles both UF850s, the attached board, 64 bucket-wall segments, table and ball. Native geometry and 32 URDF kinematics checks passed. The bucket uses a mocap pose with actual collision-transform checks after a static-cache issue was found.
- [Fresh paired evidence](../baselines/mjlab-comparison-20261102/README.md) replays 24 identical configurations three times per engine, with fixed joint endpoints, 1920 Hz physics and the original 240 Hz scorer. All 72 starts and board bounces passed per engine. Isaac scored 24 captures (14 clean); the final mjlab batch scored 41 (5 clean). An earlier mjlab batch scored 43. Rim behavior and numerical repeatability differ, so these diagnostic counts do not establish policy improvement or arbitrary-table generalization.
- The final mjlab runner took 17.41 seconds versus Isaac's 80.11 seconds, including Python scoring and reads. A separate matched mjlab batch took 12.35 seconds. Physics-kernel and RL-training speed were not measured. Contact calibration used disjoint flat drops, still with material restitution variation. Original Isaac setup and the 4070 workload were preserved.
- The user clarified that deployment is entirely simulated. Next is an interactive bucket/arm-constraint interface with batched feasibility search, then a policy conditioned on constraint masks and values. Physical UF850/Viam deployment adapters are outside the current goal. Full trajectories and comparison media are stored outside Git.

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
