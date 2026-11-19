# Project state

## Alternate joints at the last blocked bucket position — November 15, 2026

- The four original misses at **(1028.9, −19.0) mm** struck the board arm.
  CPU reconstruction identified link 4 in one trace and link 3 in three.
- With the original board and release targets fixed, **4/16** alternate
  joint settings cleared both arms. Each missed the bucket. None of the
  four exact baseline shots captured.
- The 20 drops and 18 controls ran on the 4090 at 64 substeps. Both
  control sets and the 7,393-check CPU audit passed.
  [Experiment and reproduction](MJLAB_FIXED_POSE_JOINT_BRANCHES.md).

## Drop tests at five previously blocked positions — November 15, 2026

- At 64 substeps, **27 of 43** archived setups captured. All 27 captured
  on each of three fresh repeats (**81/81**), confirming shots at **4/5**
  bucket positions.
- At the remaining position, **(1028.9, −19.0) mm**, all four shots hit a
  robot after the board rebound and none entered the bucket fully.
- All 124 study drops and 18 controls ran on the 4090. Both control blocks
  and the 23,673-check CPU audit passed.
  [Results and reproduction](MJLAB_SETUP_DROP_RECOVERY.md).

## Ball flight ranking and pose recovery — November 15, 2026

- A pre-release flight calculation chose captures at **9/16** of the
  previously missed bucket positions. Random selection among the same
  candidates would average **5.275/16**. The calculation uses held joint
  poses, rim aperture and fitted board and table rebound velocities.
- A wider CPU search found feasible setups at **5/5** old refusals.
  It certified 43 of 160 proposed poses and paths. These were setup
  checks, with no ball releases.
- On the same 50 fresh positions, the frozen flight rule captured
  **41/50 (82%)** versus **35/50 (70%)** for direct-first. Both found
  feasible setups at 47 positions. The flight rule gained seven captures
  and lost one. Changing routes accounted for six gains.
- All 94 native drops and 18 controls ran on the 4090 at 64 substeps.
  Both control blocks and the 35,166-check CPU audit passed.
  [Protocol and evidence](MJLAB_FLIGHT_RANKING.md) · [Draft PR #133](https://github.com/kadiru/sircus/pull/133).

## Candidate replays at missed positions — November 15, 2026

- The exact selected shots repeated **0/16** captures. The 55 novel
  alternatives captured 28 times and recovered **10/16 positions**.
  Two direct-first replay controls matched their archived outcomes.
  Three setups produced no candidates within geometry limits and two
  failed arm reach or endpoint checks.
- The prior ranker and the position-held-out logistic screen chose
  captures at **3/16** and **4/16** of these difficult development
  positions. Uniform candidate choice would average **5.275/16**.
  Both screening scores fell below that expectation.
  [Protocol and results](MJLAB_MISSED_SHOT_REPLAY.md).
- All 73 test drops and 18 controls ran on the 4090. The independent CPU
  audit passed 16,665 checks. [Draft PR #132](https://github.com/kadiru/sircus/pull/132).

## Route selection vs direct-first fallback — November 14, 2026

- Both policies found a feasible shot at **95 of 100 fresh positions**.
  Direct-first fallback captured **76/100** and the route selector
  captured **78/100**. Both used the same frozen specialist, expanded
  table library and one-drop budget.
- The selector chose table over direct nine times, gaining three
  captures and losing one. All 86 pairs with identical commands
  returned matching outcomes. The small number of differing outcomes
  leaves the size of the advantage uncertain.
  [Protocol and audited results](MJLAB_DIRECT_FIRST_COMPARISON.md).
- The 4090 ran 190 test drops and 18 controls. Both control sets and
  54,781 independent CPU checks passed. The 4070 and Isaac were left
  alone. [Draft PR #131](https://github.com/kadiru/sircus/pull/131).

## One-drop route selection — November 14, 2026

- On the same 50 fresh positions, allowing a pre-release choice between
  direct and table bounce raised captures from **33/50 to 40/50** and
  feasible setups from **41/50 to 47/50**. Both planners used the same
  frozen model and had at most one drop per position.
- There were seven gains and zero losses. Five gains came where only
  table passed setup, and two came where both routes were feasible and
  the selector chose table. [Protocol and audited results](MJLAB_ONE_DROP_SELECTOR.md).
- The 4090 ran 88 test drops and 18 controls. Both control sets passed
  and the CPU audit passed 34,460 checks. The 4070 and Isaac were left
  alone. [Draft PR #130](https://github.com/kadiru/sircus/pull/130).

## Table-bounce starting shots — November 13, 2026

- The same frozen table specialist tried the original and expanded shot
  libraries on 50 fresh positions. Each library received four candidate
  slots and at most one drop per position. The expansion added 154
  verified table shots and increased source coverage from 15 to 22
  positions. [Protocol and results](MJLAB_TABLE_SHOT_LIBRARY.md).
- Feasible setups rose **23/50 → 25/50** and table captures rose
  **10/50 → 13/50**. The expansion gained four capture positions and lost
  one. The paired capture difference was six percentage points on this
  split. Both control sets passed, and the independent audit passed
  27,966 checks. The study used the 4090 and left the 4070 and Isaac alone.
  [Draft PR #129](https://github.com/kadiru/sircus/pull/129).

## Matched bounce model refit — November 13, 2026

- [Protocol and audited results](MJLAB_ROUTE_REFIT.md): retrained the pooled,
  route-conditioned and specialist models with 291 new captures. Compared
  both frozen generations on 50 fresh positions, one request per route per
  model. The starting-shot library was held fixed.
- Direct captures changed **35 → 35**, **34 → 34** and **33 → 34** out of 50.
  Table captures changed **10 → 12**, **12 → 11** and **9 → 13**. With both
  attempts available, the pooled refit had the best coverage (**41/50**).
  The table specialist gained five positions and lost one, but its setup
  failures increased from 18 to 25. This single split does not establish
  a reliable gain.
- All 600 requests, 444 test drops and 18 control drops were accounted for.
  The CPU audit passed 114,690 checks. The study used only the 4090; 4070
  and Isaac were untouched. No promotion. [Draft PR #128](https://github.com/kadiru/sircus/pull/128).

## Remaining six task completed; bouncer alternatives audited — November 10, 2026

The first pass recovered 01-05, giving **30/35 confirmed recipes**. A targeted
[saved-bouncer follow-up](../baselines/mjlab-fixed35-bouncer-20261110/README.md)
then replayed all **30/30** and tested 15 alternatives across 03-04, 03-05 and
04-06: **0/15 captures**, all 15 admitted. Thirteen hit a robot before any bucket
contact; two missed after bucket contact and hit a robot later. No whole-ball
entry or invalid physics. Cells 03-00 and 03-06 have fixed bucket/base overlap.
These finite failures do not prove the other three layouts impossible.

Native producer `ac872e8e39fa181b2e716ce04633bdd57e6a9018`, 63 requests,
572.2384 worker seconds; 18 controls behaved as expected. Separate V3 auditor
`2b795d64159fc3d9718e05744c08588e9fd15d0c` passed **35,407 checks**. Original
V1/V2 audit failures remain recorded; V3 fixes only authenticated historical
metadata handling and near-zero angular diagnostics, without changing evidence,
physics, scoring or admission. All 117 protected files/services stayed unchanged.
No UI publication, 4070/Isaac change, restart, hardware action or pending run.

The learned policy is still **24/35**. This was recipe development, not training.
Its nine input features produce five shot-parameter residuals, not an IK branch.
Future training must retain/select the confirmed joint configuration as well as
shot parameters. [Workflow and handoff](MJLAB_FIXED35_BOUNCER_20261110.md).

## Remaining six fixed-grid positions — first pass completed November 10, 2026

[Audited baseline](../baselines/mjlab-fixed35-six-20261110/README.md): 641 requests, 87 native releases, zero invalid physics. One new table-assisted recipe at 01-05 passed 3/3 repeats and all 29 prior recipes recaptured. **Recipe coverage 30/35; unchanged policy 24/35.** All 170,755 independent checks passed, without time truncation. All 112 protected files stayed unchanged. No UI publication, service restart, 4070 or Isaac change. Producer `e6a11db6b0c00c278874d04800b1b38e78080040`; draft PR #123.

Three searched cells each exhausted 96 numeric poses × two branches, but only four poses/eight flights admitted per cell; zero captures. Two other cells have actual fixed-base overlap. Final review found a selection limitation: all selected second branches changed only dropper wrist posture; 18 endpoint-valid bouncer alternatives across 11 poses remained untested, including seven shoulder/elbow variants across four poses. The separate targeted replay is now completed above, preserving the first-pass evidence and all 30 recipes. No retraining started. [Workflow/handoff](MJLAB_FIXED35_SIX_20261110.md).

## Fixed-grid failure recovery completed — November 9, 2026

Five of the 11 original failures now have confirmed recipes, each with **3/3 fresh native64 captures**. All **24/24 unchanged parent recipes** captured again: **29/35 fixed grid positions have confirmed recipes**. This is development search on the original coordinates; the existing policy remains **24/35**, with no fitting, replacement or promotion. [Audited baseline](../baselines/mjlab-fixed35-recovery-20261109/README.md), [protocol](../sircus/mjlab/FIXED35_RECOVERY.md), [handoff](MJLAB_FIXED35_RECOVERY_20261109.md).

- **441 requests**: 384 search, 15 fresh confirmations, 24 preserved replays and 18 controls. All control gates and **141,041 independent checks** passed. Worker 4,105.75 s; pipeline 4,135.29 s. Four searched cells exhausted 80 proposals each; the study was not time-truncated. Three invalid-physics attempts at grid 02-03 triggered `through_wall_after_entry` and remain failed, separately identified evidence.
- Recovered grid cells: **00-00, 01-06, 02-03, 02-06 and 04-04**. Recipes at 01-06 and 02-06 used table assistance. All retain board bounce, whole-ball entry and one continuous second inside under unchanged v3 scoring. Exact values and original successes are preserved in the baseline, not automatically activated.
- Unresolved: **01-05, 03-04, 03-05, 04-06**. The latter three are limited mainly by early robot obstruction and/or finite IK admission. At 01-05, recorded robot contacts followed bucket/table contact, so they do not establish the initial miss cause. Two other cells, **03-00 and 03-06**, remain below the existing fixed-base clearance bound; no bucket relocation or guard relaxation. None of these finite failures proves impossibility.
- Frozen producer `7bcba3980a1baf24d6ac2505e35189b95daead64`; [draft PR #120](https://github.com/kadiru/sircus/pull/120), stacked on #119. Full local tests passed (2,235 Python tests with 100 skips; 390 JavaScript tests), source CI passed, and the completed UI was visually verified. Final isolation passed **31/31 checks**: the owned 4090 worker exited, the 4070 workload retained its process identity, historical reports/checkpoints and Isaac assets stayed intact. No service restart or physical action.
- `#fixed35Recovery` displays all 11 cases, exact successful settings, fresh repeats, preserved recipes and the request ledger. No new PPO, noise/128-substep qualification, bucket perturbations, dynamic arm transit or grasp validation. Next decisions can use the six remaining cases' distinct limitations; no further experiment has been launched.

## Fixed 35-position imitation comparison completed — November 9, 2026

- [Audited results](../baselines/mjlab-fixed35-imitation-20261109/README.md), [protocol](../sircus/mjlab/FIXED35_IMITATION.md) and [handoff](MJLAB_FIXED35_IMITATION_20261109.md): current imitation captured **24/35 (68.6%)**, refitted imitation **23/35 (65.7%)** on the same canonical 7×5 grid. One gain, two losses, 22 shared captures and ten shared failures. Keep the current policy; the refit did not improve this sample and was not promoted.
- Exactly **35 fixed locations**, no bucket perturbations: 70 frozen model requests plus 18 original controls, **88 total slots**. Both policies used bucket-facing gripper yaw and unchanged native64/v3 scoring. This does not compare yaw against the old orientation. Twenty-three evaluation points exactly overlap teacher coordinates; all 35 are historical development grid locations, so this is coverage/regression, not held-out or all-table generalization.
- Refit/current setup declines: **6/3**; native releases **29/32**; released noncaptures **6/8**. The refit's six include **one invalid-physics attempt** at `(1050, −200) mm`, separately identified rather than treated as an ordinary flight miss. Table-assisted captures: **3/2**. The audit validates evidence and accounting; it does not assert that every rollout stayed within the calibrated physics domain.
- Candidate training retained 97 historical action targets and added two confirmed bucket-facing recipes, one equal-weight example per layout: 99 teachers and 800 actor-only Adam steps at 0.001. Both checkpoints and all 70 commands froze before evaluation. Original models, successful recipes, historical orientation labels and geometry anchors remain preserved. No PPO or evaluation retries/search.
- All 18 pre/post controls and **40,692 independent checks** passed. Worker **1,100.49 s**, pipeline **1,114.97 s**. Frozen producer `3fa71e2f1bc658aa62d8ed56f7bbfaff67093968`; [draft PR #119](https://github.com/kadiru/sircus/pull/119) stacks on #116. Source CI passed; 48 Ubuntu tests, the Mac publisher/browser interoperability check and ten UI tests passed.
- The completed `#fixed35Imitation` panel was visually verified with its 35-cell matrix, exact commands, paired counts and invalid-physics label. All 27 final isolation checks passed: the 4090 worker exited, the protected 4070 workload and viewer retained their identities, historical reports and unrelated UI content stayed intact. No service restart, physical hardware action or Isaac change. Next: inspect the two lost captures and added setup declines before another refit; keep bucket perturbations deferred.

## Farther bucket-facing shot recovered by tilt/gap tuning — November 9, 2026

- [Audited settings](../baselines/mjlab-yaw-tilt-gap-20261109/README.md), [protocol](../sircus/mjlab/YAW_TILT_GAP.md) and [handoff](MJLAB_YAW_TILT_GAP_20261109.md): fixed board center **(653.661, −203.412, 449.818) mm**, bucket distance **562.434 mm**, gripper yaw **−36.285503°**. A 49-point tilt/gap grid found ten captures; all ten passed **3/3 fresh repeats**. At the original 30.3307° tilt, increasing gap **75.650 → 95.650 mm** changed the rotated baseline **0/3 → 3/3**.
- **106 requests / 92 native releases / 14 finite IK declines**, zero invalid physics. Discovery **10/49**, confirmations **30/30**, successful far/near references **6/6**. No discovery or confirmation robot contact. All 18 formal controls and **50,615 independent checks** passed; worker **2,000.63 s**. All 40 new captures recorded table contact; representative saved timings establish a table bounce before entry. Unchanged native64/v3 scoring permits table/rim assistance.
- Frozen producer/UI **`c9d0248743790e862cd8053f1681f4ab24867e31`**, [draft PR #116](https://github.com/kadiru/sircus/pull/116), stacked on #114. Successful recipes, model weights, physics and historical **40/70 versus 38/70** policy scores remain unchanged. This is one inspected development layout with held endpoints, not a retrained policy, noise/timestep qualification, learned transit or physical validation.
- `#yawTiltGap` passed deployed visual/scoped-DOM verification. Its **55.8 s / 1,674-frame** CPU-rendered saved-native comparison fully decoded and passed browser playback. All 11 inventoried files stayed unchanged outside owned HTML/CSS blocks; viewer identity stayed fixed and the 4090 worker exited. The 4070 fountain matched through the post-study check, then appeared under a new PID during publishing; this task issued no stop/restart or Isaac/4070 configuration change. Only the Mac SSH tunnel was restored. Raw traces/runtime configuration/media remain outside Git; no further experiment is launched.


## Bucket-facing gripper yaw removes both selected collisions — November 9, 2026

- [Audited results](../baselines/mjlab-gripper-yaw-20261109/README.md), [protocol](../sircus/mjlab/GRIPPER_YAW.md) and [handoff](MJLAB_GRIPPER_YAW_20261109.md): rotating only the dropper's commanded orientation reduced robot-contact shots **6/6 → 0/6** and improved captures **0/6 → 3/6**. Successful current-policy references stayed **6/6**. Two inspected development layouts, three nominal repeats of three frozen variants each; no new policy score.
- Nearer case `r1-c2-n0` (172.610 mm board–bucket distance) recovered **3/3** at yaw **−150.5791°**. Farther `r0-c6-n1` (562.434 mm) remained **0/3**, but cleared the robot at **−36.2855°**. Board X/Y/height, tilt, gap, bucket, original board joints and release centers stayed fixed versus each corrected recipe. Only dropper orientation and the required dropper joints changed; new full-rotation proofs passed.
- All **36 requests / 36 native releases** completed in **315.254865 worker seconds**, with zero setup declines or invalid physics. All 18 bracketing controls and **18,330 independent checks** passed. Frozen producer/UI `26ba197bc1f85150f6b76f52c27ed0173f9b5c0c`; draft PR #114. Physics/scoring remain native64/v3; no policy fitting, promotion, shared IK default change, service restart or hardware command. 4070 and Isaac remain outside the task.
- Audit checks include unchanged board branch and model FK of observed float32 reset joints, with new yaw errors below 8.3 × 10⁻⁸ rad in first repeats. Arms remain held at admitted endpoints during flight; this does not validate moving-arm transit, real grasp/release, noise robustness or all-table generalization. Historical **40/70 current versus 38/70 corrected imitation** scores and all successful recipes remain unchanged.
- Two saved-native comparison movies (33.067 s / 25.867 s, 1,768 frames) fully decoded and passed Chrome playback in `#gripperYaw`. CPU rendering; UI-only provenance fix `ba79932` preserves frozen native source. Seven historical report hashes, 57 display keys/bytes, unrelated HTML/CSS and protected viewer/4070 process identities stayed unchanged; the 4090 worker exited. Next: inspect the farther layout's now-clear miss before another intervention or training decision. No extra experiment launched. The earlier midpoint trial remains historical **0/72 executed**; this yaw study is a separate authorized test.


## Six imitation regressions diagnosed; correction blocked by GPU occupancy — November 9, 2026

- [Diagnosis and provenance](../baselines/mjlab-imitation-regression-20261109/README.md), [protocol](../sircus/mjlab/IMITATION_REGRESSION.md) and [handoff](MJLAB_IMITATION_REGRESSION_20261109.md): three finite IK refusals, two robot-contact routes, one rim miss. The 1,230-check diagnosis passed. Historical current/corrected scores stay **40/70 versus 38/70**; successful recipes and checkpoints remain intact.
- One frozen 50% interpolation of all five shot coordinates toward each successful current recipe; 54 matched diagnostic slots plus 18 controls at 64 substeps. **0/72 ran:** the 900-second GPU queue expired before acquisition because a protected Isaac job occupied the 4090. This is not a correction failure or completed native experiment. Producer `b962af9`, UI `6f489d1`; draft PR #113 stacks on #112.
- The additive `#imitationRegression` UI shows exact historical differences and six historical success/failure movies (2,705 frames, all fully decoded). CPU software rendering replays saved native centers with held endpoint joints and explicit refusal cards. Browser playback verified. No new correction movie/outcome is claimed.
- Source tests/CI and 197 actual-plan preflight checks passed. Seven inventoried report hashes, all 57 display-report keys/bytes and unrelated HTML/CSS remain unchanged; protected fountain/viewer process identities preserved. No model fitting/promotion, hardware work, service restart, 4070 or Isaac change.
- Next: a new bounded launch of the unchanged 72-slot protocol when the 4090 is free, then independent audit and actual matched-results publication. No automatic retry remains. These inspected positions are development data, not fresh generalization evidence.

## Corrected imitation evaluated on 70 new positions — November 9, 2026

- [Audited results](../baselines/mjlab-corrected-imitation-20261109/README.md), [protocol](../sircus/mjlab/CORRECTED_IMITATION.md) and [handoff](MJLAB_CORRECTED_IMITATION_20261109.md): corrected imitation captured **38/70 (54.3%)**, current imitation **40/70 (57.1%)** on the same fresh positions. Four gains, six losses, 34 shared captures and 26 shared failures. No sample improvement or policy promotion; this small paired test does not establish significant general worsening.
- Candidate retained all 92 teachers, added five previously confirmed corrections and took 800 actor-only Adam steps at 0.001. Current weights stayed unchanged. Both models and all 140 commands froze before any TEST outcome. The study measures the combined extra data and fitting steps. No PPO or TEST search/retries.
- Seed `2026110941`: two new positions per 7 × 5 grid cell, at least 5 mm from inventoried history/training and each other. Corrected/current setup declines: **23/19**; released misses: **9/11**; table-assisted captures: **3/2**; invalid physics: **0/0**. Every request counts. The older **37/70** score remains historical and used different positions.
- All **158 requests** completed; 18 pre/post controls and **55,633 independent checks** passed. Worker **1,387.39 s**, pipeline **1,414.47 s** elapsed. Frozen producer/UI `fecbe4d61bd92c912e0b4d80b1fdd7be3a2bcb48`; draft PR #112 stacks on #111. Native64/v3 physics/scoring, geometry prior and original control namespaces stayed fixed.
- The audit-gated, visually verified `#correctedImitation` panel shows all 70 paired positions, teacher lineage, exact corrections, checkpoint hashes and controls. Its 75,337-byte snapshot preserves all 57 existing report keys/bytes and historical panels. Only the Mac SSH tunnel was restored; no service restarted. Final isolation confirmed the 4090 worker exited and protected 4070/viewer process identities and historical report hashes stayed unchanged; Isaac untouched.
- Next: diagnose the six regressions (three finite IK declines and three released misses) against their successful current-policy recipes. Keep the existing policy and all successful recipes. No claims of all-table coverage, moving-arm transit, perception, physical validation, noise/timestep robustness or statistically established worsening; no further experiment was launched.

## Two remaining IK declines recovered with feasible alternatives — November 9, 2026

- [Audited results](../baselines/mjlab-regression-ik-20261109/README.md), [protocol](../sircus/mjlab/REGRESSION_IK.md) and [handoff](MJLAB_REGRESSION_IK_20261109.md): all **66 slots** finished in **203.8 worker seconds**; 18 formal controls and **25,329 independent checks** passed. Frozen producer/UI `2ec2c0b`; draft PR #111 stacks on #110. No IK algorithm, admission tolerance, physics or scoring change.
- Both selected cases captured **3/3** after restoring only parent board X/Y on their original recovery actions. `r1-c1-n1` moved X/Y **−5.892 / +23.525 mm**, changing finite valid branch counts from **0/9 to 10/8** (dropper/board). `r2-c4-n0` moved **+13.693 / +2.605 mm**, changing **8/0 to 7/7**. Requested height, tilt, gap and bucket X/Y stayed fixed; target-facing yaw and solved joints can change with X/Y.
- Eight frozen variants per case, three requests each: exact parent **6/6**, original recovery **0/6** cached declines, parent X/Y **6/6**, parent height+gap **3/6**, parent tilt **0/6**, and 25/50/75% blends **6/6 each**. Of 48 diagnostic requests, all **33 admitted drops captured**, with no diagnostic robot contact; 15 requests declined, including six cached originals. Nine of twelve new configurations captured all repeats (27/36 new-action requests). No outcome-driven winner or extra trials.
- All diagnostic captures were direct board → bucket; direct does not mean rim-free. Both control blocks retained three direct captures, three table-assisted captures and three expected misses each. Total: 51 releases, 45 captures, six intentional robot-contact misses, 15 declines, zero invalid physics. Refused recipes have no flight or measured posture; new actual postures are compared with the matched successful parent.
- The audit-gated, visually verified `#regressionIK` panel shows all 16 variants, branch/refusal details, exact commands, controls and actual first-repeat traces. Its 99,300-byte snapshot preserves all 57 prior report keys/bytes and historical HTML/CSS outside the addition. The Mac SSH tunnel was restored; the viewer was not restarted. Final isolation confirmed the 4090 worker exited and the 4070 workload/viewer retained their process identities; Isaac and historical report hashes remain unchanged.
- Combined with earlier height and drop-gap corrections, all five inspected lost captures have successful local recipes. Original **37/70** policy performance remains unchanged. These are development positions and nominal repeats, not new generalization or noise/timestep qualification. Preserve recipes, then make a separate training/selector decision and frozen evaluation on fresh positions. No model fitting, automatic fallback/promotion, hardware action or further experiment was launched.

## Two flight losses recovered by drop-gap restoration — November 9, 2026

- [Audited results](../baselines/mjlab-regression-drop-gap-20261109/README.md) and [handoff](MJLAB_REGRESSION_DROP_GAP_20261109.md): all **36 requests / 36 native releases** completed in **170.2 worker seconds**, with 18 formal controls and **16,072 independent checks** passing. No setup declines, cached refusals or invalid physics occurred. Frozen producer `1e37070`; draft PR #110 stacks on #109.
- Exact parent recipes captured **6/6**; original recovery recipes **0/6**; restoring only parent drop gap on the original recovery action **6/6**, or **3/3 for each case**. Gaps decreased **133.998 → 103.431 mm** (`r0-c5-n1`) and **147.678 → 109.932 mm** (`r0-c6-n1`). Other requested coordinates stayed fixed; the earlier height correction was not combined with this intervention.
- All corrected captures used **board → table → bucket** routes, with whole-ball entry followed by one continuous second retained. Fresh admission selected unchanged board-arm joint vectors and identical measured board centers/normals. Dropper joints changed to achieve the lower release. Diagnostic shots had no robot contact; the six formal negative controls did and remained misses.
- The additive `#regressionDropGap` panel shows exact settings, all variant outcomes, native first-repeat traces, posture comparisons and controls/audit. Its 62,097-byte snapshot preserved all 57 prior report keys/bytes and historical HTML/CSS, including the previous movie, without restarting a service or activating a selector.
- Original **37/70** recovery-imitation performance is unchanged. These inspected positions are development data; 3/3 nominal repeats are not noise robustness or all-table generalization. Parent recipes and historical evidence remain intact. No new PPO/model fitting, 128-substep trial, hardware work, 4070 workload change or Isaac change.
- Combined with the earlier height correction, three of five inspected lost captures now have local corrections; two finite IK declines remain. Retain the validated recipes, then make a separate training/selector decision and evaluate on fresh positions before claiming policy improvement. No further experiment is automatically launched.

## Recovery regression diagnosis completed — November 9, 2026

- [Audited correction](../baselines/mjlab-recovery-regression-20261109/README.md) and [handoff](MJLAB_RECOVERY_REGRESSION_20261109.md): all **63 requests / 48 native releases** finished in 221.1 worker seconds; all 18 controls and **22,266 independent checks** passed. No physics-invalid attempts. Producer `a95e303` is frozen; draft PR #109 stacks on #106.
- On the five selected capture losses, exact parent recipes repeated **15/15 captures**, recovery recipes **0/15**, and restoring only parent board-height parameter **3/15**. Only `r3-c1-n0` recovered, with 3/3 clean direct captures after height 241.129→280.424 mm. Fixed drop gap means absolute release height moves by the same amount. Two flight regressions still miss and two finite IK cases still decline.
- The nine old refusal slots reuse original decisions; the six corrected refusal slots reuse two new strict proofs. They are not independent repeated IK searches. No diagnostic robot contact; the six robot-contact misses are intentional negative controls. Every request remains in the denominator.
- CPU diagnosis accounts for all 16 setup declines and 17 released misses; all 17 missed whole-ball entry, seven had forbidden contact. Detailed settings, actual saved trajectories and separate correction outcomes appear in `#recoveryRegression`. The original 70-position result remains 37/70 versus 34/70; this posthoc correction is not a new policy score or generalization claim.
- No new PPO, policy activation, scoring/guard changes, service restart or hardware work. The 4090 worker exited; 4070 and Isaac remain outside the task. Next: retain the verified parent recipes and test a separate drop-gap hypothesis for the two contact-free flight regressions before more learning.
- The panel's 60.13-second 3D replay shows the two historical parent/recovery flight pairs and a representative robot-contact miss, using saved ball samples and held joints. All 1,804 frames decoded; Chrome playback and corrected sample replay were verified. Actual corrected traces are separate from the historical movie. All 57 existing report keys/bytes and historical HTML/CSS outside this panel were preserved; no restart.

## Completed bounded coverage recovery — November 9, 2026

- [Audited results](../baselines/mjlab-coverage-recovery-20261109/README.md), [protocol](../sircus/mjlab/COVERAGE_RECOVERY.md) and [handoff](MJLAB_COVERAGE_RECOVERY_20261109.md): all **970 requests** completed in **96.8 worker minutes**, with both control gates and **283,251 independent checks** passing. Frozen producer `5224093` remains unchanged.
- Search completed 624 proposals, with 83 captures across **24/39 development layouts**; all 24 selected recipes passed **2/2 fresh repeats (48/48)**. Actor-only imitation used 68 original plus 24 new teachers. No additional PPO ran.
- On the same **70 fresh TEST positions**, recovery imitation captured **37/70**, parent imitation **34/70**, parent selected PPO **34/70**, and earlier PPO **31/70**. Recovery still has **16 setup declines and 17 released misses**. Three baseline invalid-physics attempts (two parent imitation, one parent PPO) stay in the denominator and are separately reported; they are not ordinary valid-flight misses. Recovery gained eight/lost five against parent imitation, nine/lost six against parent PPO, and eight/lost two against earlier PPO. None of these gains overlaps an invalid baseline attempt.
- The parent's selected PPO had 22 setup declines and 20 released misses. Thirteen declines lacked a finite IK solution, four failed endpoint checks, three failed fixed-base clearance, and two were out of action bounds. Eighteen released misses never achieved whole-ball entry; two retained for one second but had robot contact. Table rebounds remain allowed.
- The three fixed-base blockers remain reported. Former TEST failures are explicitly development data now; their historical report is untouched. Nominal 2/2 repetition is not noise robustness, and extra teacher exposure prevents an equal-training-budget algorithm claim.
- Four frozen models share the same geometric prior on 70 new stratified TEST positions: recovery imitation, parent imitation, parent selected PPO and strongest earlier PPO. All requests, including declines, count. No per-position outcome optimization during TEST, extra PPO, noise, 128-step qualification, hardware action or policy promotion.
- Only the 4090 and unchanged native64/v3 physics/scoring were used. Worker, supervisor and finite publisher exited within their caps; the 4070 fountain, viewer and Isaac setup were left untouched. Raw evidence remains outside Git. The additive `#coverageRecovery` panel does not activate a selector or restart a service.
- No automatic promotion or further experiment. The remaining issue is useful unseen-position coverage, not reproducing the 24 confirmed development recipes; inspect remaining failures and lost captures before another training decision.

## Completed seven-hour-budget coverage training — November 9, 2026

- [Audited results and all 70 paired positions](../baselines/mjlab-coverage-training-20261109/README.md): new imitation **28/70 (40%)**, selected new PPO **28/70 (40%)**; 27 shared captures, one gain, one loss, 41 shared failures. Previous weights scored imitation 28/70 and PPO 31/70 under the same expanded 19-pose prior. No net new-PPO improvement and no policy promotion.
- All **2,757** requests completed in **235.1 worker minutes**: 1,260 guided TRAIN search, 1,024 PPO (512 per seed), 175 validation, 280 TEST and 18 controls. Search produced 532 captures and 68 successful teacher layouts out of 105 TRAIN. Validation improved 22/35 → 25/35 for the selected 512-request checkpoint, but not on unseen TEST. Both new models have 22 setup declines and 20 released misses; no TEST invalid physics. Twelve invalid physics shots during search/training remain failed attempts.
- Independent CPU audit passed **857,507 checks**. The original checker stopped on a mixed-unit float32 tolerance; corrected revision `normalized-command-units-v1` preserves normalized tolerance and adds exact saved-action algebra checks. Both original failed proof and frozen producer `1952dd5993f3d3dd0fdb52ea4eff2dbc72d36862` remain unchanged. [Verification provenance and follow-up](MJLAB_COVERAGE_TRAINING7H_20261109.md).
- UI `#coverageTraining7h` now has audited four-model scores, paired takeaway, validation curves and all 70 position outcomes. One frozen prediction/attempt per position; finite coverage, not all-table success or reliability. Held-endpoint native64/v3 only, not learned grasp/transit/perception. Results published without service restart; 4090 worker finished, 4070 fountain and Isaac left untouched.
- Next: diagnose 22 setup declines and 20 flight misses, and 37 TRAIN layouts without successful teachers, before another long PPO run. No automatic new experiment or interactive policy activation. Draft PR #102 remains stacked on #100.


## Completed Case 00 tilt/gap search — November 9, 2026

- [Exact successful settings and evidence](MJLAB_CASE00_TILT_GAP_20261109.md): fixed board center **(597.992, −23.557, 460) mm**, bucket **(1109.540, −221.464) mm**, gap **251.819 mm**; changing tilt to **26° or 32°** produced one discovery capture and **3/3 fresh captures each**. Original 28° baseline remained 0/3. Release-ball center Z is 749.319 mm; horizontal bucket distance 548.496 mm.
- Both recipes captured with bucket/rim assistance, no table or robot contact; neither was a clean entry. The 32° repeats partially exited after first entry and subsequently completed a continuous second inside. Capture was measured by unchanged v3 scoring, not inferred from collision avoidance or reward.
- All 70 grid proposals were checked: **42 admitted, 28 declined** (21 finite IK unresolved, 7 strict endpoint clearance/pose). All 42 discovery shots bounced the board; two captured and 40 missed, 19 with robot contact. Declines remain unflown evidence and do not prove impossible poses. Both discovered captures received all three frozen confirmation repeats; no cherry-picking after repeats.
- Total **97 requested configurations/repeats, 69 native drops**; 18 controls passed (12 captures, 6 expected misses). Independent audit passed **27,547 checks** with no invalid physics. CPU admission 69.9 s; native wall time 258.8 s. Frozen producer `71fb34aa8fd35ebfc1cd0d1ee522ad5181a0234d`; draft PR #100 stacked on #98. [Compact results](../baselines/mjlab-case00-tilt-gap-20261109/README.md) feed `#case00TiltGapStudy`.
- No new PPO, checkpoint/selector activation, noise or 128-substep experiment. This is repeated success in one previously inspected layout, not generalization or a guaranteed success interval. Next: finer bounded tilt/gap mapping around 26°/32°, then frozen evaluation on new bucket positions. The 4070 and Isaac remain untouched.

## Completed bouncer-clearance XY search — November 8, 2026

- [Exact configurations and interpretation](MJLAB_BOUNCER_CLEARANCE_20261108.md): 75 frozen XY proposals across cases 00/18/26 produced **9 admitted setups and 66 finite IK declines**. Only ΔX=0 and ΔY=0/+60/+120 mm admitted in each case. Board height, tilt, gap and bucket XY stayed fixed; moving board XY changed heading and horizontal bucket distance. All failed proposals remain evidence; finite IK failure does not prove impossibility.
- **0/9 discovery captures, zero entries, 7 robot-contact shots**. Case 00's +60/+120 mm Y alternatives cleared the robot. The selected +120 mm Y setting stayed contact-free in **3/3 fresh repeats**, compared with robot contact in **3/3 original repeats**, but still captured **0/3** and never entered. Cases 18/26 remained colliding; no colliding substitute was repeated.
- Case 00 selected: board center **(597.992, −23.557, 460) mm**, bucket **(1109.540, −221.464) mm**, tilt 28°, gap 251.819 mm. Original board Y was −143.557 mm; horizontal distance changed 517.446→548.496 mm. This is a contact-free miss, not a successful recipe. Table contact remains allowed.
- All 18 direct/table/miss controls passed; total **105 requested configurations/repeats, 39 native drops**. Independent audit passed **17,380 checks** with no invalid physics. CPU admission 34.5 s; native wall time 208.3 s. Frozen producer `a6fe6af7afce0ecfe667bbe7efbfdd9f8f39b437`; draft PR #98 remains stacked on #97. [Compact results](../baselines/mjlab-bouncer-clearance-20261108/README.md) feed `#bouncerClearanceStudy`.
- The panel includes a **43.4-second recorded native64 replay**: matched original/clear Case 00, unresolved shifted 18/26 and a table-assisted control. Actual saved ball positions and held arm joints, identical Case 00 cameras, no new physics or invented motion. CPU-only rendering; all 1,302 frames passed complete MP4 decoding. Existing UI was updated without restart; other panels/report keys preserved. Chrome verified counts, chapters, playback and speed controls.
- Failures are expected search data, not grounds for suppressing all exploration. No PPO, new hard refusal rule or policy promotion. Next: bounded tilt/gap search around the cleared case 00 family and a separately audited branch/posture alternative for 18/26 before another policy comparison. Native64/v3 physics and scoring stayed fixed; the 4070 and Isaac were left untouched.

## Recorded height/tilt failure replay — November 8, 2026

- [Video interpretation and evidence](MJLAB_HEIGHT_TILT_REPLAY_20261108.md): a 47.63-second, 1280×720/30 fps MuJoCo replay shows three saved misses and a table-assisted positive control. Actual ball samples, held endpoint joints, original geometry, 1×/¼× views and exact contact freezes; zero new physics steps. No unrecorded arm transit or ball spin is invented.
- Cases 00 g23 and 18 g23 first hit the bouncer at 0.476/0.468 s. CPU reconstruction identifies link 2/link 3; the original native force logs record aggregate robot contact only. Case 01 g17 has no robot contact but strikes the exterior bucket wall below the rim at 0.596 s. The positive control completes one-second retention at 2.084 s. Discovery remains 0/58 captures.
- `#heightTiltReplay` adds playback, chapter buttons, speed control, exact event times and an MP4 download. Media/frame hashes bind to the completed 36,371-check audit. Full movie/trace artifacts stay outside Git; compact provenance and contact evidence are committed. CPU llvmpipe rendered the movie without either GPU; no Isaac changes or server restart.
- Validation: 35 relevant unit tests, complete MP4 decode, 1,429-frame/duration checks and Chrome playback/chapter/speed verification. The existing in-app browser automation was unavailable; Chrome verified the same live interface. Next: test an admitted rebound path clear of the bouncer itself, then correct the unobstructed short flight before more training. No new corrective experiment or policy promotion was performed here.

## Completed height/tilt admission search — November 8, 2026

- [Audited evidence and all 120 grid entries](../baselines/mjlab-table-height-tilt-20261108/README.md): cases 00/01/18/26 tested board heights 320/360/400/440/460/480 mm and tilts 16/20/24/28/32°. Bucket XY, board XY, drop gap, orientation convention, IK/clearance checks, native64 physics and v3 scoring stayed fixed. This is development search on inspected layouts, not a new policy evaluation.
- **58/120 setups admitted**, split 13/14/17/14 by case. The 62 declines were finite IK failures: 52 board-only, eight dropper-only and two both. All 24 requests at 32° lacked board-arm branches. At 28°, case 00 admitted 460 mm; case 01 admitted 440/460 mm; case 18 admitted 460/480 mm; case 26 admitted none. These finite searches do not prove impossibility outside the tested solutions.
- **0/58 discovery captures and zero whole-ball entries**, despite a board bounce in all 58. Robot contact occurred in 42 shots (9/2/17/14 by case). Original repeats were 0/12 captures with six robot-contact shots. No successful candidate qualified for the conditional repeat stage. All 18 bracketing controls behaved as expected; no invalid physics. Total: **150 requested configurations/repeats, 88 native releases**.
- Frozen producer `380ac3e73b8d6b08504a31ad3797e5088b78af4e`; independent audit passed **36,371 checks**. CPU admission took 101.2 s and native phase 218.9 s of wall time, not isolated GPU compute. Full source/scene/admission/trace bindings remain in the local evidence; compact counts, grid and hashes are committed.
- The additive `#tableHeightTiltStudy` UI displays exact settings, failed IK searches, native misses and controls. No successful posture is installed. The finite 4090 worker exited; the existing 4070 job and viewer retained their PIDs/start times. No Isaac change, service restart, hardware action, new PPO or policy promotion. Next: identify the ball-path obstruction and test an admitted geometry/clearance change before further training; retain table-assisted success.

## Table-assisted failure diagnosis and tilt correction — November 8, 2026

- [Evidence and exact configurations](../baselines/mjlab-table-training-diagnosis-20261108/README.md): all 31 new-PPO released misses lacked whole-ball entry. First contact after board rebound was table 16 / bucket 8 / robot 7; 13 had forbidden contact at some point. Table contact remains allowed and is not itself a failure. Source/accounting diagnosis passed 143 checks, inheriting the original independent 823,442-check native audit.
- Original setup declines: cases 06/36 found 9/6 dropper IK branches but no board-arm branch; cases 14/46 could not meet fixed-base/bucket clearance. Finite search failure is not proof of unreachable geometry, and a zero conservative clearance lower bound is not proof of penetration. No admission checks were relaxed.
- Frozen producer `fe7f9b3` tested only tilt 32° for shallow far-shot cases 00/01/18/26, three original and three changed requests each, with 18 bracketing controls. **42/42 requests completed in 431.8 worker-seconds**; independent audit passed **12,557 checks**. Original: 0/12 captures, 12 releases, 6 robot-contact shots. Changed: 0/12 captures, 0 releases, 12 `finite_ik_unresolved` declines; valid board branches remained zero in all four cases. Both control gates passed and no physics invalids occurred.
- This is a failed tilt-only correction, not evidence that robot contacts were fixed. Original joint branches, 64-substep physics, v3 scoring and all historical results remain unchanged. Former TEST layouts are now posthoc development evidence; no new model or generalization result. Next: find jointly admitted height/tilt combinations before another PPO run.
- The read-only `#tableTrainingDiagnosis` panel shows all 35 failures, actual saved trajectories/contact times, setup proof details and separate correction results. Raw observations stay outside Git. The finite 4090 worker exited; existing 4070 workload and viewer start time were unchanged. No Isaac changes, service restart, hardware or policy activation.

## Table-assisted short training completed — November 8, 2026

- [Verified results and evidence identities](../baselines/mjlab-table-assisted-training-20261108/README.md): on the same 50 fresh TEST bucket positions, old imitation **9/50 (18%)**, old PPO **8/50 (16%)**, new imitation **14/50 (28%)**, selected new PPO **15/50 (30%)**. Each model made one frozen prediction per position. Setup rejections remain in the denominator.
- New PPO recorded nine direct captures and six captures with table contact, four setup rejections and 31 released misses. New imitation already achieved 14 captures; PPO's additional one capture does not establish a meaningful incremental advantage. Revised demonstrations, rewards and historical exposure changed together, so this is not a reward-only or equal-budget causal comparison.
- All **766** requested shots completed: 512 TRAIN / 16 PPO updates, 36 validation, 200 TEST and 18 controls. Validation captures were 5/12 → 5/12 → 6/12; the 512-request checkpoint was selected before TEST. All controls passed; the final audit passed **823,442** checks. No TEST invalid physics. Worker elapsed time was 76.4 minutes, not isolated GPU compute time.
- Frozen producer `b11e8b9`; protocol, physics and exact-action two-arm admission stayed unchanged. Historical results retain their original definitions. The other Isaac chat's feasibility work (`fcdefe6`, PR #91) was reviewed; no controller/scorer was transplanted.
- Final audit-bound results are visible in the existing read-only `#tableTraining` panel. The missing Mac loopback SSH forwarding connection was restored; no Ubuntu viewer, robot service or GPU job was restarted. The 4090 study exited normally; the 4070 and Isaac workloads were untouched. No interactive-policy promotion.
- Next decision: diagnose the 31 released misses (13 with forbidden contacts, 18 without) and four setup rejections (two finite IK unresolved, two fixed-base clearance). Six table-contact captures cluster in one far-table region; this is not all-table coverage or repeatability validation. Replicate on fresh layouts/seeds before a longer run or promotion.

## Table-assisted scoring and historical rescoring — November 8, 2026

- The user authorized version 3 scoring/rewards that allow board → table →
  bucket captures, with the same measured board rebound, whole-ball entry and
  one continuous second retained. Only table contact is removed from the old
  forbidden-contact rule; rim assistance, retention resets, physics validity,
  floor/robot/unattributed exclusions and numerical reward weights are unchanged.
- [Contract](../sircus/mjlab/SCORING_TABLE_ASSISTED.md): separate production and
  independent-oracle modules preserve the original v2 source and labels. An
  opt-in environment adapter scores the same native rollout once under v3 and
  retains its v2 labels. Future training must bind a new experiment contract.
- [Completed CPU audit](../baselines/mjlab-table-assisted-scoring-20261108/README.md):
  8,037 native trajectories from 17 compatible studies; accepted captures
  **4,409 → 4,532** (+123), zero lost captures, 64,883 checks passed. Original
  observations and labels are untouched. 3,018 compatible requests remain
  rejected/withheld; 71,155 unsupported historical index entries stay excluded.
- New `#tableAssistedScoring` panel shows before/after counts, reward examples
  and separate study/method/phase/substep groups. Its data is a verified embedded
  snapshot; publishing it requires no server restart. The historical movie stays.
- Current policy64 TEST counts remain imitation **20/50** and PPO **18/50**.
  The extra credits do not establish a better policy. No new physics, model
  training, policy activation, 4070/Isaac changes or hardware operations occurred.
- Producer `8eb5452418238eece6fb531ec0e446c496d55ee2` is frozen. Future root-wide
  indexing excludes known derivative reports instead of counting them as trials.

## Table-assisted task clarification and recorded replay — November 8, 2026

- The user explicitly accepts board → table → bucket captures with one second
  of retention. The old table-contact exclusion was an implementation choice,
  not a user requirement. Historical scorer/reward labels remain for traceability;
  this clarification does not claim a newly trained policy or silently change
  the existing runtime scorer. Future training must use a versioned definition
  that gives this route credit.
- The exact historical PPO shot has a saved 15,360 Hz ball trajectory and held
  arm endpoints. It can be rendered without rerunning physics or inventing arm
  motion. The replay is recorded simulation data, not generated imagery.

## Completed case 01 horizon-only replay — November 8, 2026

- [Frozen protocol](../sircus/mjlab/CLEARANCE_HORIZON.md): same candidate, joints,
  forecasts, refined bound, 2 mm margin, native64 dynamics and strict scorer.
  Extend only the clearance window from descending rim-band start to predicted
  table-tangent height. Three repeats each plus 12 controls: 18 requested slots.
- [Audited results](../baselines/mjlab-clearance-horizon-20261108/README.md): short
  window **0/3 captures, 3/3 robot-contact releases**; longer window **0/3 captures,
  zero releases, three abstentions**. Unexecuted contacts remain unknown. All
  12 matched controls behaved as expected; no invalid physics. All 18 requested
  slots passed the independent **44,237-check** audit.
- The window extends from 321–327 ms to 397–402 ms; the predicted clearance lower
  bound changes from +12.05 mm to −36.07 mm. Withholding this shot does not show
  improved executed trajectories or successful alternative selection.
- Frozen producer `ccfe3d9063693078ea1658359a207c42a8eda970`; draft PR #88 remains
  stacked on #87. Worker wall time 127.696331 s, pipeline 142.651 s; neither is
  isolated GPU compute time. The 4090 was released; the same 4070 PID 2016772
  remained at completion. No Isaac or service restart.
- No training, noise, alternative candidates or policy promotion. Keep the
  existing selector unchanged. Next: clarify the task objective, explore
  alternative admitted shots for case 01, then freeze a rule for a finite fresh
  held-out comparison. A [historical PPO training attempt](MJLAB_TABLE_ASSISTED_EXAMPLE_20261108.md)
  did complete board → table → bucket/rim → one-second retention; its original
  strict route/reward labels remain unchanged. Bucket retention and strict
  route success remain distinct.

## Finger-clearance and horizon diagnosis — November 8, 2026

- [Frozen protocol](../sircus/mjlab/CLEARANCE_DIAGNOSIS.md): authenticated CPU
  diagnosis found a conservative corner-bound error in prior case 07, while
  cases 04 and 39 remain below the unchanged 2 mm margin. All five remaining
  robot-contact cases occurred after the original predicted horizon; secondary
  bucket/table contact makes simple ballistic extension insufficient.
- One bounded correction is prepared: refine only the geometric lower bound,
  preserving the horizon, models, candidates, ranking, safety margin, scorer
  and native 64-substep setup. Six previously observed development layouts,
  two gates and three exact repeats plus 12 controls give 48 requested slots.
  [Completed correction](../baselines/mjlab-clearance-diagnosis-20261108/README.md):
  original clearance gate **0/18 captures**, refined bound **6/18 (3 clean)**;
  abstentions 12 versus 9; robot-contact trials 6/6 versus 3/9 releases.
  Cases 07 and 44 captured 3/3 each; case 01 still missed 3/3 beyond the old
  horizon. All 48 slots and 12 controls passed the 65,467-check native audit.
  These are six diagnosed development layouts, not fresh held-out results or
  a comparison against the active consistent-flight selector. No training or
  policy promotion occurred.
- The additive `#clearanceDiagnosis` UI requires independently bound diagnosis
  and native proofs before displaying verified results. Existing policies,
  historical evidence, 4070 workload and Isaac remain unchanged.
- Next: isolate the free-flight horizon issue, keeping later contact deflections
  distinct, then use fresh held-out layouts before considering promotion.

## Completed frozen arm-clearance comparison — November 8, 2026

- [Protocol and reproduction](../sircus/mjlab/ARM_CLEARANCE.md): compare the
  current consistent-flight selector with a post-bounce arm-clearance filter
  on 50 new paired bucket positions. Both retain the original 16 candidates,
  ranking, frozen model weights and at most four endpoint geometry checks.
  The new rule may abstain; those requests remain in the denominator.
- Five predicted ballistic paths are checked against both arms at the admitted
  endpoint joints. The fixed rule adds a 2 mm margin to ball radius, bounds the
  distance between samples, and rejects missing or overlong flight horizons.
  This is conditional predicted clearance, not a native collision guarantee.
- CPU development checks rejected the three known original robot-contact shots
  and accepted their +60 mm corrections. The unchanged top-four proposal sets
  offered one clear alternative and two abstentions. No native shots or fitting
  were added by these checks, and the rule was not tuned on the new test set.
- [Audited results](../baselines/mjlab-arm-clearance-20261108/README.md): current
  selector **20/50 captures (6 clean)**; clearance selector **16/50 (4 clean)**.
  Paired result: zero gains, four losses, 16 shared captures and 30 shared
  failures. Both had six ordinary setup rejections; the new rule additionally
  withheld 12 requests. All 50 positions remain in each denominator.
- Measured post-rebound robot-contact trials were **12/44 native releases**
  versus **5/32**. Eight former contact shots were withheld; among the 32 pairs
  that both released, no contact was removed and one was added. A lower total
  contact count therefore does not establish improved executed trajectories.
- Captures at cases 04, 07 and 39 were lost to abstention. At case 44 the rule
  selected a different shot: predicted clearance 25.77 mm, but the native shot
  missed after bucket then robot contact. Of five remaining robot-contact
  shots, only case 01 hit the robot first; others first hit bucket or table.
  The gate ends at the descending rim band and does not certify later retention.
- All **112 slots** completed; both matched control gates passed, with zero
  invalid physics. Independent verification passed **1,461,935 checks** and the
  read-only `#armClearanceStudy` UI shows counts and all 50 paired examples.
  Frozen producer commit: `84d2ffb14c9e21c5509dc38d0ec1ace2f288f2a3`.
- Do not promote this rule or start another PPO run on these results. Next:
  diagnose conservative rejections of captures and the remaining flight-model
  errors, then improve the available candidate shots on development data before
  freezing another fresh comparison. These 50 positions are now observed test
  evidence, not an unused test set. No active policy was changed or retrained.
  The worker released the 4090; the 4070 and Isaac were left untouched.


## Fifty-position failure diagnosis and completed gap correction — November 7, 2026

- [Diagnosis and exact examples](MJLAB_GAP_FLIGHT_DIAGNOSIS_20261107.md): original
  18 native misses split into first subsequent contacts with robot 9, bucket 6,
  table 3. Four entered and retained for one second but failed the complete route
  after robot contact. Original labels remain unchanged. Eight setup rejections:
  finite IK 3, fixed-base obstruction 2, endpoint self-clearance 2, action bounds 1.
- CPU reconstruction passed 702 checks against the original 1,441,075-check proof.
  New read-only `#gapFlightDiagnosis` UI contains all 50 original requests, paired
  outcomes, measured first flights and separately labeled model projections.
- [The targeted +60 mm gap test completed](../baselines/mjlab-gap-flight-correction-20261107/README.md):
  all 30 requests released, with no setup rejects or invalid physics. For cases
  12/24/48, original settings captured **0/9 complete routes**, corrected settings
  **9/9**, all with recorded rim-band/bucket contact and none clean. Each case
  changed from 0/3 to 3/3.
  All 12 matched controls behaved as expected. Independent native audit passed
  **46,625 checks**; worker wall time was 225.928 s, not isolated GPU compute time.
- All nine original diagnostic repetitions retained the ball for one second but
  failed the route. Post-board-rebound robot contact occurred in 9/9 originals
  and 0/9 corrected shots; corrected shots had no robot contact anywhere in the
  four-second trace. Contact evidence remains separate from success labels.
- These are three post-hoc development cases and exact repeats, not fresh
  held-out positions or noise trials. Increased gap changes impact energy as
  well as clearance, so the test does not isolate the causal contribution of
  clearance. Fixed64 and held endpoints only; no driven arm transit, 128-step
  convergence, policy activation or training. The 4070 and Isaac remain untouched.
- Next: preserve this diagnostic result and evaluate a frozen, clearance-aware
  selection change on fresh held-out bucket positions before more training or
  claiming that the original 50-position capture rate improved.

## Fifty-position wider-gap comparison completed — November 7, 2026

- [Audited result and provenance](../baselines/mjlab-gap-coverage-20261107/README.md):
  both frozen selectors captured **24/50 (48%)**. Clean captures: original 13,
  wider-gap 11. Each had eight setup rejects and eighteen native misses; no
  invalid physics. Paired result: 23 shared captures, 25 neither, one gain, one loss.
  Nineteen selected actions changed. There is no observed capture improvement.
- All 112 requests/96 native releases completed, including 12 passing controls.
  Independent CPU v2 audit passed 1,441,075 checks. The original v1 failure was
  a 0.0000021046-degree float32 reconstruction difference against a mixed-unit
  tolerance. The repaired audit uses normalized action units and strict saved
  action identities; original source, results, scoring and failed proof remain
  unchanged. No new native shots or fitting occurred during recovery.
- Existing `#flightResidualDiagnosis` UI now shows verified captures, clean
  captures and paired outcomes. The interactive solver was not switched;
  services, 4070 and Isaac workloads were not restarted or modified.
- Work boundary: this 112-request mjlab candidate test is separate from the
  Isaac chat's 60,000-TRAIN-request, three-seed sweep/PPO benchmark. Same overall
  task, different experiment/engine; no directly comparable winner claim.
  Keep original selection as baseline. Further changes require a new bounded
  test; these 50 results become development evidence if used for tuning.

## Flight residual diagnosis and gap-coverage comparison — November 7, 2026

- The completed 30-position comparison is now authenticated: **132 total
  requests, 116 native releases, 485,047 independent native checks passed**.
  On the same 30 TEST layouts, consistent-flight selection captured **17/30**
  (six clean), versus imitation 14/30, PPO 14/30 and old ensemble selection 13/30.
  These are one-shot paired results, not repeatability or all-table coverage.
- [Scalar results and provenance](../baselines/mjlab-flight-residual-20261107/README.md),
  [diagnosis method](../sircus/mjlab/FLIGHT_RESIDUAL_DIAGNOSIS.md). New CPU diagnosis
  passed **802 binding/reconstruction checks**, with zero new native shots or
  fitting. All 30 requests and four paired methods remain accounted for.
- The consistent selector had four setup rejects and nine native route failures.
  Eight never achieved whole-ball entry. Case20 entered and retained for one
  second but had robot contact, so the unchanged complete-route rule rejected it.
  First-flight cutoffs were bucket five, table three and robot one; contact channel
  alone does not identify a rim hit. Three rejects were finite IK failures and one
  failed strict endpoint clearance/pose admission; these do not prove impossibility.
- All sixteen model proposals had negative aperture margins at each of the nine
  failed native layouts. This suggests testing candidate coverage, not that a
  negative margin proves failure. Nominal-versus-measured rebound-origin differences
  were only about 1.03–1.71 mm for those misses; causal attribution remains open.
- [Next bounded comparison](../sircus/mjlab/GAP_COVERAGE.md): 50 newly frozen layouts,
  seed `20261801`, two methods and **112 requested slots including 12 controls**.
  Only candidate slots08/09 change from imitation-gap ±36 mm to anchor-gap ±120 mm;
  fourteen other proposals, model weights, ranking, native64 physics, scoring and
  the 16-candidate/top-four geometry budget stay fixed. No fresh results are
  published yet. The previous 30 TEST positions are now development data.
- The new worker waits for the existing 4090 lock and foreign workload; queue
  expires after 24 hours and acquired work is capped at one hour. Isaac and the
  4070 remain separate. Live queue/results belong in `#flightResidualDiagnosis`,
  including its nested gap-coverage section; no service restart, hardware action, interactive solver
  activation, 128-substep, noise or perception validation accompanies this work.

- Execution snapshot `fb86eea5fb19635194656694d3437380beb16339` passed 65
  focused CPU tests in the pinned Ubuntu environment and GitHub CI. Its bounded
  pipeline was started in a new namespace on November 7; CPU planning precedes
  the exclusive-GPU queue. This is a launch checkpoint, not a completed result.
  The existing viewer reads progress without a service restart.

## Frozen flight-consistent selector — November 6, 2026

- [Development checkpoint](../baselines/mjlab-consistent-flight-20261106/README.md):
  all 25 old native misses have a negative predicted opening margin, but so do
  13 of 24 captures. This conservative proxy ranks candidates; it is not a
  capture classifier. Case 29: old score 99.654%, new margin −71.5 mm, recorded miss.
- Corrected preparation retains an old −0.984 mm gap candidate as unsupported
  without changing it. The failed CPU-only attempt is preserved. Corrected
  source `efdacccd4fb587312f272d68307e607a406ca4ff` passed 42 focused Python tests
  locally and on pinned Ubuntu, six UI tests and GitHub CI. Fresh native results
  remain pending; no model has been activated by this work.
- [Bounded protocol](../sircus/mjlab/CONSISTENT_FLIGHT.md): same frozen models and
  sixteen candidates; rank nominal whole-sphere aperture geometry from predicted
  rebound velocities. Strict setup/path admission and actual capture scoring are
  unchanged. Ballistic plausibility does not establish capture or robot clearance.
- Prepare 30 fresh paired bucket layouts × four selectors plus 12 controls,
  132 requested slots, no fitting. All decisions freeze before outcomes.
  Original ensemble TEST is now development evidence; no counterfactual capture
  claims are drawn from changed candidate ranks.
- Execution uses only the exclusive 4090 slot when free, bounded queue/time/storage;
  no 4070/Isaac interruption, hardware action, restart or interactive solver change.
  Live status belongs in the UI, not inferred from this source checkpoint.

## Ensemble flight diagnosis — November 6, 2026

- [Recorded-flight results and exact cases](../baselines/mjlab-ensemble-flight-diagnosis-20261106/README.md),
  [method](../sircus/mjlab/ENSEMBLE_FLIGHT_DIAGNOSIS.md). CPU reconstruction of the
  original fixed64 study, with no new native requests or training. Original
  897,304-check native audit and original scores remain unchanged.
- All 25 native ensemble misses had board bounce but no whole-ball entry and
  zero retention. First subsequent free-flight contact: bucket 16, robot three,
  table six. Contact channel alone does not identify rim contact or final score.
- Scores ≥90%: 18 captures / 36 requested, with 13 flight misses and five rejects.
  Scores ≥99%: 4 / 13. Member agreement is not calibrated confidence.
- Case29: direct closest-opening head predicted 8.4 mm, measured 107.9 mm.
  Its model-velocity ballistic endpoint error was about 16.97 mm versus about
  0.07 mm with measured velocity; these conditional curves share an observed
  rebound origin and are distinct from direct distance-head predictions.
- Case36: successful BC board XY/height/tilt unchanged, gap raised exactly 36 mm;
  the ensemble missed. Original single-shot contrast, not repeatability proof.
  Capture, distance and velocity heads are independently supervised without a
  ballistic consistency constraint; this is an improvement hypothesis, not a
  proven sole cause. Nearest-TRAIN distances remain descriptive.
- The read-only `#ensembleFlightDiagnosis` panel shows six recorded examples,
  scalar/velocity errors, conditional curves and every native ensemble miss.
  Missing pre-contact rim crossing is unknown, not a fabricated clearance.
- The 4090 was occupied by another workload and was not interrupted. No changes
  to Isaac, the 4070, service processes, interactive solver, old results, scoring
  or provisional geometry. No new 128/noise/hardware work.
- Next: bounded physics-consistent flight/aperture ranking on these development
  cases before more training; changed selectors require new frozen holdouts.

## Learned physics ensemble — completed, November 6, 2026

- [Checkpoint, results and evidence](../baselines/mjlab-physics-ensemble-20261106/README.md),
  [protocol](../sircus/mjlab/PHYSICS_ENSEMBLE.md). Five forward models fit only the
  authenticated 1,024 current TRAIN requests. All 264 evaluation/control requests
  completed: 204 native releases, 60 setup rejects and zero physics-invalid
  outcomes. Full independent audit passed **897,304 checks**; positive/native-miss
  controls passed before and after. Acquired native time was 1,028.1 seconds,
  excluding CPU preparation, queueing and the later audit.
- On the same 60 TEST layouts: BC **23 captures / 3 clean / 16 rejects**;
  PPO **26 / 5 / 15**; ensemble **24 / 3 / 11**. The 113 aggregate captures include
  diagnostic validation and controls, so they are not a TEST success rate.
  This single nominal cohort does not establish a convincing method winner.
- All 252 forecasts were frozen before any evaluation; no validation tuning or
  native-flight search. Ensemble ranks 16 candidates and screens at most four
  geometry proofs; saved actors output one action. Different historical exposure
  and extra inference remain explicit, not an equal-budget or cold-start claim.
- More admission did not solve flight accuracy: all 25 ensemble native misses
  lacked whole-ball entry. Pooled across 138 native TEST flights, first-flight
  opening-distance prediction MAE was 65.2 mm. Capture scores and disagreement
  remain uncalibrated; model agreement is not a reliability guarantee.
- Training-only (126,235 checks) and planning-only (683,714 checks) summaries remain
  historical partial-proof snapshots with native verification false. The new
  completed result binds the separate full audit; those earlier proofs are not
  relabeled. No old policy, result or scene fingerprint was replaced.
- Fixed64, privileged geometry and held endpoints only. Existing 128 failures and
  provisional bucket dimensions remain. No perception, video WAM, grasping,
  driven-transit, noise or physical validation claim. Isaac, the 4070 and viewer
  were left untouched; the additive `#physicsEnsembleStudy` panel and interactive
  solver remain separate.
- Next: inspect bounded high-score misses and measured rebound/aperture errors
  before more training. These 60 TEST positions become development data if used
  for tuning; changed selectors need newly frozen held-out layouts.


## Fixed64 PPO versus imitation diagnosis — completed, November 6, 2026

- [Protocol](../sircus/mjlab/POLICY64_DIAGNOSIS.md),
  [audited results and per-case tables](../baselines/mjlab-policy64-diagnosis-20261106/README.md).
  No new policy, world model or WAM was trained. Original scores remain BC 20/50,
  PPO 18/50. Thirteen shared captures, five PPO gains and seven BC losses; five
  losses became setup rejections and two remained native flight misses.
- All 96 prespecified diagnostic slots completed in 406.8 worker-seconds:
  64 actual native releases, 32 rejected slots, zero physics-invalid releases.
  Independent CPU audit passed 164,270 checks; positive/native-miss controls passed
  before and after. Aggregate captures include controls and selected interventions,
  so they are not a new policy-evaluation score.
- Every admitted BC replay captured (21/21 across seven cases). PPO's two original
  native misses reproduced 0/6; five historical setup rejects were retained in 15
  slots without executing placeholder joints or rerunning a larger solver.
- All five setup-loss cases recovered when only PPO's board height was restored
  to BC's value, 1/1 fresh capture each under unchanged strict admission. Case 04:
  PPO lowered H by 8.098 mm; its best checked wrist/table clearance was 3.832 mm
  versus the required 5 mm (BC 7.065 mm). Raising H alone recovered a clean capture.
  Cases 13/26 also recovered with a tilt-only replacement; finite original IK failure
  does not prove the requested geometry physically unreachable.
- Flight case 07: BC 3/3, PPO 0/3; restoring only the 12.849 mm shorter BC gap captured
  once. Case 17: BC 3/3, PPO 0/3; single replacements of X, H or gap captured, while
  H+gap together missed. These one-shot interventions expose coupling/contact
  sensitivity; they do not establish robust parameter windows.
- Saved-state CPU analysis: all 50 TEST board heights moved down, gaps increased
  at 44/50; mean post-update KL was 0.3022 nats and 81.64% of sampled likelihood
  ratios ended outside [0.8,1.2] after six epochs. This is not the fraction of
  invalid actions. No explicit BC-retention penalty exists; aggressive global drift
  is a hypothesis, not a proven hyperparameter cause. TEST shaped reward also fell.
- The additive read-only `#policy64Diagnosis` panel shows all 50 paired examples,
  exact requested deltas, strict geometry evidence, recorded common-reference flights,
  separate post-hoc variants and a proposed uncertainty-aware physics ensemble/WAM
  comparison. Plot windows explicitly stop at scoring fall/capture events; complete
  raw recordings remain intact. The original policy panel/solver were preserved.
- Frozen producer commit `5d025073cbb6105f8404d1ed6f6ae95046354c32`; native package
  remains the original `f00a29a42cac69221b0896173f9be785f113a8a6`. Full mock/headless
  regression, 21 focused producer/auditor tests, five publisher checks and eight UI
  checks passed. The worker exited; the 4070 Isaac workload was left untouched and
  viewer PID 6671 was not restarted. No hardware access or new 128/noise testing.
- Next: compare feasibility-aware/BC-regularized shot selection with a compact
  uncertainty-aware forward dynamics ensemble. The former 50 TEST coordinates are
  now diagnosis/development positions; any tuned model needs newly frozen future
  holdouts excluded from all prior training, validation, test and diagnostic layouts.

## Fixed-64 success bootstrap and policy refinement — completed, November 5, 2026

- [Results, checkpoints and all nominal TEST settings](../baselines/mjlab-policy64-20261105/README.md).
  The user selected 64 substeps. All 1,284 fresh requests completed: 1,024 TRAIN,
  48 validation, 200 TEST and 12 controls; 1,103 actual native releases. Independent
  CPU audit passed 346,485 checks; fresh positive/native-miss controls passed before
  and after learning. Native study runtime was 6,625.3 s, excluding input verification
  and the separate CPU audit, not isolated GPU-kernel time.
- [Frozen protocol](../sircus/mjlab/POLICY64.md): six historical TRAIN logs supply
  6,144 requests / 4,380 native / 2,439 captures / 842 clean across 35 layouts.
  One best successful teacher per 22 contexts fits an imitation actor; all TRAIN
  outcomes fit the critic. Historical VAL/TEST labels never train it. PPO then ran
  32 updates from 1,024 new requested shots, with 615 TRAIN captures / 374 clean.
- Twelve new validation positions at prefixes 0/256/512/1024 scored 6/7/7/7 captures.
  The final prefix won the reward tie-break (2.3774 versus 2.3389 at 256). Freeze
  selection and all 200 TEST predictions before observing TEST outcomes.
- On the same 50 new test positions, requested-shot captures / clean / invalid:
  prior 19 / 2 / 2; nearest TRAIN success 18 / 3 / 19; imitation 20 / 3 / 7;
  selected PPO 18 / 4 / 11. **PPO did not improve capture rate.** It gained 5 BC
  misses but lost 7 BC captures; both captured 13. Its 32 failures were 21 native
  flight misses and 11 setup rejections (5 finite IK, 4 endpoint pose/clearance,
  2 fixed-base clearance). Diagnose paired failures before more training.
- Every rejection remains in the denominator. One nominal decision per test layout,
  no ball-flight search, retries, noise or outcome fallback at inference. Geometry-only
  source/IK/setup search is separate. This single-seed warm-start comparison has
  unequal historical exposure and does not establish all-table reliability or learned
  joint control, driven transit, grasping, perception or physical transfer.
- Historical reset/clock source differs from the current native runtime. Its guard
  stays intact; an explicit reference bridge binds old geometry/assets to the newer
  audited impact runtime. Frozen producer commit `f00a29a42cac69221b0896173f9be785f113a8a6`;
  pinned SciPy 1.15.3 and effective native options are audited. Earlier 128 failures
  and provisional bucket height interpretation remain unchanged.
- Local headless regression, 37 pinned Ubuntu CPU checks, 7 curator checks and
  9 UI checks passed. The read-only `#policy64Study` panel shows audited learning
  curves, comparison, per-position outcomes and actual saved native flights.
  The existing interactive solver has not switched to the new checkpoint. Display-only
  labels are a separate revision; frozen producer/evidence bytes were not changed.
  The worker used only the 4090, then exited; no 4070/Isaac/hardware changes or viewer restart.

## Impact-point 650 mm study — completed, November 5, 2026

- [Frozen protocol](../sircus/mjlab/IMPACT650.md): three exact XY650 source families
  receive 9 forward × 5 lateral surface-tangent offsets. Their board-arm joints,
  H/T/G and buckets are fixed; nonzero offsets solve only the dropper endpoint.
- Nominal ray point P lies on the board center plane. Gap is vertical ball-bottom
  clearance above P, preserving old releases at zero offset. Native dynamics
  must use the actual shifted release and matching dropper FK; P is not an
  observed finite-ball surface contact. Board centers remain 650 mm away, but
  nominal P-to-bucket horizontal distance varies and is displayed separately.
- Maximum 189 requests: 135 coarse64 + 24 controls + up to 30 matched64/128
  nominal verification requests. Rejects remain counted, no hidden resampling.
  At most one acquired-slot worker-hour; noise and PPO remain deferred.
- [Completed audited results](../baselines/mjlab-impact650-20261105/README.md): 189
  requests / 146 native releases in 544.3 worker-seconds. Coarse 135/92: 43 endpoint
  rejects, 5 rim-assisted captures, 0 clean. Three captures are zero-offset historical
  replays; two add -10 mm lateral offset. No nonzero-forward setting captured.
- Five selected targets each captured 3/3 at 64 and 0/3 at 128, with no whole entry
  at 128. All thirty repeats pass physics/oracle checks and ≤5 mm early paired drift,
  but no setting passes capture or clean qualification. Controls pass before/after;
  independent offset-aware audit passes 54,829 checks. All 12 clean captures in
  all-phase totals are positive controls, not 650 mm candidate successes.
- Six recent calibrated 650 mm studies total 1,501 requests / 809 native releases.
  Discovery 1,177/485 has 37 rim captures, 0 clean; these include historical repeats,
  not unique state-space coverage. This finite study is development search, not
  new RL, noise robustness, arm transit or held-out generalization.
- Source/output identities and old evidence remain separate; read-only panel
  `#impact650Study` shows verified outcomes and actual saved flights. The finite
  worker/auditor exited; existing viewer was not restarted. 4070 and Isaac were
  not changed. Stop the prescribed search here; inspect flight/rim-contact evidence
  before specifying a new experiment. More PPO/noise remain deferred.

## Remaining 30° recipes — completed verification, November 5, 2026

- [Frozen protocol](../sircus/mjlab/TILT650.md): five exact previously unverified
  30° coarse captures from the completed XY650 parent, preserving each selected
  joint pair and original geometry/calibration. Three repeats at 64 and 128,
  plus original before/after controls: maximum 54 requests, no adaptive search.
- Historical parent captures remain rim-assisted. Fresh capture qualification
  and fresh clean qualification are separate; neither is a noise-reliability or
  learned-policy result. Initial controls must pass before candidate execution.
- [Completed results](../baselines/mjlab-tilt30-650-20261105/README.md): 54/54 native,
  zero rejects/invalids; targets 15/15 rim captures at 64, 0/15 at 128 with no
  whole entry. All 12 clean captures are controls. Both control gates pass;
  independent audit passes 14,410 checks. No recipe qualifies at either tier.
- Recent calibrated 650 mm studies now total 1,312 requests / 663 native releases.
  Discovery remains 1,042 / 393, 32 rim captures, zero clean. This run adds only
  verification. Its worker elapsed was 447.3 s; timing was not isolated from
  another 4090 workload, which was not interrupted.
- Posthoc saved-trace analysis shows identical starting joints but different
  near-rim responses: 410 mm-gap shots go long at 128; 390 mm shots rebound
  short. Stop fine-grid expansion. Recommend one separately specified impact-
  point study capped at 200 requests / one worker-hour, then decide whether
  to learn from stable recipes or change geometry/contact modeling. Not started.
  Further PPO and noise remain deferred.
- Viewer restored after reboot; only the new bounded 4090 worker is authorized.
  No 4070 workload, Isaac service, hardware or historical evidence is changed.
  The finite native/audit workers have finished; draft PR #73 is stacked on #70.

## XY650 translated-plane study — November 5, 2026

- [Frozen protocol](../sircus/mjlab/XY650.md): translate board and bucket together
  by nine ΔX−10/0/+10 mm, ΔY 40/60/80 mm offsets, retaining 650 mm distance/bearing.
  243 H/T/G queries +9 translated old references +1 original =253 coarse;
  unique top 3 ×27 local refinements, top 5 ×3 repeats at 64/128 and 24 controls:
  maximum 388 requests. No noise or PPO.
- New first-flight-only ranking excludes positions after subsequent contacts.
  Center-plane aperture diagnostics remain separate from calibrated whole-ball
  entry and one-second capture. Original scorers, materials and results remain.
- CPU-only planning preflight admitted 105/253 targets, minimum 5.212 mm clearance;
  this is not evidence of a ball flight or successful recipe.
- [Completed native results](../baselines/mjlab-xy650-20261105/README.md): 388 requests,
  228 native, 160 geometry rejects. Discovery 334/174 produced 24 rim captures, 0 clean.
  Five selected settings each 3/3 rim at 64 and 0/3 at 128; 0 qualified. All 12 clean
  all-phase captures belong to positive controls. Both control gates passed.
- Original checker failed on a pre-calibration ancestor schema. Its failed proof,
  logs, native source and observations remain unchanged. Separately frozen
  audit v2 passed 80,138 checks against original evidence, with explicit calibration
  ancestry intact; publisher/exporter use the new external proof. No physics rerun.
- Unselected 30° coarse captures have UNKNOWN 128 outcomes. Frozen refinement
  chose distinct XY targets in the same H/T/G family; future studies should
  consider family diversity, impact offsets or tool yaw before more PPO.
- New scene/query/execution identities, complete requested denominators and an
  independent full audit gate the additive `xy650_study` UI. 4070/Isaac remain
  separate; existing services are not restarted. Raw observations stay outside Git.
- Draft [PR #70](https://github.com/kadiru/sircus/pull/70) is stacked on PR #69.
  The finite 4090 workers have exited; no policy or noise study was started.

## Clean650 development sweep — November 4, 2026

- [Frozen bounded protocol](../sircus/mjlab/CLEAN650.md):360 H/tilt/gap grid
  combinations plus the original650 reference, up to81 local refinements,
  top-five matched64/128 checks and initial/final controls; maximum496 requests.
- Prior650 shot is timestep-sensitive and remains separate. Clean capture is
  required before promotion; rim assistance, rejects, misses and pending values
  are distinct. Noise testing and further PPO are deferred.
- Completed496 requests:155 native releases and341 endpoint rejections.
  Discovery442 requests admitted101 native releases:0clean and2rim captures,
  both the original380mm/28°/386.8mm target. Verification reproduced3/3rim at64
  and0/3capture at128; no qualified setting. All12clean captures in all-phase
  totals belong to positive controls, not650mm discovery. Initial/final controls
  passed; the independent audit passed64,075 checks.
- [Curated v1 results](../baselines/mjlab-clean650-20261104/README.md) preserve
  every requested H/tilt/gap and rejection without raw joints or observations.
- A separately versioned fine follow-up targets H375/380/385mm,
  tilt27/27.5/28/28.5/29° and gap380/386.8/390/395/400mm, plus the exact reference.
  Fine refinement uses ±5mm/±0.25°/±2.5mm. Maximum211 requests including matched
  controls and verification. Completed211/106native;105 rejects,0clean650
  captures and0qualified settings. All12clean captures again belong to controls.
  Reference verification stayed3/3rim at64,0/3at128. Both control sets passed;
  auditv2 passed33,824 checks. [Fine results](../baselines/mjlab-clean650-fine-20261104/README.md).
- [CPU model reach diagnosis](../baselines/mjlab-clean650-20261104/reach-diagnostic.json):
  atH380/T28 the required shoulder-to-wrist distance841.528mm is0.109mm below
  the URDF triangle-inequality upper bound841.637mm. T27.5 requires842.986mm.
  Extra iterations/continuation do not repair the necessary-bound violation.
  Raising board height admitted8/9 endpoint-only probes at release center804.3mm.
- A separate raised-v3 profile therefore tests H400/420/440mm ×T26/27/28° ×
  G326.8/346.8/366.8mm plus reference;27+1 coarse requests, ±5mm/±0.5°/±2.5mm
  refinement and matched checks; maximum163. Same fixed XY650 scene and scorer.
  Completed163/120native,43rejects,0clean650 and0qualified; before/after controls
  passed and auditv3 passed31,902 checks. [Raised results and comparison](../baselines/mjlab-clean650-raised-20261104/README.md).
- Across the three profiles:870 total requests/381native, including72controls
  and90verification requests. Discovery only:708 requests,219native releases,
  190distinct native targets,8rim captures of the one original setting,0clean.
  All36all-phase clean captures belong to positive controls. No new recipe
  qualified at64/128; these finite results do not prove all650mm shots impossible.
- Further PPO remains deferred. Next bounded work should reconsider the fixed
  bounce-plane position/action constraints and rank miss trajectories before
  external/table contact, then prespecify a new scene or search. No guard removal,
  material refit, noise study, policy activation or lower capture threshold.
- Strict5mm endpoints and calibrated dense scoring remain unchanged; held endpoints
  do not validate arm transit or actuator motion. Finite IK failures do not prove
  the requested poses physically unreachable, nor does this sweep prove650mm impossible.
- Additive read-only `clean650_study` UI key; existing services, 4070 and Isaac
  configuration are preserved.
- Draft [PR69](https://github.com/kadiru/sircus/pull/69) remains stacked on PR65;
  all source and curated results are separate from raw runtime observations.

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
  after rim contact. Independent auditv2 **passed5,224 checks**; source, package,
  asset, calibration, raw-cache, initial-joint, native-option and full scoring
  bindings passed. The corrected checker runs separately from the preserved
  native producer; no flight was retried for an audit correction.
- Additive read-only interface key `calibrated650_replay`; existing viewer,
  4070 workload and Isaac configuration remain separate and unchanged.
- [Verified results and exact CSV](../baselines/mjlab-calibrated650-20261104/README.md).
  Next: seek central entry with more rim clearance before another gap window.

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
