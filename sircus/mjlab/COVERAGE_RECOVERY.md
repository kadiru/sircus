# Bounded coverage recovery study

This follow-up uses the completed seven-hour-budget coverage experiment as
**development evidence**. Its former TEST outcomes remain unchanged in their
historical report; those coordinates are excluded from the new TEST split.
The purpose is to find additional executable, repeatable demonstrations and
measure whether imitation improves on fresh bucket positions. No additional
PPO optimization or automatic interactive-policy promotion is authorized here.

## What the parent study showed

New imitation and selected PPO each captured 28/70 on identical unseen layouts.
Selected PPO declined 22 setups and missed 20 released shots. The declines were
13 finite IK searches without a solution, four endpoint clearance/pose failures,
three fixed-base clearance failures, and two out-of-bounds requests. Finite IK
failure does not prove physical impossibility. All 20 released misses bounced
the board; 18 never achieved whole-ball entry, and two retained the ball for a
second but had forbidden robot contacts. Table contact remains allowed.

The strongest previous PPO weights scored 31/70 under the same shared geometric
prior. That model is included in this follow-up so comparison is not limited to
the weaker new PPO checkpoint. Historical percentages are not directly paired
with the new split.

## Frozen limits and sequence

- Same scene, assets, action bounds, admission guards, v3 scoring, and **64
  substeps**. No noise sweep, 128-substep qualification or Isaac changes.
- The same 19-source geometry prior is shared by search and all four policies.
  Positive direct, positive table-assisted and negative controls retain their
  original authenticated joint branches under the historical 17-source prior.
- Search 39 former-TEST failures; three fixed-base blockers are reported without
  pretending that changing board height or tilt can repair fixed base geometry.
- At most 16 prescribed residual-space proposals per case, scheduled round by
  round across all cases: **624 search requests**. The complete proposal plan
  is fixed before outcomes. No clipping, action substitution or relaxed guards.
- Select the best captured proposal per case, then make two fresh nominal
  repeats: at most **78 requests**. Both must capture to admit a new teacher.
- Fine-tune the parent imitation actor for 800 deterministic full-batch steps
  using its 68 original teachers plus confirmed new demonstrations. Critic
  weights stay unchanged. If no new teacher qualifies, preserve the parent
  checkpoint byte for byte and explicitly report that no recovery fit occurred.
- Freeze all models and predictions before TEST: recovery imitation, parent
  imitation, parent selected PPO and the strongest previous PPO. Each gets one
  request on each of **70 new bucket positions**, for **280 TEST requests**.
- Nine controls before and nine after: **18 control requests**.
- Maximum **1,000 requested shots**, including setup rejections. Search stops
  after 3,600 acquired-slot seconds; repeats stop after 4,500. The worker has a
  7,200-second absolute acquired-slot cap, with a 10,800-second overall pipeline
  cap for queueing and independent CPU verification. No complete evaluation is
  claimed unless all 280 TEST requests and both control gates complete.
- Use only the RTX 4090, its shared lease and existing pinned mjlab environment.
  Preserve at least 8 GiB free disk space. Never signal foreign GPU processes.
  The 4070 fountain workload and existing viewer/Isaac processes stay untouched.

For each case, proposal 0 repeats the failed policy request; proposal 1 replays
an already observed alternative-policy capture if one exists, otherwise the
nominal geometry anchor. These are development information, not held-out wins.
For setup declines, the next probes combine board-height residuals −0.8/0/+0.8
with tilt residuals −0.75/0/+0.75, plus gap and XY probes. For released misses,
the probes cover six gap residuals (−0.9/−0.6/−0.3/+0.3/+0.6/+0.9), four tilt
residuals (−0.75/−0.375/+0.375/+0.75), two heights, an XY reset and one coupled
probe. Residual scales are 40 mm X, 40 mm Y, 50 mm height, 8° tilt and 120 mm gap.
A residual is relative to that layout's fixed geometry anchor, not necessarily
a change of that amount from the failed shot. The exact literal plan is saved.

The TEST seed is `2026110923`. Input hashes, model identities, request plans
and phase boundaries are recorded in the study contract and evidence.
Every requested outcome counts in the denominator; a setup rejection is not a
flight, and fewer robot contacts caused by fewer releases are not an improvement
in flown trajectories.

## Evaluation and limits

New TEST coordinates are stratified across the same 7×5 table grid, two points
per cell, and separated from all coordinates discovered in the frozen historical
inventory by at least 5 mm. Selection does not depend on IK success or capture.
This is novelty against inventoried records, not proof no unknown experiment
has ever used a similar coordinate. No TEST result updates a model or selects a
new shot. Any inspected TEST position becomes development data for future work.

Report captures / 70, native releases, setup declines, flight misses, contact
routes and paired gains/losses against every baseline. Repeating successful
development proposals twice is limited nominal repeatability evidence, not noise
robustness. The new model has extra demonstrations and therefore this is not an
equal-training-budget algorithm comparison.

The independent audit reconstructs requests, training and frozen predictions;
checks exact setup proofs and native dense scoring; authenticates parent evidence;
and verifies accounting and controls. Raw traces, checkpoints, runtime configs
and logs remain outside Git. Compact summaries and source code are published.
The UI shows pending results as pending until the matching audit passes.

This remains privileged-geometry, held-endpoint shot selection. It does not train
perception, grasping, moving-arm coordination, transit dynamics or physical robots.
