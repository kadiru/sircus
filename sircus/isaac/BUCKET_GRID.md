# Spatial bucket catalogue and the next learning boundary

This task extends the fixed release-plane search to a 3 × 3 development grid.
It does not train or promote a policy. It maps feasible target locations before
spending a larger RL budget, and rejects captures that depend on the table or arms.
All measurements below are synthetic, with the existing provisional material,
ball, tool-offset and bucket-height assumptions.

## Frozen experiment

- Board/release center XY: (670, −100) mm.
- Bucket X: 470, 670, 870 mm; Y: −400, −550, −700 mm. All nine opening footprints
  fit the modeled table. Horizontal distances are 300–632.5 mm.
- Board heading faces the bucket geometrically; it is not learned. The board and
  release remain vertically aligned. Search controls board-center height, tilt
  from horizontal, and vertical gap from board center to released ball bottom.
- Each cell starts with 42 candidates: six earlier anchors and a 3 × 3 × 4 grid
  over H=350/450/550 mm, gap=80/160/240 mm and tilt=12/18/24/30 degrees.
- Refine the two best valid candidates by gap ±10 mm and tilt ±1.5 degrees.
  Up to eight additional candidates. Ranking is route capture, entry tier,
  absence of external contact, historical reward, then deterministic ID.
- Screen up to two captured settings, prioritizing single-bounce capture then
  entry tier/reward/ID. Ten independent variations each; freeze the best by
  board-only captures, single-bounce count, clean count, then ID.
- Confirm with 30 new independent seeds, no retuning: H and gap ±2 mm, tilt ±0.5°.
  Invalid endpoint/settling attempts remain in denominators and are separate
  from released misses. No selected setting means this finite search found none,
  not that the target is impossible.
- Maximum 902 parameter attempts (including two controls); potentially fewer
  when no setting qualifies. Five separate contact probes precede these attempts.
  A two-hour timeout and 3 GiB free-disk floor bound the run.

Fresh confirmation measures local repeatability at *searched* positions. These
are not held-out layouts, model generalization, calibrated physical noise, or
whole-table coverage. Thirty successes do not establish 100% reliability.

## Additive route scoring

Historical whole-ball, board-bounce and continuous one-second capture labels and
entry tiers remain unchanged. New `sircus-board-route-v1` additionally requires
one or more board contact episodes and no detected table/floor/robot/unattributed
contact between release and the capture/timeout terminal state. Bucket contact,
including a rim-assisted capture, is permitted.

- `route_success`: retained capture through the board and bucket only.
- `single_bounce_success`: the above with exactly one board contact episode.
- `route_quality`: 2 clean entry; 1 assisted or marginal capture; 0 failed route.
  Clean needs at least 1 mm mouth clearance, no pre-entry/near-rim contact and no
  subsequent partial exit. It does not imply exactly one board contact.

At 240 Hz, force >0.01 N detects a contact; 50 ms without board contact separates
board episodes. Force aggregation can conceal simultaneous cancellation, and
subthreshold contact is not certified absent. This is normal-force evidence,
not a full impulse/contact-position audit.

The earlier 400 mm control (H400/gap130/tilt21) captures with **two** board contact
episodes, no external contact. The earlier 600 mm control (H450/gap130/tilt13.5)
captures after measured **table** contact. Both remain successful under their
historical definition. Smoke checks first exposed the difference; the explicit
board-only/single-bounce distinction was frozen before the spatial search. We do
not relabel either shot as a verified single-bounce shot.

Table and floor are stationary kinematic rigid bodies only in this opt-in mode,
because static colliders did not populate the filtered force matrix in the first
smoke. Geometry is unchanged, and bucket-floor/rim, table, floor and robot probes
verify sensor attribution before running controls. Earlier launch modes retain
their original support configuration and fingerprints. Failed diagnostic smoke
artifacts remain local and are not presented as completed studies.

## Runtime and reproduction

Use the pinned Isaac Lab 2.3.2 / mounted Isaac Sim 5.1 installation documented in
[README.md](README.md). One scene process owns the 4090 workspace lock at a time.
Never run a second experiment against that same scene concurrently.

On the configured Ubuntu host, the executed source is the immutable
`~/sircus-isaac/studies/bucket-grid-v3` snapshot. Its path differs from the shared
Git checkout; pulling source does not deploy a runtime. Launch from that snapshot:

```bash
docker exec -u 0 -d \
  -e SIRCUS_ISAAC_SOURCE=/workspace/studies/bucket-grid-v3 \
  sircus_isaac_4090 sh -c \
  'timeout 7200 flock -n /workspace/scene.lock /workspace/studies/bucket-grid-v3/python.sh /workspace/studies/bucket-grid-v3/launch_scene.py --headless --enable_cameras --device cuda:0 --bucket-grid > /workspace/logs/bucket-grid-v3.log 2>&1'
```

Add `--bucket-grid-smoke` for probes and two controls only. Do not overwrite an
existing log or snapshot when reproducing; use distinct names. The existing
container's mounted outputs require `-u 0`; this changes no host permissions.
No hardware, Viam credentials, audio or 4070 changes are involved.

The read-only preview displays `/bucket-grid.json` at `#bucket-grid`. It keeps
pending, no-setting, and measured-zero outcomes distinct. A validated CSV appears
only after completed results pass independent export checks. To export locally:

```bash
python sircus/isaac/bucket_grid_report.py \
  PATH_TO_RUN/summary.json PATH_TO_COMPACT_RESULTS \
  --executed-source PATH_TO_EXECUTED_SNAPSHOT
```

The validator checks the frozen contract, candidate/seed/selection rules, all
counts, settling, route constraints, raw/compact trial equality, artifact hashes
and executed-source hashes. Only compact parameters/scores and hashes belong in
Git. Raw trajectories, joint traces, logs, generated assets and captures stay on
Ubuntu. Failed or incomplete runs cannot be exported as verified baselines.

## How much RL compute to expect

An iteration is usually a **policy update**, not one attempt. With E parallel
robots and H decision steps per rollout, one update collects E×H transitions;
PPO then performs several gradient epochs on that batch. Physics substeps and
complete episodes are different units again. Our current shot policy makes one
high-level action per whole throw, unlike a locomotion policy acting at 50 Hz.

Published examples (successful training runs, not guaranteed time to first success):

| Robot/task | Policy updates | Environment transitions | Reported hardware/time |
|---|---:|---:|---|
| ANYmal, uneven-terrain walking | 1,500 | 147.5 million, derived | RTX A6000, under 20 min |
| Franka, cube stacking | ≈1,500, derived | 786 million | A100, under 25 min |
| Shadow Hand, cube rotation, standard observations | ≈5,000, derived | 655 million | A100, under 35 min |

ANYmal used 4,096 environments and a 98,304-transition batch.
[Learning to Walk in Minutes, §4.2](https://arxiv.org/html/2109.11978v3).
Franka used 16,384 environments × 32 steps; Shadow Hand used 16,384 × 8.
Approximate update counts divide rounded published steps by those batch sizes.
[Isaac Gym, §6.3–6.4 and Table 17](https://arxiv.org/html/2108.10470v2).
These are optimized parallel benchmarks, not timing predictions for our scene;
the standard hand example also lacks domain randomization.

Our previous 2,081-attempt distance catalogue took 72.4 minutes: about **2.1 wall
seconds per attempted throw**, including endpoint rejections. At that serial rate,
10,000 attempts cost about 5.8 hours and 100,000 about 58 hours, before additional
training/evaluation costs. This estimate is not a measured vectorized RL budget.
Camera rendering, IK, arm settling and GPU/CPU transfers currently cost much more
than a small policy network update. Increasing GPU memory does not automatically
parallelize this Python shot loop.

## Next actual learning experiment

1. Review this catalogue's routes and unreachable cells. Validate timestep/contact
   behavior and address repeat board hits before treating results as physical truth.
2. Profile a vectorized 32/64-environment shot implementation against the serial
   reference, with equivalent contacts and success checks, before a long run.
3. Train a centralized scene-conditioned policy: input relative bucket XY and
   geometry/state; output H, gap and tilt. Keep heading analytic initially.
   Known successful development shots can initialize a supervised policy before
   PPO fine-tuning; clearly report both stages and compare with lookup/fixed baselines.
4. Use separated spatial train/validation/test cells and within-cell perturbations;
   reserve unseen cells *and* fresh continuous positions for final evaluation.
   Keep near/mid/far and left/right sampling balanced. Existing searched cells are
   development data, not a new held-out test set.
5. Validate a versioned reward that rejects external-contact shortcuts and ranks
   retained clean capture above assisted capture. Report capture, route, single-
   bounce, invalid rate and confidence intervals separately. Final inference uses
   a frozen checkpoint, with no per-scene search or final-test retuning.
6. Run multiple seeds and stop based on measured held-out performance, not an
   arbitrary scene count. Add point-cloud observations only after the privileged
   baseline works; grasping, collision-free transit and physical transfer remain
   separate behaviors and validation tasks.
