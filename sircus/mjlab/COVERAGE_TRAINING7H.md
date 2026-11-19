# Seven-hour spatial-coverage learning study

This is a new experiment, not an extension that changes a historical pilot's
contract. It trains a bucket-conditioned, five-parameter shot selector with
privileged simulator geometry. The nine inputs describe the bucket and a
geometry-resolved starting pose. The outputs adjust board X/Y, board-center
height, tilt and release gap. Board heading faces the bucket by the existing
geometric convention. Grasping, continuous joint control and perception are not
trained here.

The motivation is limited successful demonstrations: the last imitation actor
used 27 teachers, and the corresponding PPO comparison captured15/50 versus
14/50 for imitation. More optimizer steps on the same small coverage are not an
established solution. This experiment first collects new TRAIN outcomes and
then compares refreshed imitation with two short PPO seeds.

## Startup correction and version boundary

The first v1 launch stopped after nine startup controls, before any search or
training. The three direct controls captured, but all three table-assisted
controls missed. Adding the Case00 geometry changed the selected IK wrist branch
for that control; its requested board/drop values were unchanged. The three
negative controls still missed as expected. All failed-attempt evidence remains.

Version v2 replays the three controls from their authenticated original17-source
geometry/connection proofs and exact recorded joint branches. It does not search
for a replacement positive control or relax scoring. Controls use a separate
native evidence namespace, sharing the same pinned physics environment; search
and all four policies continue to use the expanded19 bank. This is a corrected
control protocol, not a relabeling of the first launch. The retry retains the
original November9,07:59PST absolute deadline. The first nine requests are
reported separately from the new experiment's maximum2,757.

## Frozen data and layout contract

- The unchanged v3 scoring accepts board → bucket and board → table → bucket,
  including rim assistance, after whole-ball entry and one continuous second
  retained. Robot, floor and unattributed contacts remain excluded. Physics,
  calibrated containment, exact-action IK and setup guards stay unchanged at64
  substeps. Noise and128-substep qualification remain deferred.
- The geometric prior explicitly expands from17 to19 poses by adding the two
  audited Case00 captures at26° and32°. These inspected former-TEST poses are
  now declared DEVELOPMENT prior information. Original studies and labels stay
  intact. Their120mm board-Y shift is represented by a new starting pose, not
  clipped into the old±40mm residual support.
- All four compared models use this same expanded geometry bank. The current
  imitation/PPO weights are copied unchanged, but their new results are not
  their old historical evaluation, which used a smaller bank.
- Freeze coordinates before any new IK or flight outcomes. Divide the table
  rectangle X100–1240mm, Y−650–250mm into7columns×5rows. Draw two fresh TRAIN,
  one validation and two TEST positions in every cell using recorded seeds.
  Add the35 original TRAIN positions:105 TRAIN,35 validation,70 TEST total.
  New coordinates remain at least5mm apart from one another and all bucket
  coordinates in the frozen historical metadata inventory. This is explicit
  inventory-based novelty, not a claim about unavailable history or distant
  out-of-distribution extrapolation.

## Search and training

The maximum search contains12 proposals per TRAIN layout,1,260 requests total,
ordered by round and the frozen105-layout order. The first ten are prescribed
axis probes around the admitted geometry anchor: baseline, tilt±2°, gap±10mm,
board height±25mm, board X±20mm and board Y+40mm. The last two are seeded joint
refinements around the best *earlier same-context TRAIN* outcome. The requested
parameters are never projected, resampled after refusal or silently replaced.
Both misses and unflown setup declines consume slots and remain evidence.

The new actor is fitted from one best captured fresh search example per context
(clean entry, then reward, then stable ID); the critic uses all fresh TRAIN search
rewards, including failures. Old historical teachers are not silently remapped or
clipped relative to the new geometry anchors. This intentionally tests a new
fresh-data curriculum, not equal historical-data exposure across models.

Both PPO seeds begin with the same fitted imitation model. They run interleaved
32-request updates, up to512 requests/16updates per seed. The policy stays within
the existing residual support:±40mm board X/Y,±50mm board height,±8° tilt and
±120mm gap. PPO retains its Gaussian latent-action likelihood and tanh action
mapping; all rejected sampled actions still enter the update with their reward.

Evaluate the shared imitation checkpoint and each seed at256/512 requests on
35 validation layouts. Select by validation captures, mean reward, earliest
checkpoint and seed. Step-zero imitation is eligible: PPO is not forced to win.
Freeze all four model prediction packets before TEST outcomes. TEST makes one
deterministic decision per position with no flight search, optimization or
outcome-driven retry. The geometry resolver is still an explicit inference cost.

## Budgets and completion

| Stage | Maximum requested shots |
| --- | ---: |
| TRAIN search | 1,260 |
| PPO, two seeds | 1,024 |
| Validation, common BC plus four PPO checkpoints | 175 |
| TEST, four methods×70 layouts | 280 |
| Three direct/table/miss controls before and after | 18 |
| Total | **2,757** |

The pipeline has an absolute **seven-hour** deadline from launch, including queue
and independent audit. The native worker is capped at5.5hours after acquiring the
shared4090 lease. Search stops at a completed batch once its two-hour cap is
reached. PPO stops at a completed seed-pair boundary at4.5hours, leaving time for
full frozen TEST and controls. Only completed validation checkpoints qualify.
Actual prefixes, stop reasons and counts are reported. A timeout, failed control
or failed audit never becomes a completed result; all partial evidence remains.
An8GiB free-space reserve is enforced; nothing is deleted to make room.

An independent CPU auditor checks provenance, spatial splits, training membership,
search dependencies, exact requested actions, checkpoint/optimizer replay, frozen
TEST predictions and raw native64 trajectories using the independent v3 oracle.
The live `#coverageTraining7h` UI labels results provisional until that audit
passes. Setup declines remain in capture denominators, and each table cell shows
coverage/outcomes rather than implying the entire table is solvable.

Only the4090 executes native physics. CPU work hides both GPUs. The4070 workload,
Isaac setup, hardware, Viam configuration and current viewer service are untouched.
Publication is additive and requires no server restart. Checkpoints, raw traces,
configs and logs remain outside Git; code, protocol and compact verified results
are shared on the task branch. No policy is automatically promoted into the
interactive solver.
