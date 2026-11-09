# Table-assisted imitation and short PPO comparison

This experiment follows the separately audited [v3 rescoring](SCORING_TABLE_ASSISTED.md).
It accepts board → bucket and board → table → bucket after one continuous second
of whole-ball retention. Robot, floor and unattributed contacts remain forbidden.
Historical experiments, labels, native source and checkpoints are not overwritten.

## Review of the parallel Isaac feasibility work

The `sircus-isaacsim-isaaclab` conversation and `fcdefe6` / PR #91 were reviewed
before preparing this experiment. That change extracts a coupled-arm preflight
API; it reproduced 420/420 saved admissions and ten fresh CPU checks. It does
not demonstrate an IK or capture improvement. Seed 20261122 still captured
28/70 TEST requests: 17 preflight declines, 53 releases, 25 released failures.
The 52.8% conditional-on-release rate must not replace 40% overall performance.

mjlab already separates geometry resolution, exact-action IK/setup admission
and native release. Its deterministic IK branches, native collision geometry,
full-pose targets and sampled paths differ from Isaac's normal-only board IK,
capsule proxy and driven execution. We do not cherry-pick the Isaac controller.
Existing `calibrated_admission`, `calibrated_adapter`, native physics and old
`policy64_*` contracts remain unchanged.

A future policy over diverse prequalified candidates needs its own action
distribution and likelihood contract. Silently projecting PPO outputs or
resampling rejected actions would invalidate the current log probabilities.
This experiment does neither. Unresolved IK is not proof a bucket is impossible.
It also does not activate the older ballistic clearance-window selector: that
model does not describe the newly permitted table/bucket deflections.

## Source data and model

Use only 6,144 original calibrated-benchmark TRAIN requests and 1,024 original
policy64 TRAIN requests, joined to the audited v3 derivative using source study,
record identity, original row hash and native archive identity. Historical
validation/TEST labels are excluded. The 115 newly credited TRAIN captures
contribute to critic reward fitting. They are not all claimed as actor teachers.

Actor imitation preserves one teacher per TRAIN context, ranked by clean entry,
v3 reward and stable ID. This avoids averaging incompatible successful recipes.
The report discloses actual teacher changes and how many selected teachers used
table contact. New captures need not outrank a clean existing direct example.

The residual policy retains nine geometry features and five outputs: board
X/Y/height, tilt and release gap. Yaw faces the bucket geometrically. The actor
and critic each retain two 64-unit hidden layers. These are held-endpoint shots
with initialized release, not learned grasping, arm transit or perception.
The new model is initialized reproducibly and fitted to the combined TRAIN data;
PPO starts from that fitted imitation checkpoint.

## Bounded native protocol

- Fixed **64 substeps**; unchanged physics, geometry, home and calibrated
  containment. No noise or 128-substep requirement in this experiment.
- **512** new TRAIN requests, **16** PPO updates, batch32. Rejections consume
  their slots. The previous 1,024-attempt run took about 110 minutes; this shorter
  pilot leaves room for the extra CPU scoring under a two-hour acquired-slot cap.
- **12** validation positions, evaluated at 0/256/512 TRAIN requests: **36** shots.
  Selection uses captures, then mean reward, then earliest checkpoint.
- **50** fresh TEST positions, one frozen prediction each for old imitation,
  old PPO, new imitation and selected new PPO: **200** shots. Predictions are
  saved before any TEST outcomes; no inference-time flight search or retries.
- Three repeats each of direct-capture, table-assisted-capture and miss controls,
  before and after: **18** requests. Controls must pass before optimization.
- **766** maximum new requested shots, **7,200 seconds** maximum acquired-slot
  time, 8 GiB storage reserve. A partial run is retained and labeled incomplete.

TRAIN uses the original35 locations. Validation/TEST are sampled with fixed
seeds within X100–1240 mm, Y−650–250 mm, at least5 mm from one another and all
bucket coordinates discovered in the inventoried historical metadata and
published baselines. Sampling does not inspect IK or success labels. This is
novelty against that explicit inventory, not an unpublished-history guarantee.

All four policies are measured under the same new v3 success criterion. Old
checkpoints are copied byte-exact. New policies have additional historical
training exposure; this is not an equal-budget algorithm contest or a pure
reward-only causal ablation. The requested-shot denominator includes setup
rejections, and reports separate direct/table captures and released misses.

## Execution and verification

`table_training_pipeline` starts one finite worker with the shared GPU lock and
authorized RTX4090 UUID, then a CPU-only independent auditor. It never signals
foreign jobs. The RTX4070, Isaac services, hardware, Viam and existing viewer
process are untouched. The environment factory alone does not acquire a lease.

`table_training_contract` validates immutable parent evidence and splits;
`table_training_native` opts into v3 while preserving exact-action admission;
`table_training_verify` checks source/data/model bindings, optimizer membership,
frozen predictions and dense native scoring. The read-only `#tableTraining`
panel labels all interim counts provisional until the audit is bound to the
completed report. A separate finite CPU publisher embeds the small snapshot in
the existing HTML; the browser refreshes that snapshot every30 seconds. This
avoids repeatedly downloading the large historical study report. The frozen
training source is not edited, and no solver or viewer process is restarted.

Generated observations, weights, runtime configs and logs remain outside Git.
Only reviewed source, documentation and compact verified results are published.

## Launch checkpoint — November 8

Producer `b11e8b9772268d79a14a5bcb0b8d9761789da379` was launched in a new evidence
namespace after49 Ubuntu CPU/Torch tests passed and579,197 independent checks
of the actual historical input join passed. The new actor uses27 teachers,
including five table-assisted examples, versus22 original teachers. Sixteen
teacher identities changed; all115 newly credited TRAIN rewards enter critic
fitting. The coordinate inventory contains1,037 distinct historical positions.

The initial nine native controls passed:3/3 direct captures,3/3 table-assisted
captures,3/3 expected misses. Imitation fitting then began. This is a launch
checkpoint, not a completed PPO or held-out result. The existing4070 process and
viewer retained their original process/start identities. The separate publication
source can advance without changing the frozen producer or its final auditor.


## Completed comparison — November 8

All 766 requests completed and the independent audit passed 823,442 checks.
Both control gates passed all nine prescribed outcomes. The selected PPO model
is the 512-request checkpoint; validation capture counts at 0/256/512 were
5/12, 5/12 and 6/12. Worker elapsed time was 76.4 minutes.

| Frozen model | Capture / all requests | Direct | With table contact | Setup rejected | Released misses |
|---|---:|---:|---:|---:|---:|
| Old imitation | 9/50 (18%) | 9 | 0 | 6 | 35 |
| Old PPO | 8/50 (16%) | 8 | 0 | 5 | 37 |
| New imitation | 14/50 (28%) | 8 | 6 | 4 | 32 |
| New PPO | 15/50 (30%) | 9 | 6 | 4 | 31 |

The new PPO gained 12 percentage points over old imitation and 14 over old PPO
on this matched split. Its one-capture gain over new imitation is inconclusive.
Both new policies made six captures with preterminal table contact; this label
records contact presence, not an independent causal reconstruction of the route.
All TEST shots had valid physics. Four invalid-physics requests elsewhere in
the study remain recorded; they were not removed to improve the denominator.

See the [compact result baseline](../../baselines/mjlab-table-assisted-training-20261108/README.md)
for hashes, matched outcomes, paired uncertainty and limitations. The UI shows
all 50 positions under “Inspect each bucket position.” Old checkpoints were
rerun on this new split; do not compare these counts with historical TEST counts
from a different set of positions. The interactive selector remains unchanged.

Before more training, inspect the new PPO's 31 released misses and four setup
rejections. Repeat evaluation on fresh layouts/seeds before claiming a robust
gain. The successful table-contact cases cluster in one far-table region; this
study does not establish coverage of arbitrary table positions.
