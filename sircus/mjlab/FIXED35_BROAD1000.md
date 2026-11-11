# Broad fixed-grid search: 1,000 actual drops

This task searches for successful recipes at three unresolved bucket positions.
It does not fit or activate a policy. Existing evidence remains unchanged:
30/35 grid locations have confirmed recipes, while the current frozen policy
scores 24/35. These are different measurements.

The previous four study cohorts contain 100 native drops at these three cells,
all misses, alongside 737 setup declines. The next experiment targets **1,000
new search releases**, rather than 1,000 requested or rejected configurations.
Finite historical failures do not prove these bucket positions impossible.

| Fixed bucket cell | Bucket X/Y (mm) | New search releases |
|---|---|---:|
| 03-04 | 860 / 25 | 334 |
| 03-05 | 1050 / 25 | 333 |
| 04-06 | 1240 / 250 | 333 |

The two bucket/base-overlap cells remain excluded. Bucket positions, arm bases,
ball/board/bucket geometry, native64 physics and v3 scoring stay fixed.

## Proposal and admission

Freeze the proposal generator, seed, ordering, bounds and branch-selection rule
before outcomes. Generate at most 40,000 raw numeric proposals per cell; retain
at most 4,000 per cell that pass the conservative geometric reach filter.
Passing that filter is necessary, but does not establish IK or path feasibility.
Do not silently clip out-of-bounds values or relocate a bucket.

Vary board X/Y/height, tilt and drop gap jointly. Keep bucket-facing gripper
orientation. Examine distinct bouncer shoulder/elbow configurations and wrist
alternatives before spending additional branches on dropper wrist alternatives.
Allow at most two admitted bouncer branches per numeric setting. Each round
prepares up to four settings per active cell, preserving the frozen ordering.
Every release requires the unchanged full-pose IK, endpoint-clearance and
sampled setup-path checks. Rejected candidates remain in a separate ledger.

Prepare candidate proofs on CPU. Freeze the exact five shot values, both joint
vectors and the proof identity before submitting a native batch of at most 32.
Visit cells in fixed round-robin order without exceeding their remaining quota.
Do not stop broad exploration after the first successful capture.

## Counts and confirmation

Only fresh `phase=search`, `native_release=true` records with authenticated native
archives count toward the 1,000 target. Setup refusals, cached observations,
control trials, preservation trials and confirmation repeats do not count.
Do not fill a quota by replaying duplicate five-value/joint-pair recipes.
Numerically invalid releases still count as attempts, remain explicitly flagged,
and never count as successful captures.

The first three distinct successful numeric settings per cell receive three fresh repeats
of their exact admitted joint branch. A recipe is confirmed only after 3/3 fresh
captures. Preserve unsuccessful confirmations and continue the broad search.
This allows at most 27 confirmation releases in addition to the search quota.

Replay all 30 existing successful recipes unchanged. Run nine original controls
before the search and nine afterward: direct captures, table-assisted captures
and expected misses, each repeated three times. A passing control gate means
the expected outcomes occurred; it does not mean all controls captured.

The maximum flight count is **1,075**: 1,000 search + 27 confirmation +
30 preservation + 18 controls. Preparation requests and refusals have separate
counts and bounds. Report attempted search releases, valid releases, captures,
confirmation outcomes and retained prior recipes separately for each cell.

## Scoring and scope

Capture requires a board bounce, whole-ball bucket entry and one continuous
second inside. Table and rim assistance count. Robot, floor or unattributed
contacts invalidate capture. Invalid physics is separately reported.

The simulator provides scene geometry. This is not camera/point-cloud perception.
Arms hold their admitted endpoint joints during flight; sampled setup paths are
geometric admission evidence, not learned transit, grasp or actuator dynamics.
Nominal repeats do not establish noise or timestep robustness. Results are
development recipes at known positions, not a new policy or generalization rate.

## Budgets, persistence and isolation

Queue for the shared 4090 lease for at most 900 seconds. The acquired worker has
43,200 seconds (12 hours); start no new search after 36,000 seconds (10 hours). Reserve the remaining
worker time for pending confirmations and final controls. The independent CPU
audit has up to 7,200 seconds. The total supervisor limit is 51,300 seconds,
including queue time. Completion may happen earlier when the quotas are met.
Retain the existing storage-reserve checks before admitting each packet.

The initial CPU preparation probe admitted seven branches from four numeric
settings in 152 seconds. That small probe suggests about six hours of preparation
for 1,000 admitted drops before allowing for further rejections or physics;
it is a planning estimate, not a promised completion time.

Use the explicit 4090 UUID and cooperative lock. Recheck foreign 4090 processes
at packet boundaries; never stop another process to obtain capacity. Snapshot
protected result/source bytes and the 4070/Isaac process identities before and
afterward. No 4070, Isaac, service, cloud or hardware changes are authorized here.

Persist frozen requests before evaluation, then append results and packet
receipts that bind the raw archives, original IDs, proof hashes and row hashes.
Publish progress to local evidence files only. A native result becomes part of
the completed total only when its saved evidence can be authenticated.

**There is no in-place resume in this pass.** Existing six/bouncer adapters reset
packet counters on construction and cannot safely append to an old run. On an
interruption, preserve the whole partial directory and mark the task incomplete.
A later continuation needs a separate immutable study, reconciled committed
evidence, remaining per-cell quotas and a fresh explicit budget. Never overwrite
or silently discard an orphan archive or count a repeated request twice.

An independent audit reconstructs the plan, admission proofs, actual release
counts, control gates, native scores and confirmation selection. Quota shortfall,
audit failure, timeout or storage failure remains explicit. No new UI publication,
policy training, model promotion or video generation is part of this run.
