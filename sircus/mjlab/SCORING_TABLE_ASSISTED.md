# Table-assisted task scoring, version 3

This is an explicit, opt-in task-definition change authorized November 8, 2026.
A board → table → bucket capture counts. Historical version 2 files, labels,
rewards and checkpoints retain their original definitions. Importing or
re-scoring with these modules does not activate a new runtime scorer, train a
model, change a physics engine, or deploy a policy.

## Definition

The production module is `table_assisted_score.py`. The independent checker is
`table_assisted_score_oracle.py`. Both expose:

```python
score_trace(scene, bucket_xy_m, normal, calibration, arrays)
reward(scored_row)
```

| Identity | Version |
| --- | --- |
| Score | `sircus-table-assisted-native-capture-v3` |
| Route | `sircus-table-assisted-board-route-v3` |
| Reward | `sircus-table-assisted-shot-reward-v3` |
| Reused containment | `sircus-calibrated-native-capture-v2` |

The calibrated containment kernel is unchanged: a measured board rebound,
qualifying downward mouth entry of the whole ball, then one continuous second
inside the bucket. Partial exits reset retention. Rim-assisted captures count.
Invalid-physics guards and calibrated wall/base tolerances remain unchanged.
The terminal is still the first qualifying capture, invalid-physics state,
fall below the existing bound, or end of evidence. Later contacts cannot revoke
an already completed capture.

The only contact-rule change is permission for **table contact**. Bucket and
table contacts are allowed; floor, robot and unattributed contacts still prevent
task success. There is no new restriction on table contact before the board.
This task-definition change does not recalibrate table restitution or establish
the accuracy of table-impact physics; existing numerical evidence keeps its scope.
Counts use exactly the existing force intervals `[0, physics_steps)`: an
interval ending at the terminal counts; an interval starting at it does not.
The force threshold stays strictly greater than 0.01 N.

## Output and compatibility

- `success` retains the unchanged calibrated bucket-retention result.
- `route_success` and `complete_success` use the version 3 task rule.
- `external_contact_steps` is preserved: union of table, floor, robot and
  unattributed intervals. It is a diagnostic, **not the version 3 reward gate**.
- `forbidden_contact_steps` counts the union of floor, robot and unattributed
  intervals; a coincident multi-channel contact counts as one interval.
- `table_contact_steps` records table-contact intervals. `direct_capture` and
  `table_assisted_capture` partition accepted captures by absence/presence of
  table contact; `route_class` is `direct`, `table_assisted`, or `not_captured`.
  The label records contact presence, not a proven table rebound, contact order
  or causal contribution. Rebound claims need a separate velocity/event audit.
- This route classification is independent of mouth quality. A table-assisted
  capture can enter cleanly (`route_quality=2`) or with rim assistance/marginal
  clearance (`route_quality=1`). Misses/rejected routes have route quality zero.
- `single_board_bounce_success` means one board-contact episode in an accepted
  route. The existing `single_bounce_success` field is its compatibility alias.
  Neither field means there was only one physical bounce: table and bucket
  contacts may also occur. User-facing labels should say “one board episode.”

## Reward

All numerical weights are unchanged. Let `d` be the existing closest-opening
distance in metres, and `B` indicate a measured board rebound:

```text
shaped = 0.15 B + 0.5 exp(-d / 0.12)
invalid setup or physics: -1
any forbidden contact: shaped - 0.5
otherwise: shaped + 0.5 min(retention_seconds, 1)
                  + 3 accepted_capture
                  + 0.3 clean_mouth_capture
                  + 0.1 one_board_episode_capture
```

Table contact alone adds no penalty and removes no bonus. For an illustrative
clean capture with `d=0`, one board episode and one second retained, the reward
is **4.55**, with or without table contact. If a forbidden contact occurred, it
is **0.15** instead. These are arithmetic examples, not measured performance.

A non-invalid reward input must explicitly carry all three version 3 identities;
old version 2 rows cannot silently acquire new rewards. Invalid setup rows can
return -1 without fabricated native observations. A future trainer must opt into
these functions, version its artifact/reward contract, and undergo a new frozen
evaluation. Re-scoring an old policy's recorded outcomes is not retraining it.

## Future native-training integration

`table_assisted_env.make_environment` is an explicit factory for the new scoring
adapter (`sircus-table-assisted-held-endpoint-v3`). It executes the existing
native rollout once, then scores the same dense observations with both version 3
implementations. Rows retain `historical_v2_score`, the native identity and the
original containment facts. Added scoring time is CPU processing, not another
physics rollout. Importing the factory does not initialize a GPU.

A future, separately authorized experiment can supply the factory to the existing
requested-action adapter. This integration example assumes its inputs have
already been authenticated by the new experiment's contract:

```python
from sircus.mjlab.calibrated_adapter import CalibratedAdapter
from sircus.mjlab.expanded_actions import WIDE_BOUNDS
from sircus.mjlab.table_assisted_env import make_environment
from sircus.mjlab.table_assisted_score import reward

adapter = CalibratedAdapter(
    ref, assets, new_output_directory,
    prior_records=prior_records, calibration=calibration, home=home, batch=32,
    environment_factory=lambda: make_environment(
        ref["scene"], assets, calibration,
        n=32, bounds=WIDE_BOUNDS, substeps=64,
    ),
)
# The future trainer applies reward(row) to each evaluated row.
# Rejected setups return -1 without claiming a native release occurred.
```

The experiment must record a new study identity, all three version 3 scoring
identities, the scoring-adapter version, source hashes and the unchanged physics
settings. Do not reuse an old frozen training contract or overwrite its output
directory. The requested-action admission layer remains version 2; this does not
mean its native rows use version 2 rewards. Their explicit scoring identities
and `score_adapter_version` identify the opt-in behavior.

The factory inherits the existing single-4090 device guard, but does not acquire
the shared GPU lease. A future runner must acquire that lease, pin the authorized
4090 UUID and enforce its run budget before constructing the native environment.
This recipe neither launches training nor authorizes a 4070 or Isaac workload
change.

## Independent checks and evidence

Production reuses the untouched version 2 production containment kernel; the
oracle reuses its independently implemented version 2 oracle. Their new contact
counts, route decisions and reward arithmetic are implemented separately.
Synthetic tests cover table permission, all forbidden channels, mixed contacts,
terminal boundaries, partial exits, clean/rim quality, invalid physics, missing
board rebound, version opt-in and preservation of historical source bytes.
Saved native re-scoring must verify cache identity, calibrated scene, source and
historical labels before writing separate derivative results.

## Saved-shot rescoring workflow

`table_assisted_rescore` indexes saved `records.jsonl` and `shots.jsonl`,
verifies scene/calibration identities and dense archive checksums, then compares
production and independent-oracle scores under both definitions. Archive identity
and row index remove duplicated observations; request names are scoped to their
study. Missing or incompatible evidence stays unscorable. Contradictory evidence
fails verification. No GPU or physics engine is initialized.

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
  -m sircus.mjlab.table_assisted_rescore \
  --root SAVED_OUTPUT_ROOT --scene-contract MATCHING_SCENE_CONTRACT \
  --output NEW_DERIVATIVE_DIRECTORY --workers 2
```

Add `--index-only` first with a separate output directory to inspect coverage.
Use `--source STUDY_DIRECTORY` repeatedly to constrain a source set. All paths
are relative to the saved output root except `--root` and `--output`. Dense data
and generated manifests remain outside Git. Historical inputs are never edited.
The derivative includes old and new scores, source identities, exclusions,
per-study/method/phase/substep totals, examples, and an audit binding the exact
summary, manifest and rule bytes. These are retrospective saved-shot outcomes,
not a new held-out policy evaluation.

`prepare_table_assisted_scoring` validates that derivative and can write a
sanitized scalar snapshot for a repository report. Its optional HTML installer
requires the target's expected SHA-256 and replaces only its own panel atomically.
The existing viewer serves the updated HTML without a service restart.
