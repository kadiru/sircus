# Board-first capture and corrected selector study

The ball must first rebound from the board, then enter and remain entirely in
the bucket for one continuous second. Subsequent robot/table/other contacts are
allowed. Historical labels are preserved alongside version4 scores.

[Scoring contract](../../sircus/mjlab/SCORING_BOARD_FIRST.md) ·
[Study protocol](../../sircus/mjlab/BOARD_FIRST_STUDY.md)

## Saved evidence

The bounded historical pass covers **60 selected studies, 37,111 native drops**
and 3,688 setup rejections. Of 41,463 indexed rows, 664 copied metadata rows were
deduplicated. Original captures **19,620 → 19,985: +365, zero lost**. Invalid
physics remains excluded (284 drops); no missing evidence or evidence errors.
The independent re-scoring audit passed **298,782 checks**.

Of the 365 gains, **242 involve post-board robot contact** (118 from original v2
sources, 124 from v3). The remaining **123 are table-only gains from v2**, where
table assistance was still excluded. They do not represent a new benefit of
allowing robot contacts. Contact presence is not proof of a causal rebound.
These counts include controls and repeats; they are recorded attempts, not
unique settings, newly simulated drops, or a fresh policy accuracy estimate.

Native64 contributes 37,057 drops, 19,608 → 19,973 accepted captures. The 54
historical native128 drops remain 12 → 12. All new comparison drops use native64.
Seven additionally compatible study sources were discovered after this pass was
frozen and remain outside it; their identifiers are recorded in
[the sanitized result](rescoring.json). This is not every historical experiment.
The original records, scores and evidence remain immutable.

The neural-selector corpus below is a **subset**, not an extra 5,150 drops to
add to the broad totals. It has 5,150 authenticated native trajectories.
Re-scoring accepts 3,344 versus the original 3,331: 13 additional captures and
zero losses. The gains contain robot contact after the confirmed board rebound;
contact presence is not proof of a causal rebound. Invalid physics remains
excluded (27 trajectories). Independent re-scoring passed 41,375 checks.

Of the changed labels, ten belong to TRAIN, two to validation and one to the
historical TEST split. Only the original TRAIN labels affect gradients. The
same 44→64→32→2 supervised selector is fitted with equal total weight per
position in each head. Its outputs estimate setup feasibility and capture given
feasible setup. The refit uses 8,670 TRAIN and 2,131 validation candidate records,
including refused and admitted-but-unreleased proposals. These are not drop
counts: the native TRAIN and validation subsets contain 3,901 and 613 drops.
Validation selected epoch 500. Original weights remain frozen.

The comparison uses 50 new positions, both frozen selectors, identical proposal
libraries, a common contact-permitting flight heuristic, unchanged arm setup/path
checks, a 30-second decision allowance, and at most one drop per selector per
position. Native MuJoCo uses 64 substeps and holds the arms at checked endpoints.
The heuristic supports direct and one-table trajectories; it does not model
arbitrary robot-assisted bounce paths. This comparison tests corrected labels,
not an independently measured benefit of removing the flight-clearance penalty.

The fresh comparison completed and passed its independent audit (36,090 checks):

| Frozen selector | Captures / 50 | No checked setup | Released misses | Mean decision time |
|---|---:|---:|---:|---:|
| Original | 38 | 9 | 3 | 26.64 s |
| Retrained on corrected labels | 37 | 8 | 5 | 26.94 s |

Both captured at the same 37 positions. The original alone captured at
(575.704, −273.659) mm; the retrained model gained no captures. Original captures
were 26 direct and 12 table-assisted; retrained captures were 25 direct and
12 table-assisted. None of these fresh captures involved robot assistance.
No invalid-physics outcomes. Both nine-drop control blocks passed, for 101
native releases total: 83 comparison drops plus 18 controls. The 4090 lease
lasted 311.53 seconds. Decision time excludes initialization, shutdown and HTTP
handling; its common cap was 30 seconds.

Corrected labels did not improve this batch. Fifty positions provide a limited
comparison, not a table-wide guarantee. The live interactive planner is unchanged. No camera input, physical robot test,
driven-arm execution, noise robustness or arbitrary-table reliability is claimed.

Producer identities: scoring/rescore `1359b36`; selector fitting/preparation `3c57311`;
execution recovery `310bc48`.
Raw evidence and full checkpoints are saved on Ubuntu outside Git. Artifact identities and all 50 outcomes are in [comparison.json](comparison.json);
[rescoring.json](rescoring.json) records the historical counts and changed positions.


## What this does and does not change

The new scoring/reward adapter and contact-permitting ranker are explicit opt-ins.
The read-only results panel is `#boardFirstStudy`; the running interactive
planner and its old runtime remain unchanged. No new physics/perception model,
PPO update or arbitrary robot-bounce trajectory generator was trained. The
useful next direction is more diverse board-first assisted training examples
and candidate paths, not automatically promoting this one-point-lower refit.

## Reproducing the comparison

Use the pinned environment in `sircus/mjlab/requirements.txt` and authenticated
original neural-selector, parent control and corrected-grid evidence. Source
hashes, calibration, model weights, position lists and chosen commands are bound
in the experiment contract and evidence manifest. The Git checkout alone does not
contain the excluded raw evidence or machine-specific calibration.

Run `python -m sircus.mjlab.board_first_rescore --help` for explicit source
selection, then `python -m sircus.mjlab.board_first_selector_study --help`.
The ordered stages are `fit`, `prepare`, `run`, `finalize`, and `audit`.
Hide CUDA for fitting, preparation and audit. For `run`, expose only the 4090
and use the existing cooperative GPU lock. Keep the audited output immutable.
Publish using `python -m sircus.mjlab.board_first_publish` with both the completed
comparison directory and its independent audit. Publication does not activate a
new interactive planner.

The first native-stage launch stopped before creating a simulator because
`signal.alarm` received `3600.0` instead of integer `3600`. The explicit
`resume-empty` stage in `310bc48` verifies zero native evidence, authenticates
both source versions, and records an additive recovery receipt. The original
contract, both checkpoints, all 100 selected choices and original start receipt
remain unchanged. Recovery cannot resume a partially executed native test.
