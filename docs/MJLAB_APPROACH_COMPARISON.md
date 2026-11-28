# Frozen planner comparison: complete bucket approach

This experiment tests whether the new planner improves first-drop capture at
50 previously unused bucket positions. It does not train a network or replace
the interactive service. The native task is unchanged: the ball must first
bounce from the board, enter fully, and stay inside the bucket continuously for
one second. Later table, bucket or robot contacts may count toward a capture.
A partial exit resets the retention clock.

## Compared methods

| Receipt key | Method |
| --- | --- |
| `current` | The live hybrid planner's original frozen selector, source library and flight ranker. |
| `expanded` | The previously trained expanded selector and library, a larger fixed proposal menu, and the complete bucket-approach forecast. |

Both methods use the same arm/path checks and reference-first planning, switching
to the candidate's direct path after five seconds of reference-path search,
30-second decision limit, two independent persistent CPU workers, and 5 mm
predicted-clearance early-return threshold. Worker initialization and shutdown
are reported as full-request time separately from decision time. A deadline may
return an already completed, checked candidate; otherwise the position remains
a setup decline in the denominator.

The new menu retains up to 56 expanded-library proposals and adds up to 14
fixed tilt/gap variations around each of four seeds ranked by the frozen
expanded selector, at most 112 proposals. The unchanged planner checks at most
16 initial proposals, followed by its existing joint-alternative fallback when
needed. It cannot use native test outcomes to select or revise a shot.

The approach forecast uses compiled bucket geometry, the ball radius, and the
predicted path from release through board/table rebounds and entry. It lowers
a candidate's ranking when the path approaches a bucket wall. This is a
forecast, not a ban on contacts. Its simplified flight model can miss second
board rebounds and other useful contacts. Native captures are the deciding
measurement.

This compares two complete products. A gain cannot be attributed solely to the
bucket check: the library, selector, candidate menu and contact ranking also
differ from the current method.

Both comparison lanes include the same small correction to the shared IK
adapter: nested reference searches retain their explicitly supplied solver
options instead of receiving duplicate `preserve_known`/`seed` keywords.
Models, ranking rules, search budgets and endpoint/path gates are unchanged.
The live service has not received this correction or been restarted.

The first preparation attempt (`approach-comparison-v1`, producer `6f73b86`)
stopped after 78 saved planning requests when this bug appeared in the expanded
planner's alternate board-joint search. It performed **zero native drops**.
That failed directory is preserved. The corrected comparison uses a new source
snapshot and output directory; the failed attempt's positions join the
historical exclusion inventory before the new 50 positions are sampled.

The second attempt (`approach-comparison-v2`, producer `2e63eb6`) passed
preflight but sent each native shot separately. Measured packet times of about
40 seconds made the full comparison likely to exceed its one-hour execution
budget. Only this comparison process was stopped. Its immutable evidence
contains 16 completed evaluation records and 9 pre-controls; an interrupted
packet may not have a complete observation, so these are saved records rather
than a claim about total attempted releases. No target outcomes were inspected
or used to change the planners. V3 uses fresh positions excluding both earlier
sets and batches independent simulation worlds. Models, menus, decision limits,
scoring and physics settings remain unchanged.

## Frozen inputs and placements

- Copy both checkpoint files and all library inputs byte for byte from the
  completed, independently audited `expanded-selector-v1` study. No fitting,
  optimizer updates or new training labels occur here.
- Bind the current checkpoint/library to the audited `stalled-reference-v1`
  live-planner inputs, and the scene, calibration and native64 settings to the
  original controls.
- Sample 50 positions with seed `2026112717`: two in each of 24 inter-grid
  rectangles, plus two near fixed robot bases. Respect the table boundary and
  conservative fixed-base clearance.
- Exclude all bucket coordinates discovered in the inventoried historical
  metadata under both output roots, published baselines, the training/library
  datasets and the prior comparison's 50 positions. Keep at least 5 mm between
  new positions and those coordinates. This is novelty against available
  evidence, not a claim about missing or unpublished data.
- Shuffle position order with seed `2026112718`. Exactly 25 positions give each
  method the first CPU submission turn. Both independent planning lanes run
  concurrently at each position.
- Before any drop, freeze `native-schedule.json`. Group admitted commands by
  method and native adapter (`reference` or `direct_first`), retain the frozen
  position order within each lane, and execute up to 32 independent worlds per
  packet. Visit the current then expanded method, reference then direct
  adapter, for each 32-row chunk. Native order is explicitly batched rather
  than paired position by position; every method still has one attempted shot
  at each of the same 50 positions.

## Execution and audit

The stages are `freeze`, `prepare`, `preflight`, `run`, `finalize`, `audit`.
Use `python -m sircus.mjlab.approach_comparison_study --stage STAGE` with:

```text
--neural-study /mnt/robot-training/sircus-mjlab/neural-selector-v1
--neural-audit /home/kadir/sircus-mjlab/output/neural-selector-v1-audit.json
--current-planner-study /home/kadir/sircus-mjlab/output/stalled-reference-v1
--current-planner-audit /home/kadir/sircus-mjlab/output/stalled-reference-v1-audit.json
--expanded-study /mnt/robot-training/sircus-mjlab/output/expanded-selector-v1
--expanded-audit /mnt/robot-training/sircus-mjlab/output/expanded-selector-v1-audit.json
--library /mnt/robot-training/sircus-mjlab/output/expanded-shot-library-v1
--parent-study /home/kadir/sircus-mjlab/output/fixed35-sixhour-v2
--parent-audit /home/kadir/sircus-mjlab/output/fixed35-sixhour-v2-audit.json
--reference /home/kadir/sircus-mjlab/output/corrected-grid-v1
--assets /home/kadir/sircus-isaac/assets
--history-root /home/kadir/sircus-mjlab/output
--history-root /mnt/robot-training/sircus-mjlab/output
--output /mnt/robot-training/sircus-mjlab/output/approach-comparison-v3
--gpu-lock /home/kadir/sircus-isaac/scene.lock
--gpu-wait-s 3600
--preflight-output /mnt/robot-training/sircus-mjlab/output/approach-comparison-v3-preflight.json
--audit-output /mnt/robot-training/sircus-mjlab/output/approach-comparison-v3-audit.json
```

Run CPU stages with `CUDA_VISIBLE_DEVICES=''`. For `run`, expose only the
verified 4090 UUID through `CUDA_VISIBLE_DEVICES`; the cooperative lease waits
for other work rather than stopping it. Freeze the computational source before
`freeze` and use that exact source for every stage. Audit outputs are outside
the immutable study directory. The existing Isaac environment and 4070 workload
are never altered.

Preparation saves the full menu, checked prefix, path and endpoint evidence,
worker/deadline receipts, predictions and chosen command. All preparation bytes
are frozen before the independent preflight. Native execution refuses a missing
or mismatched passed preflight, and never replans a prepared command. Preflight
also verifies every selected command occurs exactly once in the native schedule,
with the correct adapter and within-lane order. The two adapters sharing a
method directory use a common monotonic cache index so their arrays cannot
overwrite each other.

Native execution is bounded by one acquired GPU-hour and 118 releases: at most
100 evaluation drops plus 9 original matched controls before and 9 after. Each
control block contains three repeats each of the original direct capture,
table-assisted capture and known miss. Each method receives one release per
position; declined setups receive no substitute attempt. No test resampling,
parameter tuning, or selection after observing a drop is permitted.

A queue expiration before native begins can be retried explicitly using the
same frozen evidence. A partial native run cannot be implicitly resumed or
overwritten. Failed outputs remain available for diagnosis.

The final independent audit rechecks exact model/source bytes, historical
exclusions, every accepted planning publication, deadline/early-return rules,
path proofs, declared packet boundaries and the native request ledger. It
checks each independent world's record ID and initial state, and recomputes historical and current
scores from every dense native trace, verifies time-zero identity and controls,
and rejects uncounted or duplicate releases.

## Reported measurements

For each method: captures out of all 50 positions, setup declines, released
misses, invalid physics, and mean/median/p90/max decision and full-request times.
P90 is the nearest-rank value: the 45th sorted observation for 50 positions.
Also report paired gains/losses and per-position outcomes. GPU queue time and
native lease time are separate from planning latency. No claim of perception,
learned arm motion, real-robot validation, or guaranteed capture at arbitrary
bucket positions follows from this simulation test.
