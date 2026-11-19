# Frozen arm-clearance comparison

This experiment tests one selection change on **50 new paired bucket positions**.
The current flight-consistent selector and the new clearance selector use the
same frozen learned ensemble, imitation/PPO proposal sources, 16-candidate bank,
flight ranking and at most four endpoint-geometry visits. No model is trained.

**Completed result:** [audited scalar evidence and examples](../../baselines/mjlab-arm-clearance-20261108/README.md).
Captures fell from **20/50 to 16/50**, with zero paired gains and four losses.
Robot-contact trials fell from 12/44 releases to 5/32, but the rule withheld
12 requests, including three baseline captures. Another changed shot lost a
capture. All 112 slots and both matched control groups passed the independent
1,461,935-check audit. The rule remains an evaluated candidate; the interactive
selector was not changed.

The earlier +60 mm intervention improved three diagnosed cases from 0/9 to 9/9
captures, with post-bounce robot contacts falling from 9/9 to 0/9. Those cases
are development evidence. Their results are not counted in this comparison,
and a universal +60 mm gap increase is not applied here.

## Frozen selection rule

For the new selector only, an endpoint-admitted candidate must also satisfy
`arm_clearance_selector.RULE` before a native shot is requested:

1. Use all five frozen ensemble predictions of the ball's post-board velocity.
2. Place both arms' collision geometry at the exact admitted joint positions.
   Exclude only the intended impact-board collision geometry. Use convex mesh
   hulls and enclosing boxes for primitive geometry conservatively.
3. Check the predicted spherical ball path from the nominal board tangent point
   to the first descending rim-band boundary, with **2 mm additional margin**.
   Sample no farther than **2 ms** apart and subtract a swept-distance bound
   between samples. All five predicted paths must pass.
4. Reject missing predictions or a horizon beyond **1.5 s**. If no candidate
   within the fixed geometry budget passes, **abstain**. The unexecuted request
   remains in the 50-request denominator and is shown as a clearance abstention.

This is a conditional geometric check, not observed native ball clearance. It
does not test incoming drop clearance, dynamic arm transit, bucket collisions,
one-second retention or learned-model accuracy. Later rim contact can change
the real ball path. The native scorer remains authoritative.

Before freezing TEST, CPU checks on the three prior development cases produced
the following conservative **predicted** ball-to-arm clearance bounds:

| Prior case | Original gap setting | Previously successful +60 mm setting |
|---|---:|---:|
| 12 | −24.759 mm; reject | +25.112 mm; pass |
| 24 | −25.161 mm; reject | +25.143 mm; pass |
| 48 | −18.963 mm; reject | +18.006 mm; pass |

The required bound is +2 mm. These are geometry/model checks, not six new native
shots. Within the unchanged top-four proposal lists, case 24 has a clear
alternative; cases 12 and 48 abstain. No +60 mm proposal was added to the bank.
The rule fingerprint is
`ecc86f1c701437ca9da10988ddda8ba33a6c4c84f5d523c57bfa09dfc737503e`.

## Comparison protocol

- Seed `20261808`; 50 newly frozen TEST positions within X 100–1240 mm and
  Y −650–250 mm. Carry forward the source studies' recorded train, validation
  and test exclusions, including the previous 50 diagnosed layouts, with the
  declared minimum 10 mm layout separation.
- Two methods, one requested shot per position: **100 evaluation requests**.
  Six matched controls before and six after give **112 total requested slots**.
- Same calibrated complete-route scorer and native **64 substeps**. Success
  requires board bounce, whole-ball entry and one continuous second inside,
  without a disallowed robot/table/floor route. Clean captures stay separate.
- Freeze the rule, source hashes, model hashes, complete candidate rankings,
  selections and requested actions before the first native outcome. No native
  per-position search, retrospective switching or tuning against this TEST set.
- Report captures/50, clean captures, other setup rejections, clearance
  abstentions, native misses and invalid physics. Show paired gains/losses.
  Measured robot contacts use force strictly greater than 0.01 N; report known
  native observations and unexecuted/unknown cases separately. Contacts after
  scoring termination do not retroactively change the completed score.
- CPU preparation precedes the dedicated RTX4090 slot. The worker honors the
  shared scene lock and existing foreign workload, with at most 24 hours queued
  and one acquired GPU-hour. The 4070 and Isaac are not modified or stopped.

## Execution and independent verification

Use a new output namespace and an immutable source snapshot. Preserve every
parent checkpoint, study, raw trace and audit. The bounded supervisor accepts:

```bash
python -m sircus.mjlab.arm_clearance_pipeline \
  --parent-study PARENT --parent-audit PARENT_AUDIT \
  --ensemble-study ENSEMBLE --ensemble-audit ENSEMBLE_AUDIT \
  --consistent-study CONSISTENT --consistent-audit CONSISTENT_AUDIT \
  --gap-study GAP --gap-audit GAP_AUDIT \
  --native-source PINNED_NATIVE_SOURCE --assets EXISTING_ASSETS \
  --output NEW_STUDY --gpu-lock EXISTING_SHARED_LOCK \
  --display EXISTING_DERIVED_DISPLAY
```

These placeholders are host configuration, not checked-in paths or credentials.
Do not run a duplicate supervisor or replace a frozen source snapshot. The
pipeline runs a separate CPU auditor after native completion. `arm_clearance_verify`
checks request/source identities, predictions, exact endpoint admission,
conditional clearance, native dense traces, contact windows, controls and score
accounting. A completed worker without a matching passed audit is not a result.

## Read-only result interface

`arm-clearance.{html,css,js}` adds `#armClearanceStudy`, reading
`arm_clearance_comparison` from the existing `/api/study` response. Insert its
style, section and module without replacing older panels or restarting the
viewer. `prepare_arm_clearance_report` authenticates the manifest, source/plan
bindings and full independent proof before publishing verified outcomes:

```bash
python -m sircus.mjlab.prepare_arm_clearance_report \
  --display EXISTING_DERIVED_DISPLAY --study NEW_STUDY --audit NEW_AUDIT
```

The panel separates pending, queued, running, awaiting-audit, failed and verified
states. All 50 paired positions remain visible, including abstentions and setup
rejections. A selected example shows exact board position, tilt and gap,
predicted clearance and measured contact/outcome. Missing observations remain
unknown. It cannot launch a shot, change the active interactive solver or deploy
the candidate rule.

## Limits and next decision

This is a finite frozen-selector evaluation, not another RL training run. It
uses simulator geometry and held endpoints; board/bucket dimensions retain their
existing provisional status. No noise, 128-substep convergence, grasping, driven
transit, perception or physical validation is established. Native performance
may decrease if the conservative gate rejects useful shots. Compare captures
and paired changes together with contacts and abstention rates before deciding
whether to continue, revise the rule on development data, or train again.

For this completed comparison, do **not** promote the candidate as a capture
improvement. Among the 32 pairs executed by both methods, zero robot contacts
were removed and one was added; the aggregate reduction came from abstention.
Review the rejected successful shots and remaining flight-model/secondary-contact
errors before generating better candidates and freezing another test. Preserve
the rule and this result; do not tune against these 50 positions and call a rerun
fresh evaluation.
