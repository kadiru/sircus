# Case 01: a longer clearance window

This is a development diagnostic at fixed 64 substeps, not policy training or a
fresh held-out evaluation. The previous refined-distance test still reproduced
case 01's robot contact three times. Its forecast check ended at 321–327 ms after
board rebound, before the measured robot contact at about 335 ms.

## Frozen intervention

- Reuse original `arm-clearance-v1` case `arm-clearance-test-01`, candidate 14,
  with exactly the saved endpoint joints, physical command and five forecasts.
- Short window: the previous refined support bound through descending rim-band
  start. Long window: the same bound through the predicted first descending
  crossing of ball-center height `tabletop_z + ball_radius`.
- Retain the 2 mm required margin, 2 ms temporal sampling, 1.5 s cap, collision
  meshes, unchanged 64-substep native dynamics, strict scorer and ranking.
  Require both original and extended certificates to pass. No new candidate is
  selected when the longer window refuses the shot.
- Three requested repeats per method, plus three positive and three negative
  matched controls before and after: exactly 18 slots. Abstentions stay in the
  denominator and have no native flight or contact observations.
- No fitting, noise, 128-substep replay, automatic policy promotion or transit
  validation. Held endpoint configurations do not establish driven-arm safety.

CPU preparation predicts a longer boundary of 397–402 ms. The conservative
clearance bound changes from +12.05 mm to −36.07 mm, so the long method withholds
this candidate. A negative lower bound means clearance is not certified; it is
not itself a measured collision. Native repeats and unchanged controls test
the admitted baseline and confirm the refusal behavior without executing it.

## Evidence and interpretation

Freeze the full request plan, source hashes, rule and endpoint proofs before any
native outcome. GPU use is restricted to the 4090 UUID with the existing shared
lock, at most one acquired GPU-hour, 18 requests and an 8 GiB storage reserve.
The 4070 and Isaac workloads are not modified. A separate CPU auditor recomputes
the time boundary by bisection, verifies geometric support witnesses and exact
joint branches, and rescores recorded native traces with the existing scorer.
Only an independently bound passing audit may publish verified UI outcomes.

The window extension assumes uninterrupted ballistic flight. Bucket and table
rebounds change velocity and must be diagnosed separately. Reaching table
elevation does not mean the projected XY is within the tabletop footprint.

The current strict route score excludes intervening table, floor and robot
contact. Bucket retention is displayed separately. A historical table-assisted
capture exists, but permitting that route for future training would require an
explicit new objective/version; this experiment does not change old definitions.

## Reproduce

Run the pipeline from a frozen source snapshot using the existing isolated mjlab
environment. Supply the authenticated original study/audit, immutable fixed64
native source, read-only assets, new output directory, existing shared GPU lock
and current interface-report path. The pipeline prepares on CPU, queues for the
4090, runs the finite native test, audits on CPU and publishes one additive key.

```text
python -m sircus.mjlab.horizon_window_pipeline
  --parent-study ORIGINAL_ARM_CLEARANCE
  --parent-audit ORIGINAL_AUDIT
  --native-source PINNED_POLICY64_SOURCE
  --assets READ_ONLY_NATIVE_ASSETS
  --output NEW_EVIDENCE_DIRECTORY
  --gpu-lock EXISTING_SHARED_LOCK
  --display EXISTING_INTERFACE_REPORT
```

UI: `#clearanceHorizon`. Compare requested slots, native releases, abstentions,
strict captures, bucket retention and measured robot contacts. Refusal of a bad
shot is not evidence of a better executed trajectory or improved capture rate.
