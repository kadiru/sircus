# Finger clearance and flight-horizon diagnosis

This development experiment follows the completed [50-position clearance
comparison](ARM_CLEARANCE.md). It does not retrain a model or activate a selector.
The original results, geometry, scoring and source evidence remain immutable.

## Findings from saved native evidence

The face-only geometric lower bound was unnecessarily pessimistic around some
finger corners. A tighter conservative support-plane certificate gives:

| Prior case | Original guarded bound | Refined guarded bound | Decision at +2 mm |
|---|---:|---:|---|
| 04 | −10.805 mm | −2.963 mm | Reject |
| 07 | −3.632 mm | +4.541 mm | Pass |
| 39 | −7.474 mm | −7.321 mm | Reject |

These are predicted clearance bounds with the same between-sample allowance,
not measured continuous clearance. Case 07's separate exact-distance diagnostic
was +12.843 mm after allowance; the implemented conservative certificate is
the smaller +4.541 mm value above. Reconstructed clearance at every saved
first-flight sample was approximately 1.691, 17.086 and 0.020 mm respectively,
using the admitted held joints. That reconstruction is not a continuous trace.

All five remaining robot-contact cases occurred after all five frozen predicted
rim-start horizons. Case 01's first robot contact was 334.896 ms after observed
board rebound, versus predicted horizons of 321.488–326.923 ms. The predicted
position differed by approximately 33–44 mm there. Some other cases first hit
the bucket or table, after which a free-flight ballistic forecast is no longer
applicable. A hypothetical longer ballistic check catches case 01 only.
Neither horizon extension nor model correction is part of this test.

## One frozen correction

`arm_clearance_distance.PreciseArmClearanceGate` retains the old rule but tightens
its lower bound. CPU MuJoCo geometry queries propose a direction; acceptance
uses a support plane recomputed over every compiled collider vertex. The native
distance return value never directly decides acceptance. At each sample use the
larger of the original face-plane bound and the certified support-plane bound,
then subtract ball radius and the unchanged inter-sample swept-distance bound.

The 2 mm safety margin, five predictions, 2 ms maximum sample spacing, old
rim-start horizon, 1.5 s horizon cap, collision geometry, checkpoint weights,
16 candidates, ranking, top-four endpoint budget, physics and scorer are fixed.
Both arms remain at the exact previously admitted joint branches. Only the
intended board impact geometry is excluded. Missing clearance means abstention,
and an abstention remains an attempted request with no fabricated observations.

## Prespecified development replay

- Layouts `00`, `01`, `04`, `07`, `39`, `44` from the prior clearance study.
- Original clearance gate versus corrected clearance gate; three exact repeats
  per layout/method: **18 requested shots per method**.
- Six matched positive/negative controls before and six after: **48 total
  requested slots**, including policy abstentions that never enter simulation.
- Native calibrated 64-substep contract and identical scoring. Complete capture
  requires a board bounce, whole-ball entry and one continuous second inside,
  with disallowed contacts handled by the existing versioned scorer.
- Freeze the complete plan, certificates and source hashes before acquiring the
  dedicated 4090 slot. No native search, fitting, outcome-based substitutions,
  noise or 128-substep tests. Maximum one acquired GPU-hour and 24 hours queued.
- Six previously observed layouts are development evidence, not a fresh
  generalization test. Exact repeats check execution repeatability, not noise
  robustness. Report per-layout outcomes as well as repeated-shot totals.

## Execution and verification

Use a new evidence namespace and an immutable source snapshot. First build the
CPU diagnosis from the authenticated prior study:

```bash
CUDA_VISIBLE_DEVICES='' python -m sircus.mjlab.clearance_prediction_diagnosis \
  --study ORIGINAL_STUDY --audit ORIGINAL_AUDIT --assets EXISTING_ASSETS \
  --output NEW_DIAGNOSIS_DIRECTORY
python -m sircus.mjlab.clearance_correction_pipeline \
  --parent-study ORIGINAL_STUDY --parent-audit ORIGINAL_AUDIT \
  --native-source PINNED_NATIVE_SOURCE --assets EXISTING_ASSETS \
  --output NEW_CORRECTION_STUDY --gpu-lock EXISTING_SHARED_LOCK \
  --diagnosis-report NEW_DIAGNOSIS_DIRECTORY/diagnosis.json \
  --diagnosis-audit NEW_DIAGNOSIS_DIRECTORY/diagnosis-audit.json \
  --display EXISTING_DERIVED_DISPLAY
```

The pipeline separates CPU planning, GPU execution and independent CPU audit.
It respects the shared GPU lock and foreign 4090 workloads, reserves 8 GiB of
storage, and can terminate only its own worker. The 4070 and Isaac runtime are
untouched. Existing collision assets are read-only inputs.

`clearance_correction_verify` independently reconstructs support-plane bounds,
exact selected endpoint branches, requested-slot accounting, raw contact traces,
native options, calibrated scores and matched controls. Proofs bind every source,
plan and evidence file. A completed producer alone is not a verified result.

The additive `#clearanceDiagnosis` panel reads `clearance_prediction_diagnosis`
from the existing viewer report. Diagnosis and native correction have separate
proof gates. It shows missing observations as unknown and never starts a shot,
changes policy selection or hides older results. Static publication requires no
service restart. Driven-arm transit, incoming-drop clearance, sensor perception
and physical validation remain outside this experiment.
