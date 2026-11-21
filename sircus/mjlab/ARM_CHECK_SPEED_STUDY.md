# Arm-check speed diagnosis and proposed test

This follow-up investigates the missed placement in the completed
`interactive-fast-acceptance-v1` comparison. All numbers below are historical
evidence; no improvement or new capture outcome is claimed here.

## Observed timeout

Case02 places the bucket at **(971.201, -285.388) mm**. Fast planning returned
no checked shot after **30.168 s** HTTP wait; the original planner captured
after **172.395 s** planning. The fast worker was reused with **0 s startup**.
Its recorded CPU stages were:

| Stage | Seconds |
|---|---:|
| Generate shot menu | 0.058255 |
| Neural ordering | 0.013297 |
| First arm setup check, interrupted at deadline | 29.927377 |
| Physics ranking | 0.000237 |

No candidate or request proof completed before the deadline. Arm checks used
99.76% of the 30-second decision budget; the profile does not establish which
internal instruction was executing at interruption.

The original planner's completed proof isolates the expensive common setup:
**0.384298 s endpoint IK**, **85.733423 s path search**, **86.136828 s total**.
The endpoint is reachable. Both direct serialized movement orders violate
the required 5 mm clearance: bouncer link3 comes within **3.824940 mm** of
bucket wall36 on the forward segment; the reverse segment encounters link2
at **4.910049 mm** from bucket wall12. Six vertical/translate/orient
alternatives fail before bounded RRT finds a 22-waypoint path, with minimum
clearance **5.016732 mm**.

Both rejected direct orders consumed 543 gate checks each. RRT extensions
consumed 538 bouncer and 102 dropper checks; its final forward/reverse path
audit consumed 1,110 checks. The eventual selected shot instead passed a
direct home-to-shot path with 3 waypoints and 1,186 checks, after its
anchor-to-shot suffix failed. Thus this successful route did not use the
expensive reference path. The completed original run explains a plausible
shared bottleneck; the interrupted fast run has no complete internal proof.

## Evidence and reproduction

Runtime evidence remains outside Git on Ubuntu. Read these JSON records;
do not rerun or overwrite the historical requests:

```text
/home/kadir/sircus-mjlab/output/interactive-fast-acceptance-v1/report.json
/home/kadir/sircus-mjlab/studies/interactive-fast-v1/evidence/fast/interactive-24853e1aa5ee4a0d9b4d245359fed1e5/plan.json
/home/kadir/sircus-mjlab/studies/interactive-fast-v1/evidence/fast/cpu-lane-eddef504288d4b8e844f7d7e1ebaedcf/fast-journals/interactive-24853e1aa5ee4a0d9b4d245359fed1e5/
/home/kadir/sircus-mjlab/studies/interactive-fast-v1/evidence/interactive-dbec7cba8c084e7bb20dd3259bf555c8/plan.json
/home/kadir/sircus-mjlab/studies/interactive-fast-v1/evidence/interactive-dbec7cba8c084e7bb20dd3259bf555c8/native-heading/resolutions/8ceed83067685323435839f183ad39e3839e1fa0df756b5cad083359b02c91db.json
/home/kadir/sircus-mjlab/studies/interactive-fast-v1/evidence/interactive-dbec7cba8c084e7bb20dd3259bf555c8/native-heading/heading-connections/3298d1391c04326e458552e71d47d586be3cbdbeeb57cab99c0cba552c508936.json
```

The resolution record's `timing` provides the IK/path split. Its
`proof.attempts` records the rejected direct/Cartesian paths and RRT. Count
executed samples as `total_samples` for a passed edge or `sample + 1` for
a failed edge. Count a path once: its final proof is also repeated inside
the attempt history. The selected heading connection's `proof` is the
actual accepted route; `path_attempts` retains the rejected suffix.

## Focused change and validation plan

Two per-instance, opt-in optimizations preserve the historical audited
implementation and its accepted-shot checks:

1. Vectorize conservative AABB distances for the same geometry pairs,
   retaining the exact distance/SAT/contact/corridor checks near obstacles.
2. Stop a path audit after its first failed segment. Audit every segment in
   both directions for an accepted path, in the original order with the
   original 0.5-degree sample spacing. Mark partial rejection evidence
   explicitly; it is never an accepted certificate.

The path installer is `arm_path_speed.install_path_short_circuit(planner)`.
It changes only the supplied `SerializedPlanner` instance. Historical
source files, thresholds, IK branches, model weights, physics and scoring
remain unchanged. CPU tests compare accepted certificates and every gate
sample against the original, and check early failures in both directions.

Next measure CPU planning on this diagnostic case, then freeze fresh
placements and compare baseline versus optimized planning time, setup rate
and first-drop capture at 64 substeps. No-setup attempts stay in the capture
denominator. Native testing is restricted to the 4090; the 4070 workload and
Isaac setup remain outside this task. Do not claim faster planning or equal
capture performance until the comparison completes.
