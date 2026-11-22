# A bounded fallback when reference-path search stalls

This experiment keeps the ordinary reference-first planner, but changes what
happens when its reference-path search consumes **5 seconds** without returning
a checked shot. It then tries the candidate-first planner within the **same
30-second decision deadline**. The model, shot ordering, ball physics and
capture scoring stay frozen. No new policy is trained.

A reference path moves the arms from their home pose to a previously known
shot pose. Searching this path can be expensive even when the selected shot
ultimately uses a direct home-to-shot path. The previous experiment recovered
one such target with candidate-first planning, but slowed the average request
on fresh positions. The new experiment tests a delayed fallback instead of
changing the order for every placement.

## Frozen comparison

| Item | Protocol |
|---|---|
| Baseline | Reference-first planning with optimized geometry/path checks |
| Hybrid | Same baseline; switch after 5 seconds of actual reference-path search |
| Total decision budget | 30 seconds, shared across both phases |
| Fresh evaluation | 20 new bucket positions, one selected drop per method per position |
| Diagnostic | Previously timed-out target, separate from the 20 fresh positions |
| Accepted arm paths | Same 5 mm clearance, detailed collisions, 0.5° samples in both directions |
| Native drops | MuJoCo, 64 substeps, arms held at the selected endpoints |
| Controls | Matched saved controls before and after comparison drops |

The fallback is cooperative and sequential. It interrupts only before a
selected shot has been published; it does not run two planners in a race.
The decision deadline does not restart when the fallback begins. Full request
time also includes worker initialization and result handling, which can take
it beyond 30 seconds. Browser/network time, GPU waiting and native drop time
are separate measurements.

Candidate-first planning can use a different initial joint-angle hint from
reference-first planning. It therefore may find different arm configurations
for the same requested board and release position. Every accepted endpoint
and complete path must still pass the same checks. No clearance test is
relaxed to recover a timeout.

Capture means a board rebound followed by one continuous second with the
whole ball inside the bucket. Direct and table-assisted captures count;
robot-assisted outcomes remain separately labeled. No-setup positions remain
in the denominator. A feasible setup or a predicted capture is not a measured
capture. This experiment does not validate camera perception or moving-arm
execution.

## When the interactive planner may change

Before observing results, the rollout criteria are:

1. The independent evidence audit passes.
2. Hybrid captures at least as many fresh positions as the baseline.
3. Its mean full planning request is no more than **10% slower**.
4. Its selected native drop captures at the old timed-out target.

All four are required. Twenty fresh positions provide limited comparison
evidence, not a guarantee across the table. An adapter must separately verify
the frozen source, input and proof identities before advertising the hybrid
planner. The existing fast and original planners remain selectable. A failed
request never starts another drop or changes planners silently.

## Results and reproduction

The task driver is `python -m sircus.mjlab.stalled_reference_study`. It freezes
inputs, coordinates and selected commands before native outcomes are observed.
The output report uses `stalled_reference_study`, method IDs `baseline` and
`hybrid`, separate per-position outcomes, and a separate diagnostic replay.
Raw evidence, generated configurations and logs remain outside Git.

The UI section is **`#stalledReference`**. It displays progress while running.
Final counts appear only after the report is complete, the audit passed and
its totals reconcile with the per-position records. Missing native outcomes
remain “Not run.” The UI report alone never enables a planner; available
choices come from the running server's verified capabilities.

This document describes the frozen protocol. Record the actual results,
artifact hashes and execution paths here after the comparison is completed
and audited. Until then, no improvement or interactive rollout is claimed.
