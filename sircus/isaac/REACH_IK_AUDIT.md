# Why the current requests are rejected

This is a separate CPU-only audit of a frozen prefix of the running 35-table
benchmark. Native source, simulator/coordinator, policy, rewards, limits, splits,
and the 60,000-request budget stay unchanged. No additional ball shots are run.
The derived audit panel is at `http://127.0.0.1:18797/#reach-ik-audit`.

## Verified snapshot

[Compact evidence](../../baselines/isaac-reach-ik-audit-20261106/README.md) covers
7,490 paid TRAIN requests, 214 at each of35 distinct bucket coordinates, from
sweep seed20261121. The earlier 88.9% figure described the 3,850-request prefix;
this later prefix rejects **6,363/7,490 =84.95%**. Its1,127 releases contain491
qualified captures and132 clean captures. These exploratory counts are separate
from frozen validation and final TEST.

| First recorded rejection | Count | All requests |
|---|---:|---:|
| Outside an arm's conservative geometric reach bound |3,449|46.05%|
| Dropper endpoint IK unresolved |1,479|19.75%|
| Board endpoint IK unresolved |1,032|13.78%|
| Clearance proxy |251|3.35%|
| Path/transit |69|0.92%|
| Settling/correction |83|1.11%|

These are disjoint first-failure labels. Geometry, reach, dropper IK, board IK,
clearance, path and native settling run in sequence; later checks may be absent.
In particular1,479 dropper IK failures prevented board IK from being evaluated.
Conservative collision proxies are not exact mesh-collision proofs.

## Reach and target interpretation

The action domain varies a full bucket-relative bearing,280–1,200mm board/bucket
radius,140–600mm board height,80–560mm drop gap and12–52° tilt. The five ranges
are not conditioned on the shared two-arm workspace. Many combined requests are
therefore mechanically impossible even though each scalar is inside its range.

The URDF origin-length outer bound is931.637mm about each arm's first pivot
(Z=364mm in the common workcell frame). A target outside it cannot be reached by
that chain. A target inside it may still violate joint/orientation constraints.
The actual dropper target is the flange, not the ball: ball-center release Z is
board-center Z+gap+37.5mm, and the flange adds95mm. The board target removes the
100mm mount and the board-arm base transform. No frame-conversion error was found
in these calculations. Board IK constrains position and normal, leaving in-plane
spin free; it does not establish the full requested rectangle orientation.

Example `batch00162-train-019`: board center(993.966,−3.881,355.007)mm and gap
424.112mm request a dropper flange at Z=911.619mm. Its distance from the pivot is
1,134.843mm, exceeding the outer bound by203.206mm. The board-arm target is inside
its bound; the pair still cannot perform that requested shot.

Among outer-bound failures, dropper violations have median overshoot228.9mm,
and board-arm violations202.9mm (some requests violate both; do not add those
populations). The reach bound is generous, not a complete feasible volume.

## What IK failure means

The active native controller uses the full robust solver, not the old two-start
survey: up to10 starts for the dropper and12 for the board arm. The original
combined residual gate remains<1e-4. Saved best candidates were independently
recomputed with FK, including positions, normals/orientations and joint limits.
Median failed position errors are48.34mm for the dropper and32.61mm for the board;
median angular errors are3.35° and2.78°. Not every failure is a tolerance-edge case.
Failure of this finite search does not prove no joint solution exists.

A separately preselected10-case CPU recheck used eight failed TRAIN endpoints
(four per arm, spanning near-threshold/large residuals) plus two solved controls.
It used saved-best/home and eight independent Sobol starts, unchanged objectives,
limits and acceptance, capped at10starts/case,1,000nfev/start,180seconds/100,000
residual calls. **0/8 failed endpoints recovered;2/2 controls verified** in2.934s
and23,523 residual calls. These are diagnostic endpoints only: no native motion,
collision-certified path, capture evaluation, learner feedback or relabeling.
Eight examples cannot establish that all IK failures are impossible.

## Sampling diagnosis and next experiment

Global Sobol proposals reject1,490/1,505(99.0%); TRAIN-best refinements reject
3,597/4,480(80.3%). The broad search is spending much of its budget on feasibility,
before ball-bounce performance can be evaluated. Marginal bins also show higher
rejection at larger radius/gap: radius1.0–1.2m rejects294/296; gap.44–.56m rejects
625/644. These correlated descriptive slices are not controlled causal tests.

The current study continues unchanged. A later, separately versioned comparison
could constrain proposals with geometry-only shared-reach checks, preserve diverse
feasible poses and still test genuinely new positions without outcome search.
Do not insert this diagnostic's endpoint candidates into the running learner or
filter its paid denominator retroactively.

## Reproduction and interface provenance

`shot_benchmark_reach_audit.py` accepts a frozen progress snapshot outside the
active experiment, the original run and read-only URDF assets. It checks trial
SHA256 bytes and native geometry-module identities, reconciles every paid request,
recomputes saved reach/FK arithmetic and selects13 median/extreme failed requests.
`shot_benchmark_reach_probe.py` consumes its separate10-case candidate list and
writes a new bounded CPU diagnostic. Neither imports a simulator or GPU learner.

The new UI shows first-gate counts, bucket heatmap, marginal bins, proposal phases,
and selectable top/side reconstructions. Orange marks requested board/targets;
blue marks a saved numerical joint candidate; dashed circles project the outer
reach bound. Measured readbacks remain separately labeled. These diagrams do not
claim a failed target was achieved or draw the full feasible workspace.

`shot_benchmark_reach_panel.py` embeds the frozen audit separately from the
byte-identical live report in a new static staging bundle with source/file hashes.
The existing GET/HEAD reader can serve those updated static files after a browser
reload; no server, simulator or training process restart is needed. Preserve the
original static files and record a distinct UI deployment receipt. Live counters
can advance beyond the clearly labeled frozen7,490-request audit cutoff.

Native source archive, experiment and historical scoring/geometry stay pinned.
Bucket reported-height interpretation and material/attachment assumptions remain
provisional. This is synthetic kinematic evidence, not hardware calibration.
