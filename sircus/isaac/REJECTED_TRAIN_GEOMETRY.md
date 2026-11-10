# Frozen geometry layer on rejected seed-22 TRAIN proposals

This is an offline CPU check of all **6,344 rejected paid TRAIN proposals** from
the original seed-22 run. The layer, solver, bounds and admission rules from the
[successful-action preservation check](CONTINUOUS_GEOMETRY_LAYER.md) are frozen.
No model is loaded or updated, no simulator or new task shot starts, and no
candidate library or held-out shot outcome is used. The completed benchmark,
its sources and checkpoints, prior geometry evidence and V4 evidence stay
unchanged.

The [completed independent audit](../../baselines/isaac-rejected-geometry-20261110/README.md)
found **4,175 geometrically repaired, 348 unchanged and 1,821 unresolved**.
All unchanged cases are original native movement/settling declines. Remaining
causes were 1,522 unresolved arm IK cases, 291 clearance/path proxy failures and
eight necessary outer-bound failures. These are geometry results, not new
captures. All 38 focused cases and the full repository suite passed.

## Population and procedure

The new standard-library `rejected_train_evidence` collector first runs the
unchanged original TRAIN collector, verifying all 289 paid TRAIN packets,
10,000 charged slots, the original independent passed audit and final checkpoint
bytes. It then extracts every enabled request with a non-null original rejection
label. No released miss, capture, unpaid padding or held-out request is added to
the rejection population. Raw actions, original classifications, measured arm
starts and qualification/path provenance remain private.

There are 5,996 original geometry/path declines and 348 original movement or
settling declines. For the latter, original endpoint qualification, path proxy
and home eligibility all passed. These native failures must remain visible;
another offline geometry pass does not establish that movement is fixed.

Every proposal receives the exact existing `filter_shot` defaults. Fresh robust
IK starts from that request's recorded measured arm positions, without a saved
endpoint-answer lookup. An admitted continuous setting stays unchanged. A
declined setting receives one local continuous action/joint least-squares
attempt, followed by the original full admission gates. Its fixed limits are
24 optimizer function evaluations, 512 residual evaluations and a maximum 10%
of the original span in each action component. The latter permits at most 36
degrees bearing, 92 mm radius, 46 mm board height, 48 mm release gap and four
degrees tilt, further clipped by original action bounds. These are limits, not
required adjustment sizes or a globally nearest-solution guarantee.

The release-gap definition remains ball-bottom release Z minus board-center Z,
with the original 80 mm minimum and separate tilted-board clearance check.
There is no ball-flight search, reward ranking or evaluation-driven tuning.

## Reporting and independent audit

The output separates:

- **Unchanged:** the original five settings pass the fresh geometric check.
- **Adjusted:** a different continuous setting passes all original geometric
  gates after the bounded repair.
- **Rejected:** no geometrically admitted output was found within that fixed
  procedure. The final gate cause, original cause and optimizer termination are
  retained independently.

Counts are joined to original reasons, including the separate native movement/
settling group. Neither unchanged nor adjusted means a native release or a
capture succeeded. Actions span the original training history, rather than
being predictions sampled from the final model.

Each mathematical result is saved durably under its original request identity.
A source/input/protocol-bound manifest and writer lock allow interrupted **CPU
analysis only** to reuse independently checked records without recomputing them.
This does not reopen a simulation run or change paid-request accounting. The
timing sample uses the first original request for each rejection reason; those
records stay in the same full analysis. Completion requires all 6,344 cases and
the independent `rejected_geometry_audit` reader. Partial timing samples never
receive a complete-result status.

The reader recomputes original FK, geometry, endpoint clearance and sampled
paths from each finite answer, checks exact identities/defaults/classifications,
and verifies protected source/input hashes. It performs no new IK search. Unit
tests use explicitly synthetic inputs and cannot issue a real historical pass.

## Limits

These remain inherited capsule/enclosing-volume and 16-sample path proxies,
not full mesh/self-collision or native-motion certificates. Unresolved IK or a
failed local repair is not proof that a bucket position is impossible. Runtime
versions and CPU costs are reported; fresh numerical solving may differ from
the original producer's branch choices. The result can measure geometrical
repair coverage, not improved capture, generalization or training efficiency.

No training, native evaluation, model promotion, reader/container restart or
new shot is authorized by this analysis.
