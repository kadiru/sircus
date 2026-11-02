# Stable IK and the ten remaining bucket positions

The query-order instability is fixed in a new opt-in solver. The ten-target
native search is complete, but **no new setting passed final reliability
qualification**. No policy was trained or deployed. Original results, physics,
scoring, and the existing 17-template prefix remain unchanged.

## Endpoint validation

The checked audit contains 16 saved-source rows, including repeated requests.
The old solver's random start and batch-wide early stopping made a request's
answer depend on its neighbors. The new solver uses per-target hashed starts,
independent branch convergence, explicit source continuation, and native
FK/limit/collision checks of alternative joint pairs.

| Mode | Endpoint-valid rows /16 | Order-mask changes | Batch-mask changes |
|---|---:|---:|---:|
| Original cold 4 starts /90 iterations |14|3|2|
| Stable cold 4 /90 |12|0|0|
| Stable cold 8 /120 |12|0|0|
| Stable with exact source provenance |16|0|0|

All 16 known nominal joint pairs were preserved exactly. Stable joint differences
under reversal/single-query partitions were at most 1.8e-15 rad, below the declared
1e-9 tolerance. Determinism does not imply that a small cold-start budget finds
every solution. The new matched-budget cold solver did not improve this audit's
endpoint count; it made the answer reproducible. No failure is called physically
unreachable.

A separate CPU-only budget diagnostic found:

| Cold budget | Endpoint-valid rows /16 | Warm whole-call median |
|---|---:|---:|
|8 starts /120 iterations|12|0.312s|
|16 /180|16|0.530s|
|32 /240|16|0.906s|

Increasing iterations alone to 240 with eight starts still gave 12/16. An
exploratory 16/120 call already recovered 16/16. The four rejected rows represented
two unique requests repeated twice; this is not a broad workspace benchmark.
All 44 accepted endpoints across the three primary budgets passed independent
URDF/native checks. For a future cold caller, use the explicitly configured
16-start budget, or a validated exact-source cache when available; nearby hints
alone were not a reliable substitute. Current native study defaults remain
recorded as 8/120 rather than being changed after execution.

Exact nominal reuse requires matching original bucket/action values and strict
original IK/native validity. Changed requests are refined toward the new target
with 1e-8 internal polishing; original 1e-4 acceptance is unchanged. A20µm jitter
test verifies that the old pose is not silently reused. See
[implementation and protocol](../../sircus/mjlab/IK_STABILITY.md).

## Fresh native search

Each remaining coordinate received 1,024 endpoint-qualified native shots using
the stable solver, source-branch continuation and the existing wider action
envelope. Source success labels never become query outcomes. Proposal ranking
now favors direct entry/retention and opening proximity **before external
contact**, rather than final table-assisted entries.

| Original index, now DEV | Direct captures /1024 | Best screening | Independent final |
|---|---:|---:|---:|
|8|0|—|—|
|9|0|—|—|
|10|209|6/10|Not advanced|
|12|0|—|—|
|21|405|8/10|13/20 — failed|
|23|49|3/10|Not advanced|
|26|176|4/10|Not advanced|
|35|0|—|—|
|39|46|2/10|Not advanced|
|41|0|—|—|

Totals:10,240 search throws,400 screening attempts,20 final attempts and 90
native-control attempts;28,580 geometry proposals. Screening includes 34 geometry
rejects in the denominator. A source needed a native capture, at least 8/10
screening captures, then at least 16/20 separately seeded final captures.
Only target 21 advanced to final confirmation, and it failed that threshold.
No source was promoted and no failed trial was removed.

Five positions had observed direct captures; five had none in this finite
search. Position 23 is a new direct-route hit after the earlier deeper search
found none. These are development findings, not held-out policy performance or
proof that the remaining positions are impossible. Selected candidate details
are in `candidate-catalogue.json`, explicitly labelled **unqualified**.

All 20 final target 21 endpoints were valid and all balls rebounded from the
board. Its seven failed throws did not achieve whole-ball entry; one also had
disallowed external contact. Those failures are ball-flight/entry failures,
not IK readback failures. The rigid held-endpoint contact model and its response
to small release/tilt changes need separate investigation before more PPO.

## Native controls and repeatability

Six captured source configurations were repeated three times per mode:

| Control | Attempts | Released | Captured |
|---|---:|---:|---:|
|Saved exact joints|18|18|18|
|Original fresh IK|18|18|15|
|Stable matched cold 4/90|18|12|12|
|Stable production cold 8/120|18|12|12|
|Stable source-seeded|18|18|16|

Identical saved/source-seeded joints did not always yield identical capture
counts across separate resets. Endpoint reproducibility and numerical contact/
retention repeatability remain separate. These small controls do not establish
a capture advantage, real-world reliability, or a policy-performance result.

## Verification and runtime state

The independent CPU evidence audit passed 156 checks: source/artifact identities,
the exact ten coordinates, permutation/partition arrays, source-joint preservation,
strict original full-pose residuals, native joint/collision/collider gates,
hint/request provenance, native-release counts, both confirmation stages and
the preserved template prefix. It checks records and independent FK; it does
not independently replay dynamics or certify arm transit.

New source lives in `stable_ik.py`, `stable_geometry.py`, `stable_study.py` and
`verify_stable.py`. The original solver/environment/model/scorers remain byte
unchanged. The selected policy checksum remains
`768d21bdbe104f64c87eeec96bda81395f9ffea9af14f211baa53c70f7a898bd`.

Native source/evidence: `~/sircus-mjlab/studies/stable-ten-v1` and
`~/sircus-mjlab/output/stable-ten-v1`. CPU budget evidence is the sibling
`stable-ten-cold-budget-v1`. The independent auditor uses source snapshot
`stable-audit-tools-v1` and verifies against the exact native snapshot. The
display helper was subsequently extended to show failed confirmation results;
that does not alter native code or evidence.

The existing localhost preview's `#ikStability` panel shows these findings.
Its original PID/start time and all prior results remain unchanged. Existing
solve controls retain the previous policy; this validated solver/catalogue is
not silently substituted into an older checkpoint contract.

Only the 4090 ran new native work through the shared lock. No 4070, Isaac setup,
hardware, Viam, audio or cloud changes accompanied this task. Raw native trials,
force/pose arrays, generated configs and logs remain outside Git. Bucket
diameters, landscape board and the reported 175 mm height uncertainty are
preserved. `protocol.json` is a portable excerpt; its original contract
fingerprint and host-local directory mappings remain in runtime evidence.
