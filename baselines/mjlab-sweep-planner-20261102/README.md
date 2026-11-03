# Sweep-planner verification — November 2, 2026

Synthetic development checks of the independently usable, corrected-contact
shot planner. No RL policy was trained or evaluated. These two requests were
chosen near different existing qualified grid sites and optimized by search;
they are not held-out generalization evidence or continuous-table coverage.

| New bucket XY, mm | Native search captures / shots | Nominal | Fresh variations | Board H, mm | Tilt | Gap, mm | Ball-center release Z, mm |
|---|---:|---:|---:|---:|---:|---:|---:|
|490, -640|1/5|3/3|20/20|256.69|37.219 degrees|194.84|489.03|
|300, 240|5/7|3/3|20/20|340.11|32.128 degrees|311.55|689.16|

The first displayed nominal capture is clean; the second is rim-assisted. Both
require the unchanged board rebound, whole-ball entry and one continuous second
inside, with no detected external assistance. Nominal/fresh outcomes selected
after search must not be pooled into a policy-performance statistic.

Each query checked64 development proposals plus20 fresh endpoint variations.
Search endpoint admissions were5/64 and7/64. Nominal replay used the exact
selected joint branches. Both common-home paths passed sampled geometric
admission. End-to-end new-query times were172.27s and180.09s, including startup,
geometry, native contact, confirmation and path checks; these are not kernel
benchmarks or a guaranteed interactive latency.

- [Case1 selected result](result-01.json), [independent audit](audit-01.json).
- [Case2 selected result](result-02.json), [independent audit](audit-02.json).
- [All12 published cache positions and one blocked-home check](cache-checks.json).

All12 exact saved grid recipes passed source/asset/trajectory/path identity
checks without a GPU or new physics trial. The previously checked bucket at
(100,25)mm fails clearance for the prescribed common home. Another staging
strategy or a larger search is outside this check; this is not an impossibility
claim.

## Independent verification

Case1:919 checks, zero failures. Case2:986 checks, zero failures. All58 native
records were independently rescored from saved240Hz positions, velocities and
categorized forces. Native arrays, row IDs, manifests,28 source hashes and10
asset hashes matched. Maximum numerical differences were3.47e-18m in opening
distance and1.14e-13mm in mouth clearance; capture/count fields matched exactly.

The original worker saves search geometry but not the20 fresh endpoint/local
path proof records. The supplementary audits explicitly preserve that evidence
gap and independently recompute all40 exact recorded fresh branches' strict5mm
geometry and local paths under frozen source. All passed. These are additional
CPU checks, not repeated physics rollouts. Raw original bundles were unchanged;
the exported audits retain their original result/audit hashes and source bindings.

The prescribed-endpoint native model resets arm joints before every substep.
It does not establish real arm compliance, grasping or driven-path dynamics.
Joint path proof is sampled geometry. Balls, contact properties and mount remain
provisional. The reported175mm bucket-height ambiguity is unchanged.

## Runtime/source identity and interface

Native execution used immutable snapshot `studies/sweep-planner-v2`, archive SHA256
`35061a888a330c0125a599e15d139650ce531ca42c7d845468966d9f0ede5dc9`.
The final standalone UI/adapter snapshot is `studies/sweep-planner-v3`, archive SHA256
`4f67811debe8116b1671f62fb15f8f7e753569d7d56f1360e6202aca85734bb9`.
Its native worker, proposal contract and28 verified source dependencies are
identical to v2. The adapter scales CPU proposals with the selected search budget.
The independent audit hashes are recorded inside the curated audit exports.

Live interface: loopback18786 on Ubuntu, forwarded to the same Mac port. It
provides a single interactive workcell and a35-target selector. Cached and newly
searched results are labeled separately; policy inference remains gated. Both
new requested positions replay through the HTTP/UI from their verified caches
with budget16. This is a native-data schematic replay, not an Isaac viewport.

The matching [planner guide](../../sircus/mjlab/SWEEP_PLANNER.md) documents source,
generated asset/reference requirements, budgets, failure states and launch.
Raw native arrays, full paths, runtime configs, caches, logs and screenshots
remain outside Git. The4090 lock was released after finite verification; the
4070 workload, original simulation services, Viam, audio and hardware were not
changed. This task added only its separately owned loopback planner service.

Repository mock/numerical regression passed:420 mjlab tests(41 expected local
dependency skips),128 Isaac tests(one expected skip),82 JS checks and the existing
physical/mock, numerical and scoring checks. New planner contract/worker tests
passed23/23, and mode-specific API/standalone UI checks passed. Native rollout
verification is the separately described58-shot development check.
