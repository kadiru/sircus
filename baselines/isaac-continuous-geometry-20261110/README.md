# Continuous geometry layer: successful TRAIN action preservation

The separate, bank-free [continuous geometry layer](../../sircus/isaac/CONTINUOUS_GEOMETRY_LAYER.md)
was checked against every recorded successful paid TRAIN action from original
seed 20261122. All results passed the independent CPU geometry audit.

| Historical population | Unchanged | Adjusted | Rejected |
|---|---:|---:|---:|
| All 1,534 successful TRAIN actions | 1,534 | 0 | 0 |
| Included 485 clean TRAIN actions | 485 | 0 | 0 |

These actions span the original 10,000-request training history and cover 21 of
35 TRAIN positions. They are not 1,534 outputs of the final model and are not new
captures. The original outcome labels identify the test population only; they
do not enter geometry decisions.

The unchanged collector freshly verified all 289 original paid TRAIN packets,
the original passed audit and final checkpoint bytes. Its TRAIN-only export was
byte-identical to the previous export. No held-out shot packet was read. Fresh
robust IK used each request's actual recorded measured starts for both arms;
the checker did not consult a bank or supply the saved successful endpoint
answer. The layer found an admitted solution for every exact original setting,
so none invoked continuous repair.

Analysis took **77.84 s wall**, **225.59 aggregate worker CPU-seconds** and
0.85 coordinator CPU-seconds using three CPU workers. Independent audit took
**55.59 s wall / 54.46 CPU-seconds**. The numerical runtime was Python 3.12.14,
NumPy 2.5.4 and SciPy 1.18.1; original geometry math files were byte-verified
against the frozen producer audit. Runtime differences do not authorize a
cross-runtime numerical bit-equality claim.

All 38 focused checks passed without skips. They include a genuine continuous
numerical repair on synthetic chains, unchanged original final admission gates,
bounded failures, tampered evidence, stale starts, output-root protection and an
end-to-end producer/independent-auditor check. The historical test does not
establish how often repairs succeed on real robot geometry.

Full `scripts/test.sh` also passed: 1,362 Isaac cases (121 optional-dependency
skips), 34 physical mocks, five simulation checks, 59 Node cases and all nine
numerical scripts. The new 38 geometry cases had no skips. Independent source
and evidence reviews passed; the compact summary binds the private full log.

The analysis rechecked 25 protected inputs; a separate before/after inventory
also verified **245 original files unchanged**, including original native source,
models, frozen benchmark artifacts and the existing comparison report. No model
was loaded, no weights updated, no simulator started and no new task shot ran.
No reader or container restarted. Raw coordinates, per-action records and CSV
remain private; [compact counts, bindings and artifact hashes](summary.json)
provide the reviewable evidence.

Admission means the original endpoint, FK, clearance and finite path proxies
passed. It does not certify full mesh/self-collision safety, driven settling or
ball capture. This is a preservation check, not improved policy performance or
generalization. The layer is not yet connected to training or execution; those
remain outside this task's authorization. The frozen benchmark and the separate
V4 hybrid evidence remain unchanged.
