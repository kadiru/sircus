# Calibrated 650 mm replay — November 4, 2026

The matched replay completed **18 requests / 18 native releases**, with no
invalid physics. The independent CPU audit **passed 5,224 checks**. The candidate
does **not** pass the two-resolution gate, so the conditional gap window did not run.

Candidate: board center **380 mm** above tabletop, tilt **28°** from horizontal,
gap **386.8 mm**, released ball center **804.3 mm**, horizontal board–bucket
distance **650 mm**. Source joints, bucket coordinates and materials are fixed.

| Configuration | 64 substeps | 128 substeps |
|---|---:|---:|
| Matched positive control | 3/3 captures | 3/3 captures |
| Exact 650 mm candidate | 3/3 captures; all rim-assisted | 0/3; no whole-ball entry |
| Matched 750 mm negative control | 0/3 captures | 0/3 captures |

Candidate position difference at 350 ms is **1.734 mm**, below the declared 5 mm
limit, but capture agreement is **0/3**. Both versions contact the rim; the finer
trajectory deflects outside and later contacts the table. Thus a small early
flight difference does not establish full-flight numerical convergence. The64
capture is not a dependable seed across these timesteps. These are three nominal
copies per configuration, not independent randomized robustness tests.

All five predeclared gaps **376.8/380.8/384.8/386.8/388.8 mm** retain explicit
**not run / gate closed** rows. No gap-window throws, retries, refitting, new
policy training or policy activation occurred. Nine total captures include the
six positive controls; that aggregate is not a policy capture rate.

- [Sanitized summary and evidence hashes](summary.json).
- [Exact configuration and outcome CSV](configurations.csv):16 group/resolution
  rows, including ten unexecuted window rows; no joint/trajectory arrays.
- [Export checksums](checksums.json) and [publication provenance](provenance.json).
- [Geometry-only preflight](cpu-preflight.json): eight admitted groups,
  minimum clearance 5.826 mm; predates final source freeze and proves no flights.

Native producer: `56a4ec68bc824e83e82c29c8aec00032f02434df`, immutable Ubuntu
namespace `studies/replay650-v1b`; evidence stays in `output/replay650-v1b`.
The initial startup namespace stopped before any release on a dependency-set
mismatch. No package was changed. Auditor v2 / reader source
`0aaa862bdee6187e4814069b6e6e04600f21406e` corrects a checker-only schema
assumption and records its identity separately from the producer. Original
startup/failure ledgers and native report/manifest bytes remain preserved.

The [read-only interface](http://127.0.0.1:18781/#calibrated650Replay) shows controls,
candidate, skipped window, timestep agreement and selectable actual saved flights.
Its full-audit badge requires exact version, complete checks and byte bindings.
The existing viewer was not restarted; legacy solver routes remain unchanged.

[Frozen experiment and launch contract](../../sircus/mjlab/REPLAY650.md).
Historical shallow-study misses remain under their original scorer/fingerprint.
This is privileged-geometry held-endpoint simulated contact engineering, without
grasp/transit, actuator, perception, physical or new-ball validation. The 175 mm
bucket-height interpretation remains provisional. The 4070 and Isaac setup were
left alone. Next: seek a less rim-dependent central-entry candidate before a
new conditional gap study.
