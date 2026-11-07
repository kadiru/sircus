# Consistent-flight selector: completed comparison and residual diagnosis

On the same 30 fresh bucket layouts, velocity/aperture ranking captured **17/30**,
versus imitation 14/30, PPO 14/30 and the old ensemble selector 13/30. These are
single nominal shots per method/layout. They do not establish repeatability,
all-table coverage or an architecture winner.

| Method | Requested | Native releases | Complete captures | Clean | Setup rejects | Native route failures |
|---|---:|---:|---:|---:|---:|---:|
| Imitation | 30 | 26 | 14 | 5 | 4 | 12 |
| PPO | 30 | 26 | 14 | 2 | 4 | 12 |
| Old ensemble rank | 30 | 26 | 13 | 3 | 4 | 13 |
| Consistent-flight rank | 30 | 26 | 17 | 6 | 4 | 9 |

There were zero physics-invalid requests. Compared with imitation/PPO/old
ensemble, consistent ranking gained 3/3/4 captures respectively and lost none
on these particular layouts. The full 132-slot study includes 12 separate
controls, 116 native releases, and **485,047 passed independent native checks**.

## What the residual diagnosis found

Of nine failed native routes, eight never achieved whole-ball entry. Case20
entered and retained for one second but had robot contact, so it failed the
unchanged complete-route rule. First-flight cutoff channels were bucket five,
table three and robot one. Bucket contact alone does not identify a rim hit.

Three setups failed finite IK and one failed strict endpoint clearance/pose
admission; all four checked proposals failed in each rejected layout. Those
finite failures do not prove the geometry physically impossible.

All sixteen candidate aperture margins were negative in each of the nine native
failure layouts. This motivates a candidate-coverage experiment; it does not
prove wider gaps will capture. Negative margins remain a conservative proxy,
not a hard rejection or a success probability.

Case01: board height 274.148 mm, tilt 27.534°, gap 96.365 mm. Its old capture head
scored 92.35%, but it never entered. The direct closest-opening head predicted
13.29 mm versus observed 133.06 mm. A conditional model-velocity curve was
6.31 mm from the observed first-flight endpoint, versus 0.066 mm using measured
velocity. These diagnostics have different meanings and do not rescore capture.

The CPU diagnosis passed **802 separate evidence-binding/reconstruction checks**,
adding zero shots and no training. Its nominal-versus-measured rebound-origin
separation was about 1.03–1.71 mm for the nine native failures. This mismatch is
not established as their cause. The read-only UI at
[#flightResidualDiagnosis](http://127.0.0.1:18781/#flightResidualDiagnosis) shows
bounded native segments and clearly separate conditional forecasts.

## Evidence and next step

- [summary.json](summary.json): all 30 consistent-selector requests as scalar
  configurations/outcomes/diagnostics, paired comparisons and provenance.
- [checksums.json](checksums.json): hashes of the curated files. The summary also
  preserves hashes of the external complete diagnosis, scalar source and derived
  checks, plus the exact original report, full native audit and manifest.
- [Diagnosis method and reproduction](../../sircus/mjlab/FLIGHT_RESIDUAL_DIAGNOSIS.md).
- [Planned gap-coverage comparison](../../sircus/mjlab/GAP_COVERAGE.md): 50 new paired
  layouts, 112 slots including controls, only two gap proposals changed. **No
  new comparison results are published in this baseline.**

Raw trajectories, contact arrays, checkpoints, logs and host paths are excluded.
The previous 30 TEST layouts are now development examples. Fixed64, privileged
geometry and held endpoints remain the scope; no 128/noise, perception, driven
transit, grasping or hardware validation is claimed. The 175 mm bucket-height
interpretation remains provisional. Isaac, the 4070 and the interactive solver
are separate from this diagnosis and the bounded follow-up.
