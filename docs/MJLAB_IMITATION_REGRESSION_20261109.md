# Six corrected-imitation regressions — diagnosis and midpoint test

November 9, 2026. Simulation only, fixed 64 substeps. The historical paired
result remains current imitation **40/70**, corrected imitation **38/70**.
The six losses are inspected development cases, not a new held-out test.
Successful recipes and both model checkpoints remain unchanged.

The [frozen protocol](../sircus/mjlab/IMITATION_REGRESSION.md) and
[literal correction](../sircus/mjlab/imitation-regression-correction.json)
define one action interpolation, not further training or a change to the IK
solver. Draft [PR #113](https://github.com/kadiru/sircus/pull/113) stacks on #112.
Frozen producer: `b962af9f43de2379234b38a4c277283336ebcfca`.
Historical movie/UI publication: `6f489d185d717803756547e1688d048324cd9b62`.

## Diagnosis

The CPU diagnosis authenticates all 158 original requests, identifies all six
paired losses, and independently rescores their nine saved native flights.
Its **1,230 checks** pass. It inherits native validity from the parent's
55,633-check proof. All admitted board poses were stable in these saved traces;
finite endpoint refusal does not establish that a pose is impossible.

| Context suffix | Bucket X/Y mm | Observed corrected-policy failure |
| --- | --- | --- |
| `r0-c0-n0` | 140.398 / −608.427 | No dropper IK branch found; 14 board branches |
| `r0-c1-n0` | 343.413 / −579.043 | No board IK branch found; 8 dropper branches |
| `r0-c6-n1` | 1107.027 / −536.266 | Board at 0.1167 s, robot at 0.1393 s; no entry |
| `r1-c1-n0` | 316.695 / −453.070 | Two bucket/rim contacts, no whole-ball entry |
| `r1-c2-n0` | 496.391 / −414.691 | Robot contact at 0.1468 s, then retained inside; route disallowed |
| `r3-c0-n1` | 214.975 / −70.589 | No board IK branch found; 6 dropper branches |

Full IDs begin `corrected-imitation-test-`. Robot contact is a measured aggregate
channel, not an identified link. Table and rim assistance remain accepted.
The retained robot-contact example is not relabelled a valid capture.

## Frozen correction and accounting

For all six cases, use exactly:
`corrected_action + 0.5 * (current_action - corrected_action)`, where the five
coordinates are board X/Y/height, tilt in degrees, and nominal drop gap.
This jointly changes the shot and derived target-facing yaw/release height;
it does not identify the causal effect of any individual coordinate.

There are **72 prespecified slots**: 54 diagnostics (six cases × exact current,
exact corrected and midpoint × three nominal requests), plus nine original
controls before and nine after. Exact originals retain their admitted joint
branches or their cached finite refusals. Each midpoint gets one fresh strict
setup proof reused across three requests. Nine original refusal slots remain
in the denominator and are not nine new IK searches. New admitted requests run
fresh native ball physics. No result-driven tuning, replacement or retries.

The independent audit reconstructs the plan, action algebra, setup evidence,
realized poses, original branch preservation and all dense native scoring.
Nominal repeats do not qualify noise robustness or other timesteps.

## Execution and evidence

The repository suite passed (1,811 mjlab tests, with dependency-specific skips,
plus existing physical mocks, simulation, Isaac and JavaScript checks). The
pinned Ubuntu environment passed all 45 focused CPU tests and 197 independent
checks of the actual 72-slot plan. Both original proof namespaces hydrated
without allocating a native environment. Source CI passed.

A separate Isaac job acquired the 4090 during preparation. The new experiment
was queued cooperatively at 22:03 UTC; no Isaac process was interrupted.
The queue is limited to 900 seconds. The worker has a 3,600-second acquired
budget, with a 1,800-second CPU audit reserve and 6,300-second pipeline cap.
The queue expired at 22:18:41 UTC after 900.897 seconds: `worker_started=false`,
**0/72 slots executed**. No study report, acquired budget or native audit was
created. This is not a negative shot result. A new bounded launch must use the
unchanged frozen producer when the 4090 becomes free; no automatic retry remains.

The additive `#imitationRegression` panel shows the diagnosis and explicit
not-run status, plus six historical success/failure movies. All 2,705 frames
(90.17 seconds, 1280×720 at 30 fps) decoded successfully. CPU llvmpipe rendering
uses saved native ball centers and held endpoint joints, without new physics or
invented arm movement. IK refusals use no-flight cards. Browser playback was
verified; these are not new correction results. [Compact evidence and video
provenance](../baselines/mjlab-imitation-regression-20261109/README.md).

Final verification preserved all seven inventoried report/diagnosis hashes,
57 display-report keys and original bytes outside the new HTML/CSS panel.
The protected fountain and viewer processes retained their identities. The
foreign Isaac workload was not interrupted; no service restarted. Source CI
passed on `6f489d1`. Midpoint effectiveness remains unknown.

| Evidence | SHA-256 |
| --- | --- |
| Historical comparison report | `1fde3f7de5b5c2912a133c858105388a704a80ab0e2cb39e8d79c0c7a189bf14` |
| CPU diagnosis | `e16f551802c6c378e7911a0bca2470e4302babb075f8ef570df189839a7bd428` |
| CPU diagnosis audit | `84bcab5c1a49a291e3d4f06af49cd31e1285f72772a42be5422b384a9af26d73` |
| Frozen rule | `f764f47023b5e229bb9e4a689088fdeb1fb4368a8c3509d12515f5fea529c96e` |
| Preflight plan digest | `f78d1f42abc653b537ad0df5248092a021464a6d9421875e1f8bfe9d6b6ea624` |

Ubuntu private evidence uses `output/imitation-regression-diagnosis-v1` and
`output/imitation-regression-v1`, with adjacent pipeline/budget/audit receipts.
Raw trajectories, proofs, logs and host configuration are excluded from Git.
The 4070 workload, Isaac setup, running services, historical reports, model
weights and physical hardware are outside this task's mutation scope.
