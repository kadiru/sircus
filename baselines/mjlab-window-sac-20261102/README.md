# Fixed-bucket gap-window/SAC study

The fixed-bucket comparison completed in29.1 minutes on4090. Independent CPU
audit: **886 checks passed**, rescoring all397 native releases. No active policy
was switched. This is a conditional **gap-only** learning pilot; height/tilt are
inputs and bucket/board XY remain fixed.

| Frozen method | Captures/all requests | Native releases | Invalid | Clean |
|---|---:|---:|---:|---:|
| Supervised success predictor |23/184|69|115|0|
| Terminal SAC |0/184|69|115|0|
| Bounded uniform gaps |0/184|0|184|0|

There are eight held-out height/tilt contexts, each with3 nominal +20 fresh
2 mm gap-noise repetitions. Only3/8 board contexts resolved under the current
IK budget/staging. All23 model captures occurred at one context: board center
**440 mm**, tilt **32°**, gap **285.781 mm**, ball-center release **763.281 mm**,
horizontal board-to-bucket distance **472.826 mm**. It captured3/3 nominal and
20/20 fresh; all were rim-assisted. SAC chose77.396 mm gap there and captured0/23.
The uniform baseline has only eight random base actions; all failed geometry in
this seed. It is not an oracle-feasible uniform sampler. These results do not
establish broad coverage or statistically proven algorithm superiority.

Both learners were charged1,008 training requests:504 shared coarse samples plus
504 exclusive proposals. Total unique TRAIN=1,512; validation=144; final test=552;
quarantined anchor controls=3. Of2,211 requested attempts,397 released natively and
1,814 were rejected; aggregate46 captures includes selected training/control
shots and must not be presented as a policy success rate. Only6/24 TRAIN,2/6 VAL
and3/8 TEST board contexts resolved. Surrogate selection used milestone1,008;
SAC used672. All validation capture counts were0/18, so the prespecified reward
tie-break selected checkpoints. Networks/checkpoint identities were independently
verified; no test outcomes entered training or selection.

## Sampled windows

At tilt36.1548° and distance472.826 mm, the adaptive TRAIN sweep found separated
capture runs. These are sampled points, not certified continuous intervals:

| Board center mm | Observed captured gap samples mm | Points in run | Neighboring tested nonhits mm |
|---:|---:|---:|---:|
|397.388|224.750 only|1|209.000 /240.500|
|397.388|272.000–370.438|11|268.063 /374.375|
|500.000|177.500–181.438|2|173.563 /185.375|
|500.000|209.000–268.063|6|205.063 /272.000|

Misses between these runs show why a single min/max range is misleading. The
92-request follow-up repeated their four centers separately; its engineering
checks did not alter the frozen comparison or train either model.

| Board center mm | Tilt | Tested center gap mm | Ball-center release mm | Nominal | Fresh gap-noise |
|---:|---:|---:|---:|---:|---:|
|397.388|36.1548°|224.750|659.638|3/3|5/20|
|397.388|36.1548°|321.219|756.107|3/3|20/20|
|500.000|36.1548°|179.469|716.969|0/3|5/20|
|500.000|36.1548°|238.531|776.031|3/3|20/20|

[Separate confirmation results](confirmation/report.json) and
[39-check audit](confirmation/audit-window-confirmation.json) rescore all92
releases; all59 captures are rim-assisted. Narrow sampled patches are unreliable
under2 mm gap variation, and the midpoint between two captured samples can miss.
The two wider-run centers plus the held-out predictor case each passed3/3 +20/20
gap-only checks. This does not certify every point in their bands, placement noise,
all reset batches, actuator tracking, contact-model accuracy or physical transfer.
Confirmation source: `~/sircus-mjlab/studies/window-confirm-v2`; raw evidence:
`~/sircus-mjlab/output/window-confirmation-v1`. Both finite jobs completed normally;
4070 and the running viewer were preserved. Display-only source snapshot
`studies/window-display-v1` adds the table and hashed saved TEST trajectory examples.

## Evidence and reproduction

- [Protocol](../../sircus/mjlab/WINDOW_STUDY.md), opt-in version
  `sircus-fixed-bucket-gap-study-v1`.
- Immutable execution source on Ubuntu: `~/sircus-mjlab/studies/window-sac-v2`.
  Evidence: `~/sircus-mjlab/output/window-sac-v1`.
- Source v1 was a CPU-only packaging preflight missing the audit scene fixture;
  v2 includes it and passed all59 new tests on pinned Ubuntu CPU before launch.
  Source v1 never launched native dynamics. No prior experiment was overwritten.
- [CPU preflight](cpu-preflight.json):23 isolated geometry/learning tests passed,
  exact anchor and ±10 mm gaps admitted; both full flange poses verified, fixed
  board branch preserved. This checks geometry, not flight or robot tracking.
- Existing viewer PID186578 remained unchanged. New static panel `#windowStudy`
  and derived-report updates do not change active inference. Finite supervisor
  PID928174 completed normally; identifiers are historical, not current liveness.
- Finite pipeline uses the shared4090/Isaac lock, waits rather than interrupting
  work, leaves4070 untouched, and preserves raw arrays/logs outside Git.

[Curated report](native-results.json), [audit](audit-window.json),
[exact sampled windows](observed-windows.csv), [frozen actions](selected-test-actions.csv)
and [selected checkpoint identities](selected-checkpoints.json) are included.
The JSON predictor and CPU SAC `.pt` retain their original hashes. `manifest.json`
covers the curated package; `curation.json` identifies original report/audit/raw
manifest hashes. Raw NPZ observations, per-shot logs and geometric path frames
remain in the original Ubuntu evidence directory. Full evidence audit requires
that directory, not this deliberately smaller curated package.

The one successful selected test configuration also passed the separate specified
common-home geometric path audit. This is sampled geometry, not dynamic tracking.
The rigid board/contact model is uncalibrated and the bucket's reported175 mm
height interpretation remains unresolved.

Next: expand reachable TRAIN height/tilt neighborhoods around
successes and diagnose feasible misses before more SAC training. Use fresh
validation/test contexts for the next learning study; current evaluated contexts
are now known. No all-table or quasi-perception readiness claim.
