# Fresh held-out geometry and guard integration — no training or shots

The fresh coordinate/candidate set is frozen and independently audited. The
simulator guard adapter is wired and tested with mocks and genuine class
resolution, without initializing or stepping a simulator.

| Preparation result | Count |
|---|---:|
| Fresh validation / TEST positions | 35 / 70 |
| Distinct excluded historical/original positions | 788 |
| Raw proposals / unique preflight checks | 8,505 / 8,505 |
| Geometry-qualified finite candidates | 847 |
| Supported validation / TEST positions | 34 / 35 and 68 / 70 |
| Empty positions retained | 3 |
| New CPU software cases passed / skipped | 39 / 0 |
| Model updates / task shots / diagnostic flights | 0 / 0 / 0 |

The empty supports are `fresh-validation-024`, `fresh-test-024` and
`fresh-test-058`. They stay in the evaluation denominator. No position or
candidate was replaced after qualification, and failure of this finite bank
does not prove a bucket location impossible.

Declined candidates: 6,457 necessary outer-reach bound failures, 611 unresolved
dropper IK, 504 unresolved board IK, 76 board/bucket proxy overlaps and 10
board-arm/table proxy overlaps. All original endpoint, path, settling and
release rules remain unchanged. The original ball-bottom/board-center gap
definition and 80 mm lower bound remain intact.

[Preparation summary](summary.json) records split audit, counts, hashes, runtime
and scoped costs. [Independent bank audit](audit.json) reconstructs all Gaussian
and Sobol proposals and verifies recorded FK/geometry/path gates without new IK
search. [Verification receipt](verification.json) binds the exact 12 new module/
test files, source/input bundle, original file identities, private evidence
mirror, parent before/after hashes and software test logs. Raw positions,
actions, fits, predictions and logs remain outside Git.

Coordinate discovery examined 107 fetched refs/task HEAD and 466 unique baseline
JSON blobs. Only coordinate fields affected inclusion. The 683 discovered sites
plus all 140 original benchmark sites have 35 overlaps, making 788 exclusions.
The fixed fresh seed needed 429 draws; minimum exclusion distance is 25.057 mm
and minimum fresh-pair distance is 25.460 mm. No reachability/outcome filtering
or spacing relaxation selected the positions. Novelty is relative to this
recorded discovery pool, not all unpublished history.

The Ubuntu CPU pipeline took **146.568 s wall / 456.261 aggregate CPU-seconds**,
using at most four workers. Geometry used 102.767 s wall / 412.347 aggregate
CPU-seconds; independent audit used 38.777 s wall / 38.891 aggregate CPU-seconds.
Parent lifetime RSS peaked at 984.8 MiB; the largest individual child lifetime
peak was 1,047.0 MiB. Child peaks are not summed or isolated-phase measurements.
These scopes exclude Mac Git discovery, software checks and transfer. CUDA
stayed hidden; original 30-minute / two-CPU-hour caps and 6 GiB floor held.

All 39 new CPU cases passed without skips, including the unmocked inherited
native method chain, with no application construction. Full project mock/
numerical checks passed: 897 Isaac cases with 121 optional dependency skips;
real tensor and method-resolution coverage is provided by the dedicated CPU
workflow. Existing environments were used without package changes. The new CI
Pillow dependency supports importing the original native class definitions;
it does not enable a simulator.

The [native adapter contract](../../sircus/isaac/CANDIDATE_FRESH_GUARDS.md)
verifies durable reservation/action identities before home/dispatch, pins
original scene/assets/camera/poses and measured guard method resolution, treats
fits only as warm starts, and prohibits duplicate dispatch through new-only
durable claims. Full trace/original reward receipts are retained by its future
explicit caller. No startup probes, native task shots or training ran. Existing
services, hardware and frozen benchmark files stayed untouched.

Geometry admission and mock/class-resolution tests do not certify native
dynamics, mesh/self-collision safety, capture performance or real hardware.
This is geometry-assisted, outcome-held-out preparation. Full native scoring/
guard audit, bounded one-start supervision and final-only assessment remain
execution gates. The previous 13/35 versus 16/35 native comparison stays
unchanged; there is no new capture rate or model promotion.
