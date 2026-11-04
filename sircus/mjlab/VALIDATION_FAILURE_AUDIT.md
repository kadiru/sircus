# Validation-only failure audit

This additive CPU diagnostic reads the completed calibrated-v2 benchmark. It
never trains, predicts a new shot, changes a selected checkpoint, runs physics,
or changes an original outcome. The original report, policies, raw observations,
manifests, scorer and independent audit remain frozen. No TRAIN or TEST native
packet is opened. Mixed-role JSONL metadata are filtered before native evidence
is accessed; every packet's record IDs must identify validation requests only.

The original passed audit is bound by exact SHA256, report/contract/manifest
identities and producer/auditor source hashes. Each accessed validation file is
checked against the original manifest and its row-to-packet association. Derived
output must be new and outside the original study. Raw observations and replay
trajectories remain local; Git receives compact summaries and evidence hashes.

## Cohorts and interpretation

The primary view uses the checkpoint selected by validation for each method and
seed:12 shared bucket layouts ×3 repeats ×3 seeds ×2 methods =216 requested
validation trials. The0/256/512/1,024-request checkpoint views are separate,
equally sized cohorts. Their union has864 requests. The selected cohort overlaps
those milestones and must never be added to them. Requests rejected before
release remain in denominators. Repeated trials are not independent layouts.
These validation observations already selected checkpoints; this diagnostic is
not a fresh or unbiased performance evaluation and collects no new shots.

The heatmap shows requested-denominator capture rates at the12 observed bucket
coordinates; it does not interpolate all-table success. Missing cells stay
unknown. Each observed miss has an exclusive outcome category, with separate
setup reasons and secondary contact/trajectory observations. Short/long/lateral
descriptions require a defined observed target-height crossing, not a prediction
from final ball position. Rim contact alone does not establish failure cause.
Representative selection is deterministic and development-only, with exact
source row, validation checkpoint, raw-packet and derived-file identities.

## What pose checks establish

Recorded board centre, normal and quaternion are **initial native reset/FK
observations**, before any settling interval. The ball's recorded initial centre
is compared with the requested ball release centre. Flange error is a separate
measurement; this experiment spawns the ball and does not validate a physical
gripper release. Normal comparisons account for vector normalization.

The engine holds arm endpoints every integration interval. Available scalar
maxima include sampled cached board-body displacement, integrated joint
excursion and velocity. The board displacement is sampled at240Hz and may
describe the step's pre-integration FK cache. There is no time-varying board or
joint pose history, measured board contact point/normal, or ball spin history.
Dynamic settled pose, impact orientation and compliant response are therefore
**not recorded**. Replays hold the recorded initial arm configuration while
displaying the observed ball samples; no arm motion is fabricated.

## Separate read-only interface

`validation_failure_server` extends the read-only calibrated reader. It requires
the completed original independent audit and a CLI-pinned derived report hash.
Original source identities and report bytes are checked on every read. Only
listed, opaque, hash-bound validation replay IDs can be served through
`GET /api/validation-failures/replay/<id>`. All POST commands return405.

Launch a separate loopback process and state directory on18792; preserve18791
and all earlier services. The new panel is `#validationFailureAudit`. Neither
the view nor its replay buttons can run the solver, train, or command hardware.
Use `python -m sircus.mjlab.validation_failure_audit --help` and
`python -m sircus.mjlab.validation_failure_server --help` for required paths and
hash bindings. Use the pinned Ubuntu environment with `CUDA_VISIBLE_DEVICES=""`.

This audit is development diagnosis. If its findings change the method, future
performance claims require a fresh held-out test cohort; the completed benchmark
keeps its original definitions and results.
