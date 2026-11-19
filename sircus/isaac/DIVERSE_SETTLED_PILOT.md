# Diverse-region PPO and measured driven setup

This completed opt-in follow-up keeps the broad and single-local-box pilots
frozen. The direct setup check, full checkpoint/comparison audit and independent
native telemetry audit passed. It improves setup admission, but does not improve
held-out capture rate. The new policy is not promoted over the broad baseline.

## Completed learning result

All methods used 1,600 attempted training shots and 16 PPO updates. Frozen final
checks use eight nominal repeats at each inherited site, not independent new
layouts or perturbation trials.

| Method | Training captures / attempts | Rejected training requests | Frozen train | Validation | Test |
|---|---:|---:|---:|---:|---:|
| Broad continuous | 62/1,600 | 1,214 | 32/200 | 8/40 | 8/40 |
| One local region | 23/1,600 | 879 | 8/200 | 0/40 | 8/40 |
| Diverse regions + measured settling | 57/1,600 | 736 | 24/200 | 0/40 | 8/40 |

The new policy's frozen train result improves from 0 to 24/200; validation stays
0/40 and test stays 8/40. The three newly scoring sites (`r1c5`, `r2c3`, `r3c2`)
are training sites. Test site `r3c3` remains 8/8 and changes from rim-assisted to
clean entries. That quality change does not establish a held-out capture-rate
gain. Training has four clean captures; final frozen test has eight.

The expanded proposal domain increases reach/IK rejects from 54 in the local
pilot to 389, while native setup/correction failures fall from 679 to 196.
The remaining rejects are 128 unsupported requests and 23 geometry-proxy
failures. Every reject remains in the optimizer and denominator. No capture
used requested bounce radius at least 600 mm; all 461 admitted training shots
in that range missed. This is not proof that those distances are impossible.

The unchanged reward gives **1,513/1,600 (94.56%)** attempts one of two flat
penalties. All 777 external-contact misses receive -0.5 despite closest-opening
distances spanning 2.02–366.72 mm. Thus most misses do not give useful distance
feedback. The [aggregate reward audit](../../docs/audits/2026-11-05-diverse-settled/README.md)
recomputes every training return without changing the benchmark. Of 807 misses,
176 touched the bucket before their first external contact; they cannot all be
described as undershoots. Sixty-eight legacy retained-ball outcomes fail the
required board-route definition and are not counted as captures.

Next: version the miss-feedback design while keeping the capture criterion
fixed, then repeat a bounded pilot. More compute, quasi-perception training and
model promotion are not justified by this result alone.

The run completed normally in 7,944.428 seconds, including setup, recording and
evaluation. This is not a matched headless compute comparison. The final result
SHA256 is `433ed3496ada64b0d6edf626e4d796c25285441b014b649d1d1fdccc34f236f9`.
Checkpoint/comparison audit: `e6df3cac143080cdd99f879fc5eb446ffabfa5078fabf50908035d92925361bd`.
Native audit: `820a6341859a470a93836a7e45f9ca73b602622f37093ff1bd1ad91f13941f12`.
The native audit rescored 1,442 releases across all 84 batches, verified
2,400 recorded release-frame bindings and 704,725 setup windows. Arm travel and
ball release initialization remain scripted; the model learns shot selection.

## Recording and portable evidence

The [curated baseline](../../baselines/isaac-diverse-settled-pilot-20261105/README.md)
contains byte-identical initial/final checkpoints, full signed proposal support,
comparison, learning curves, requested settings and compact source-bound audits.
All 25 selected files passed manifest verification. The final checkpoint loaded
and predicted on Ubuntu CPU with 16 updates/1,600 attempts; a mismatched geometry
fingerprint was rejected. This prediction check is not a new native trial.

The actual Isaac film has 20 prescheduled chapters at the same fixed camera:
27m38.44s full, or 22m42.94s with quiet setup dwells removed. The action edit
retains all 2,400 release frames, uses normal-time playback and keeps visible
scripted arm travel. No outcome selection, ball tracking or generated imagery
was used. Recorded training chapters use policy updates 0/4/8/15; the final
update-16 model is assessed numerically, not falsely presented as filmed.
Full video decode and source/export checks passed. Videos remain outside Git;
`film.json` binds their SHA256 hashes and exact edits.

## Verified setup checkpoint

Producer `40dd24ffdfeec0cd521fdfbf409c31c68c69a2d7` replayed the same 100
saved training requests: admission improved **48 to 80**, with **32 repairs and
zero regressions**. The independent CPU audit checked 33,950 measurement windows,
43 corrections, original holding gains and all eight dropper/six board-arm DOFs.
Remaining failures were eight unsupported requests, two board-arm IK failures,
five moving endpoints and five incomplete transits. All remained withheld.
This check released zero task balls and made zero PPO updates; startup contact
probes are separate. It is not capture-rate or held-out evaluation evidence.

The setup summary SHA256 is
`a49de1e750599953611de58aa6217f3f4649a9a87b4aa7b9321d000212040a20`;
independent audit SHA256 is
`39b88834a3c622cf9f0d970e57b90d261def84143a82e106d4c657d676bacfe5`.
The final pilot uses the same immutable producer/archive, with 1,600 attempted
training shots, 16 updates and the unchanged 25/5/5 split. Raw native traces and
media remain outside Git.

## Two changes, measured separately from learning

The previous guided pilot ended its planned transit after a two-second tail,
even when large joint target error remained. Cartesian feedback could then
compensate unfinished motion rather than steady bias. Across 1,600 training
requests, all 606 not-settled and all 73 correction-IK failures missed the
transit's 0.001 rad readiness target. CPU FK and native board poses agreed within
0.00243 mm, ruling out a board-frame transform error in these traces.

`settled_transit.py` versions a different setup controller. It removes only
FK-verified legal full turns/null wrist spin, freshly checks the requested
endpoint and path proxy, and drives toward the target without an endpoint state
reset. A bounded eight-second tail precedes a one-second ramp back to the
original holding gains/caps. The independent 250 ms stability window can repeat
for up to 40 seconds initially, then eight seconds after each of at most three
feedback iterations (64 seconds total). Moving or grossly unfinished poses stay
pending through those bounded waits, fail closed at the deadline, and never
receive bias feedback. Only quiet Cartesian bias is corrected for each arm.
Every correction is freshly solved, verified and path-proxy checked.

Original release thresholds remain: board position 2 mm, normal 0.5 degrees,
legacy numeric joint-speed limit 0.005, board drift 0.1 mm and normal drift 0.05 degrees; dropper
position must be within 2 mm. Gravity compensation, original impact holding
drives, collision assets, material parameters and the one-second capture scorer
are unchanged. This does not certify full mesh/self-collision safety.

Controller v5 restores the historical velocity gate over **every articulation
DOF**, including both simulated gripper fingers. Revolute values are rad/s and
prismatic finger values are native m/s; the legacy numeric gate is preserved
without a new conversion. All DOF names and velocities are logged separately
from six-axis FK/lag state. Active finger targets retain their configured open
defaults; inactive sites hold their full actual pose.

Before training, replay the first four saved training-request batches from the
immutable guided run as a setup-only check. Do not release task shots or use any
held-out outcome to select these requests. Save every target, measured state,
gain transition, stability window and correction. Compare setup admission, not
capture performance, against their original saved outcomes. Standard startup
contact/rim diagnostic probes are separate from these setup attempts and PPO.

The diagnostic 100-request check with eight-second windows admitted 62 versus
48 originally, but 26 targets still had measurable board drift. Traces localized
the slow convergence predominantly to board-base yaw; reported joint velocity
understated the observed finite-difference motion. The longer bounded window
preserves the original impact holding gains and independent drift guard. Rerun
the same fixed requests to verify it before the PPO pilot. This is controller
calibration from setup-only traces, not tuning on ball capture or held-out scores.

The diagnostic 40-second initial-wait check recovered 80/100 requests with no regressions.
Its remaining quiet/gross-lag cases revealed an early-exit bug: quiet motion alone
could return before using the configured wait budget. Controller v4 requires
quiet motion **and** lag at most 0.02 rad before bias feedback; a release-ready
pose can still finish immediately after the minimum measurement hold. These
conditions preserve the original release gates and avoid premature timeouts.

**Superseded prototypes:** versions 2–4 checked only six robot-axis velocities,
omitting the finger DOFs checked by the historical implementation. Their setup
counts describe diagnostics under that incomplete integration and are not an
authoritative comparison at the original full gate. The first native pilot
(`diverse-settled-pilot-20261105-v1`, producer `7c1f5d9`) was stopped during
coverage/initial checks, before any training batch or PPO update, when this was
found. Its partial data stays separate and must not be used as learning or
held-out benchmark evidence. Rerun the same fixed setup requests with v5 before
a fresh pilot; do not reuse the aborted initial checkpoint or samples.

The final controller also preserves the inherited home-eligibility and exact
recomputed 280 mm minimum board-to-bucket separation guards. A separate bounded
source-applicability receipt may connect the measured v5 preflight to that
guard-only restoration: it must verify the complete controller AST, unchanged
runtime files, original archived source, all 100 saved requests and their actual
recomputed distances. Home eligibility is a source invariant in this preflight,
not a newly observed flag. Such a receipt is not a resimulation, a new producer
measurement or a feasibility certificate for future actions. Each newly sampled
PPO action still passes the full native admission checks.

The minimum-distance check retains an earlier rejection cause instead of
overwriting it. Every otherwise eligible request below the exact limit is still
rejected. This changes reason precedence for already-invalid requests, not
admission or motion. Ordinary unresolved PPO accounting uses an in-bounds
global midpoint, not a zero-radius physical shot; the regression also covers
under-limit inactive, unsupported, home, IK and motion-failure cases.

The new proposal support retains **every** geometry-map anchor passing the
original canonical-home path proxy, rather than selecting one anchor and
shrinking its box. Fixed residual half-widths are 7.5 degrees bearing, 40 mm
radius, 40 mm board height, 80 mm gap and 8 degrees tilt. Bounds are intersected
with the original global domain during preparation. No successful-shot labels,
rewards, outcome-based pruning, runtime projection or rejected-sample retries
choose the regions. An anchor's endpoint/path qualification does not certify
its box interior; every sampled request still receives native checks.

Each of the 33 locally supported contexts also gets one explicitly unqualified
**global-exploration escape** spanning the entire original action domain. Its
first mapped anchor supplies an IK warm start only. This preserves all original
in-bound shot possibilities at these contexts without using any previous
capture outcomes to select them. The actor can learn to avoid the escape's
reach/IK penalties. The two contexts without a map anchor remain unresolved
and their attempted requests are still counted as rejects. There are 153 local
regions and 33 global domains, not 186 certified feasible regions. Neither local
nor global interiors have a feasibility certificate.

## Policy and comparison

The fresh hybrid actor selects a categorical region and five continuous
tanh-Gaussian residuals. Its joint PPO log likelihood adds the categorical log
probability to the conditional squashed-Gaussian log density. Entropy is in joint region and
normalized-residual coordinates, not physical shot-volume entropy. Deterministic
inference selects the highest-probability region then that region's residual
mean; it is not a mixture-averaged shot. At clipped domain edges, zero residual
means the box midpoint, which can differ from its original anchor.

Use the unchanged seed, 25/5/5 bucket split, 64 batches × 25 attempted training
requests and 16 PPO updates. Initial/final frozen checks use eight nominal
repeats per site. All active failures—including unsupported contexts—remain in
the optimizer and denominator. Geometry coverage is separate from training.

The actor has privileged known bucket geometry. Held-out rewards stay outside
optimization, but held-out geometry assists support construction. These inherited
coordinates are not globally unseen layouts; repeated identical nominal shots
are not independent robustness trials. Proposal support, policy family and setup
controller change together. Historical comparisons do not isolate PPO or prove
that any improvement is caused by learning alone.

## Recording and isolation

Keep the user's exact saved camera, silver-gray stage, 35 distinct bucket
coordinates, source geometry and provisional bucket-height interpretation.
Preschedule early/middle/late chapters; retain driven setup and complete release
frames. Only declared motionless setup dwells may be cut. Do not follow balls,
select shots by outcome, or call scripted arm travel learned joint control.

Native jobs use only the existing 4090-assigned Isaac container and shared scene
lock. The read-only comparison viewer uses a new port, 18796. Existing reports,
readers, 4070 containers, hardware, Viam and audio are independent. Raw physics,
RGB, per-joint histories, runtime configuration and logs stay outside Git.
