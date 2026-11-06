# Diverse-region PPO and measured driven setup

This opt-in follow-up keeps the broad and single-local-box pilots frozen. Native
results are pending the fixed-request setup check and short matched pilot.

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
tanh-Gaussian residuals. Its PPO likelihood is the categorical probability plus
the conditional squashed-Gaussian log density. Entropy is in joint region and
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
