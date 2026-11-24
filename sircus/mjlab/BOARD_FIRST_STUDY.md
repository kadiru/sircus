# Board-first capture: saved evidence and frozen selector comparison

The task is to place a ball in a bucket after its first bounce on the held
board. After that board bounce, table, robot, bucket, floor and other contacts
are allowed. A capture still requires a downward whole-ball entry and one
continuous second fully inside the bucket. Invalid simulated physics is not
a capture.

This is a new, explicit scoring version. Original labels, traces, calibration,
successful shot settings and model checkpoints remain unchanged. Contact
channels establish measured order; they do not by themselves prove that each
touch caused a bounce. Ambiguous first contacts are reported separately.

## 1. Re-score saved trajectories

Read the recorded positions, velocities and contact forces. Authenticate each
trace and its original scene/calibration, retain its original score, and append
the new score. Count a saved native trajectory once even if it was copied into
multiple reports. Report newly accepted captures and lost captures separately,
by bucket position. Missing or incompatible evidence remains unscored.

Re-scoring changes the interpretation of an existing simulation; it creates no
new shots and supplies no evidence beyond the saved trace's stopping point.

## 2. Fit the same learned shot selector

Use the original neural-selector study's TRAIN and validation positions, with
corrected capture labels. TEST positions do not enter training. Keep the
44-input, 64/32-hidden-unit, two-output network and its training settings so the
comparison isolates the label change. Each bucket position has equal total
weight in each supervised loss. The outputs estimate setup feasibility and
capture probability given a feasible setup; their product ranks proposed shots.

Keep the original checkpoint frozen. Select the new checkpoint using only
validation loss, then freeze its weights before preparing evaluation shots.

## 3. Compare on 50 new bucket positions

Both frozen selectors receive the same candidate construction and arm setup
checks. Both use the opt-in ball-contact-permitting flight ranking rule. That
rule retains predicted arm clearance as a diagnostic, without treating a later
ball/robot contact as automatically disqualifying. It does not predict arbitrary
robot-assisted bounce paths. Arm/arm and arm/environment setup/path collision
checks remain unchanged.

Use 50 positions excluded from all inventoried training, validation and previous
evaluation positions. Freeze both methods' choices before any test drops. Each
method gets one attempt at each position; a setup refusal stays in the denominator.
Run native MuJoCo at 64 substeps, with unchanged physical parameters and control
shots before and after. Report captures out of 50, setup refusals, released misses,
invalid physics, paired gains/losses and planning time.

The arms are held at checked endpoints during these native drops. This does not
validate dynamic arm motion, camera perception or physical robot deployment.

## Resource and publication boundary

Use the RTX 4090 only, under the existing cooperative GPU lease. Do not interrupt
another GPU task. Leave the 4070 workload and Isaac installation alone. CPU-only
rescoring and fitting do not need the GPU. Publish derived, audited results in
the existing interface without restarting its service or switching its planner.
Raw trajectories and machine-specific configuration stay outside Git.
