# Contact repeatability and serialized arm transit

This opt-in protocol investigates the saved `stable-shot-0004711` at development
layout `old-test-21`. It does not train, promote or deploy a policy. Historical
results, original physics/scorer source and selected checkpoints remain unchanged.

## What failed

The historical 13/20 confirmation used twenty independently perturbed geometries,
not twenty identical nominal drops. All twenty passed endpoint gates and bounced
once from the board. Seven failed whole-ball entry; one also contacted the table.

Two separate issues were verified:

1. Every arm body already had `gravcomp=1`, but the old native loop also applied
   `qfrc_bias`. At the nominal pose without contacts, this double compensation
   gives the board 9.887 m/s² upward acceleration between repeated pose clamps.
   Removing the extra force leaves CPU joint acceleration near numerical zero.
2. The stiff, lightly damped bounce contact is sensitive to integration resolution.
   Correcting gravity alone does not improve the original coarse-step capture
   count. Smaller time steps produce the strong measured improvement. Capturing
   all trials at two resolutions is outcome agreement, not exact trajectory
   convergence or physical calibration.

`contact_env.py` offers explicit `gravity_mode`, `board_mode` and `substeps`.
The corrected articulated model uses passive gravity compensation only. No ball,
board, bucket, friction, solref, scorer, retention threshold or action is retuned.
The optional kinematic-board experiment changes effective board inertia and is
labeled an idealized diagnostic, not the chosen articulated model or a bug fix.

The first diagnostic snapshot measured a maximum across arm hinges and finger
translations; that mixed-unit field must not be reported as purely radians. The
subsequent confirmation snapshot measures arm hinges only. Old zero board-drift
fields sample cached/preintegration body poses and do not prove zero impact motion.

## Exact frozen nominal configuration

All XYZ values are metres in the dropper-base/table frame, with table top Z=0.
Tilt is from horizontal. Release is the ball center; flange is 95 mm above it.

| Quantity | Value |
|---|---|
| Bucket center XY | (0.16275255472120964, −0.6365356466995716) |
| Board center XYZ | (0.4708757793012148, −0.2778934014509746, 0.3973878458850598) |
| Board tilt | 36.154791124080255° |
| Board-facing heading | Toward the bucket |
| Board-center to released-ball-bottom gap | 0.3032930420811218 |
| Released ball center Z | 0.7381808879661815 |
| Dropper flange Z | 0.8331808879661815 |
| Horizontal board-center to bucket distance | 0.4728257412647234 |

The exact twelve joint values and every tested perturbation belong in the dated
baseline CSV/JSON. No scene-specific retuning is performed after seeing the fresh
confirmation results.

The measured board remains 200×300×5 mm with one landscape mount rotation. Bucket
inner diameters 177/150 mm and outer 200/158 mm remain separate contact/clearance
geometry. Reported height 175 mm remains unresolved; the same provisional 175 mm
interior plus 8 mm bottom interpretation is retained. Ball size/mass/material,
board mass/mount stiffness and table geometry remain provisional.

## Matched diagnostic and fresh confirmation

`contact_study.py` freezes 16 identical nominal plan rows, the exact 20 old final
geometries and 20 new seeded jitters. Nine model/control/time-step combinations
are each run forward and in reversed batch slots:1,008 throws,41 unique requested
geometries. The reversed run is a paired repeat, not another independent jitter.
The old 8-substep rate is 1,920 Hz;16/32 give 3,840/7,680 Hz while scoring stays 240 Hz.

`contact_confirm.py` separately freezes 100 fresh jitters, seed 20261125, plus 16
identical nominal rows. Original 8 and corrected articulated 32/64 each run twice,
then two corrected protocols each get two single-world checks:700 attempts.
The 100 new geometries were not used to select the nominal setting or protocol.
Noise is Gaussian with standard deviations 1 mm XY,2 mm board H/gap and 0.35° tilt.
Geometry rejects must remain explicit; the paired protocol fails rather than
silently excluding a rejected request.

All throws save 240 Hz positions, velocities and categorized contact-force maxima.
The original CPU scorer independently rescores these arrays. Complete success
requires a board rebound, whole-ball entry and one continuous second retained,
with no table/floor/robot/unattributed assistance. Rim-assisted bucket captures
count. Clean captures remain a separate, stricter quality result.

Reset comparisons report pre-bucket flight, common scored prefix and full 4 s
motion separately. Later chaotic motion inside the fixed bucket can differ without changing the
capture verdict. Tiny numerical differences are not evidence that reset state
was forgotten. Bitwise trajectory determinism is not claimed.

## Reachability and transit

`transit_audit.py` uses a separate CPU validation model. It enables non-adjacent
self-collision checks, checks both arms, table, bucket, floor, the full board and
an attached carried-ball sphere. Mechanical adjacent/welded mounts and intentional
finger grasp contacts have explicit exclusions. Every accepted edge is sampled
at at most 0.5° per joint with a 5 mm modeled clearance requirement. AABB/SAT
separation lower bounds handle inconsistent native convex-distance queries;
unresolved pairs fail closed. Bounds are conservative, not always exact distances.

The serialized cycle lifts/retracts the dropper, parks it high, lowers/retracts and
transfers the board, restores the board, then returns the dropper high and descends.
The board's full upper extent is checked in low phases. Exact source joints are
restored. A naive straight 950 mm lift was unreachable; retracting while raising
avoids that request. Fixed Cartesian paths can still fail even with valid endpoints.

`transit_detour.py` tries bounded alternative near-source board goals and seeded
single-arm joint-space detours. A locally valid reconfigure-and-return is labeled
separately from a full 240 mm-low-board cycle. An initially colliding endpoint stays
rejected. No passage through another arm or obstacle is accepted to manufacture
reachability. These are geometric audits, not dynamic tracking, torque, grasp or
formal continuous-collision certification.

## Reproduce on the isolated Ubuntu mjlab environment

Use the existing pinned MuJoCo 3.11/mjlab 1.6 environment and read-only native assets.
Create a new immutable source snapshot and a new output directory for each run.
Expose only the 4090 UUID through `CUDA_VISIBLE_DEVICES`; pass the existing
`scene.lock`. The finite runner waits for other GPU jobs without stopping them.
No command below starts a cloud part, physical dashboard or training job.

```bash
python -m sircus.mjlab.contact_study --prior-study /path/stable-ten-v1 \
  --assets /path/native-assets --output /path/new-diagnostic --gpu-lock /path/scene.lock
python -m sircus.mjlab.contact_confirm --prior-study /path/stable-ten-v1 \
  --assets /path/native-assets --output /path/new-confirmation --gpu-lock /path/scene.lock
CUDA_VISIBLE_DEVICES='' python -m sircus.mjlab.gravity_audit --help
CUDA_VISIBLE_DEVICES='' python -m sircus.mjlab.verify_contact --help
CUDA_VISIBLE_DEVICES='' python -m sircus.mjlab.verify_contact_confirm --help
CUDA_VISIBLE_DEVICES='' python -m sircus.mjlab.transit_audit --help
CUDA_VISIBLE_DEVICES='' python -m sircus.mjlab.transit_detour --help
```

The read-only viewer section is `#contactRepeatability`. It presents contact
protocols, exact configuration CSV and sampled native-FK transit previews. Timing
of the geometric preview is illustrative. Previous inference routes and policy
readiness guards remain unchanged. Do not silently evaluate old checkpoints under
new physics or attach new performance claims to their original version.
