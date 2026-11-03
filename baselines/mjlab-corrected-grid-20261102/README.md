# Corrected-contact spatial study — November 2, 2026

Synthetic mjlab/MuJoCo Warp feasibility map before another learning run. No PPO,
checkpoint replacement, perception, hardware, Viam or audio work. The 4090 job
respects the existing scene lock; Isaac and4070 workloads were not signalled.

## Verified CPU preparation

| Initial stage | Result |
|---|---:|
| Development bucket positions |35 (7×5)|
| Geometry requests |11,760 (336 per position)|
| Strict admitted endpoint pairs |460|
| Positions with an admitted endpoint |27/35|
| Positions clear at the fixed common home |29/35|

[CPU admission](cpu-admission.json) comes from the immutable audited proposal
plan and initial home checks, with no native capture/learning results. Bucket
centers span X100–1240mm/Y−650–250mm. This is a finite proposal count, not the
fraction of the continuous workspace covered.

The common home has dropper flange(250,−100,950)mm and board center
(670,−220,300)mm; exact joints, transforms and hash are recorded. Four bucket
layouts obstruct this posture, not necessarily every possible posture.
Two near-base layouts fail initial clearance; a supplementary
[convex collision-hull proof](base-overlap.json), reproduced by
[this CPU script](base_overlap_probe.py), verifies fixed-base overlap at exactly
(100,25) and(1240,25)mm in this model. Its positive intersection-ball radius
is not a signed penetration depth. Original conservative categories are preserved.

## Native stage and qualification

All 460 admitted requests are selected for fresh native search. At each position,
up to the top two new captured sources must pass a sampled serialized connection
from the frozen home to the exact joint pair and back. Each then receives three
nominal repeats and twenty independently seeded perturbed requests.
Qualification requires 3/3 nominal and≥18/20 fresh captures; strict geometry/local
path admission is required for each counted success. Rejections stay failures
in the twenty-request denominator.

At this code checkpoint the native study is running; no final capture rate,
coverage or model improvement is claimed. `#correctedGrid` shows live progress
and audit status. The finite pipeline automatically rescores saved native data
on CPU; a separate supplemental audit checks all selected sources and original
home classifications. Raw arrays/logs and immutable snapshots remain on Ubuntu.

Corrected physics uses an articulated board, passive gravity compensation only,
64 substeps per 240 Hz scorer(15,360Hz), unchanged material parameters and the
existing board-rebound/whole-ball-entry/continuous-one-second-retention score.
Rim-assisted captures count; externally assisted routes fail. Historical results
retain their original contracts. Arms remain at prescribed endpoints during
ball flight; sampled geometric paths do not validate actuator dynamics.

## Future learning boundary

The contract freezes 35 future training coordinates(this development grid),
30 validation and100 test coordinates before current native outcomes. Future
validation/test exclude the supplied historical mjlab coordinates by 5 mm; this
is not a claim against every experiment ever run. They receive no outcome search
here. Future policies must use matching corrected physics/admission, compare
against uniform feasible selection, and keep rejected test layouts in denominators.

Board300×200×5mm and tapered bucket measurements are preserved. Reported175mm
bucket height still has unconfirmed meaning. The provisional175mm interior
depth plus8mm bottom remains; ball/material/mount properties are uncalibrated.
See [protocol and launch commands](../../sircus/mjlab/CORRECTED_GRID.md).
