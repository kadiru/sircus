# Ensemble flight diagnosis — original fixed64 recordings

The [diagnostic report](../../baselines/mjlab-ensemble-flight-diagnosis-20261106/README.md)
explains high model scores that failed in the completed comparison. This is CPU
analysis of the original recorded outcomes, not another training run or a new
native evaluation. The existing study's scores and fingerprints remain unchanged.

## What the evidence shows

The ensemble made 60 requested decisions: 49 native releases, 24 captures,
25 native flight misses and 11 setup rejects. All 25 flight misses had a board
bounce but no whole-ball entry and zero retention. Their first post-board free
flight ended at bucket contact in 16 cases, robot contact in three and table
contact in six. A bucket contact channel alone does not distinguish rim from
wall contact, and a missing first-flight rim crossing is an unknown crossing,
not independent proof of the final outcome. Full recorded scoring supplies the
no-entry labels.

Of 36 requests with model scores at least 90%, only 18 captured; 13 were native
flight misses and five were setup rejects. Of 13 scored at least 99%, four
captured. These uncalibrated scores are not measured success probabilities.
Five bootstrap members can agree while sharing the same error.

## Concrete examples

| Original TEST case | Board height | Tilt | Drop gap | Model score | Observed |
|---|---:|---:|---:|---:|---|
| 29 | 268.099 mm | 31.520° | 261.560 mm | 99.654% | No entry; bucket-contact cutoff |
| 49 | 262.108 mm | 33.125° | 264.839 mm | 99.495% | No entry; bucket-contact cutoff |
| 36 | 413.012 mm | 31.014° | 184.645 mm | 98.752% | No entry; BC/PPO references captured |
| 35 | 284.167 mm | 23.896° | 193.984 mm | 99.312% | Admitted but missed; BC/PPO setups rejected |

Case 29's opening-distance head predicted 8.4 mm, but the recorded first free
flight came no closer than 107.9 mm to the opening center. Its modeled rebound
velocity was (1.4632, 0.0086, 0.4604) m/s versus measured
(1.4488, 0.0047, 0.5426) m/s in the bucket-forward/lateral/vertical basis.
The board position gate error was approximately 0.00022 mm, normal error 0°,
recorded board drift zero and bucket-collider placement error zero. These
observations do not support a large endpoint-pose error as this case's cause.

Case 36 is an exact one-coordinate contrast: its board XY, height and tilt
match successful BC, while drop gap increased from 148.645 to 184.645 mm.
The original BC shot captured and the ensemble shot missed. This is one nominal
sample per action, not a repeated parameter window or a new intervention.

## Distinguish the three predictions

- The **opening-distance head** directly predicts the closest 3D distance during
  the contact-free first rebound flight. It is not signed rim clearance.
- The **model-velocity ballistic curve** uses the observed rebound position,
  model velocity and gravity, with no later contacts. It is a conditional
  diagnostic, not a complete learned rollout or a bucket-collision prediction.
- The **observed-velocity ballistic curve** uses the same observed origin and
  measured velocity. It checks the free-flight calculation against the saved
  native segment. It also stops at the first measured later contact.

In case 29 the model-velocity curve's endpoint error is about 16.97 mm, while
the observed-velocity check is about 0.07 mm. The direct distance head's
8.4-versus-107.9 mm error is a separate quantity. None of these curves alone
predicts subsequent rim contact, containment or one-second retention.

The frozen model independently supervises capture, distance and velocity heads;
its ranking does not enforce consistency between those outputs and ballistic
motion. This is an implementation limitation and a plausible improvement
direction, not proof that it alone caused every miss. Normalized nearest-TRAIN
distance is descriptive support information, not a confidence guarantee.

## Reproduce and view

Use the pinned mjlab Python with NumPy and a new output directory. No CUDA,
robot credentials or running simulator are required:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  /path/to/pinned-mjlab-python -m sircus.mjlab.ensemble_flight_diagnosis \
  --study /path/to/immutable/physics-ensemble-v1 \
  --audit /path/to/physics-ensemble-v1-audit.json \
  --output /path/to/new/ensemble-flight-diagnosis
```

The producer verifies the original completed report, full native audit, bound
records, target files and consumed dense cache bytes before writing a separate
diagnosis, scalar summary and reconstruction checks. Raw traces remain local.
The additive read-only UI at `#ensembleFlightDiagnosis` shows six examples,
predicted/recorded metrics, conditional curves and all 25 native misses. It does
not activate a solver, change the policy or restart the viewer.

The 4090 was occupied by an unrelated workload when inspected; it was not
interrupted. No new GPU requests, training, Isaac/4070 changes, noise study or
128-substep tests accompany this diagnosis. The reported 175 mm bucket-height
interpretation remains unresolved.

## Next bounded step

Compare a physics-consistent flight/aperture ranking with the independent
capture head on these development cases before another fit. Keep geometric
admission, flight prediction and measured capture separate. Any selector tuned
using these cases must be evaluated on newly frozen held-out bucket positions.
