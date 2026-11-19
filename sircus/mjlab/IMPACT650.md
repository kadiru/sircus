# Bounded 650 mm board-center impact-point experiment

Completed: [audited results and exact settings](../../baselines/mjlab-impact650-20261105/README.md).
The 189-request study found no 64/128-qualified recipe; both control gates and
all 54,829 independent checks passed. The protocol below is preserved as executed.

This study changes where the vertical drop ray intersects the board's nominal
center plane. It keeps the three seed boards, bucket positions, board-arm joints,
tilts and drop gaps fixed. New dropper endpoints move the actual released ball;
offsets are not just a visualization or a score label.

## Geometry and gap convention

For horizontal unit direction `d` from board center to bucket, tilt `t` from
horizontal, define surface tangents:

```text
forward = (cos(t)*d.x, cos(t)*d.y, -sin(t))
lateral = (-d.y, d.x, 0)
P = board_center + forward_offset*forward + lateral_offset*lateral
release_center = P + (0, 0, gap + ball_radius)
```

`P` is a **nominal drop-ray point on the board center plane**, not an observed
finite-ball contact point or a point on the physical top face. Actual contacts
come from unchanged native dynamics. Gap is vertical ball-bottom clearance
above `P`; positive forward offsets lower both `P` and release height. At zero
offset the original release center and requested joint branches are preserved
exactly. Board thickness, sphere radius and material coefficients are unchanged.

Board-center-to-bucket distance stays **650 mm**. Nominal `P`-to-bucket horizontal
distance varies by several centimetres and is reported separately for every
query; it must not be advertised as a fixed 650 mm ball flight.

## Prespecified source families and inclusive budget

Use the byte-bound, independently audited original XY650 parent and its explicit
external audit-v2 proof. Select these three exact source records in order:

| Family | XY650 source | Board H mm | Tilt | Gap mm | Shared ΔX/ΔY mm |
|---|---|---:|---:|---:|---:|
| family-00 | refinement-64-refine-s02-h1t0g2-r0 | 380 | 27.5° | 391.8 | -10 / 60 |
| family-01 | coarse-64-xy-grid-0043-r0 | 380 | 30° | 410 | -10 / 60 |
| family-02 | coarse-64-xy-grid-0096-r0 | 380 | 30° | 390 | 0 / 40 |

Each family receives nine forward offsets `[-40,-30,-20,-10,0,10,20,30,40]` mm
and five lateral offsets `[-20,-10,0,10,20]` mm, in family/forward/lateral order.

- Initial historical positive/negative controls: 3 copies each at 64/128 = 12.
- Search only after those controls pass: 3 × 9 × 5 = 135 requests at 64.
- At most five selected targets: 3 nominal repeats each at 64/128 = 30.
- Final original control block: 12.
- Maximum **189 requests**, within the user's 200 cap. Endpoint rejects remain
  in the denominator. No extra requests spend the remaining eleven slots.

Retain source q14 exactly. Zero offset retains source q10; every nonzero offset
uses the existing finite 16-start / 180-iteration CPU dropper solver with at
most 17 branches. Every pair, including cached mathematics and verification,
receives fresh native FK/collider admission and 5 mm clearance. Nominal ray
points need 5 mm board-edge margin. No board-arm branch search or cheap reach
sphere veto is introduced. Held endpoints do not validate grasp or arm transit.

## Selection and verification

Rank native-valid records using the unchanged first-flight-only diagnostic:
clean capture, capture, descending ball-center rim-plane crossing, larger
aperture clearance, smaller pre-contact opening distance, stable record ID.
Exclude all positions after the next contact from this ranking. Diagnostic
crossings never replace whole-ball entry, one-second retention or capture.

Select the best distinct target from each of the three literal families, in
family order, then the best remaining globally ranked distinct targets until
five. Empty families are skipped, not replaced with new requests. Uniqueness
includes the offset. Freeze complete selected source records, plan, exact
joints, geometry and preceding ledger hash before verification. Do not refine
height, tilt, gap, scene, material or calibration based on outcomes.

Capture qualification needs a captured search source, 3/3 fresh captures at each
resolution, both control gates, valid physics, independent oracle agreement and
at most 5 mm paired displacement at 350 ms. Clean qualification additionally
requires every fresh capture to be clean. Source quality remains explicit.
Nominal repeated replay is not statistical noise reliability or policy testing.

## GPU and evidence boundary

Use only the 4090 UUID and the shared cooperative lock. Queue waiting is
separate and bounded. Once the slot is acquired, charge all worker time against
at most **3,600 seconds**. An interrupt deadline reserves ten seconds for saving
failure status; the supervisor independently stops only its own worker before
the cap. It never signals a foreign process. Do not start or change the 4070,
Isaac services, cloud configuration or physical hardware.

Native arrays must show actual offset release position and corresponding
dropper FK. The new independent CPU auditor calculates this geometry separately,
rescans dense traces with the unchanged calibrated contact oracle, and verifies
source/package/asset identities, time options, requests, groups, selection,
controls, execution hashes and displayed traces. Old center-only release checks
cannot validate this new action. The external proof is `NEW_OUTPUT-audit.json`.

The additive read-only UI key is `impact650_study`, panel `#impact650Study`.
Historical tilt/XY/clean650 results and fingerprints remain intact. No PPO,
noise experiment, new learned policy, grasp/transit or physical validation is
part of this study. Bucket 175 mm is still a reported height of uncertain
meaning, with the original explicit interior-depth approximation preserved.

```bash
python -m sircus.mjlab.impact650_pipeline \
  --parent-study XY650_PARENT --parent-audit XY650_EXTERNAL_V2_PROOF \
  --assets ASSETS --output NEW_OUTPUT --gpu-lock SHARED_4090_LOCK \
  --display EXISTING_INTERFACE_REPORT --batch 32 \
  --queue-timeout 1800 --wall-time-limit 3600
```

Raw joints, actual pose arrays, trajectories, logs, host paths and credentials
stay outside Git. Only curated nominal geometry, scalar results, checksums and
audited provenance should be committed. After this finite experiment, decide
whether to learn from timestep-stable recipes or revisit geometry/contact
modeling; do not automatically launch another sweep or training job.
