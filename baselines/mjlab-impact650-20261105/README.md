# Bounded impact-point study — November 5, 2026

The [prespecified experiment](../../sircus/mjlab/IMPACT650.md) completed **189
requests / 146 native releases** in **544.3 seconds** of reported study runtime,
below the 200-request / one-hour user cap. The remaining eleven requests
were not spent. Five selected settings failed cross-timestep capture verification;
**zero capture-qualified and zero clean-qualified recipes** were found.

The independent CPU audit passed **all 54,829 checks**, including actual shifted
ball initialization, dropper FK, fresh endpoint admission, native contact rescoring,
source/asset/manifest identities, literal request accounting and final gates.
Passing that audit validates the evidence and its negative result; it does not
turn the failed capture gate into a successful shot.

The runtime timer starts after imports and parent-input reads, and excludes the
independent CPU audit. The full supervisor completed in 610.8 seconds from launch
through that audit, also bounding the acquired-slot interval well below one hour.
Neither number is an isolated GPU-kernel benchmark.

| Phase | Requests | Native | Captures | Clean | Rim-assisted |
|---|---:|---:|---:|---:|---:|
| Initial controls | 12 | 12 | 6 | 6 | 0 |
| Prescribed impact grid | 135 | 92 | 5 | 0 | 5 |
| Five settings × three repeats × 64/128 | 30 | 30 | 15 | 0 | 15 |
| Final controls | 12 | 12 | 6 | 6 | 0 |
| All phases | 189 | 146 | 32 | 12 | 20 |

The 43 endpoint/clearance rejections remain in the requested denominator; they
are not simulated misses. All twelve clean captures are positive controls,
not 650 mm candidate successes. Positive and negative controls passed at both
resolutions, before and after the search. Every candidate capture was rim-assisted.
Three discovery captures exactly replay the zero-offset historical recipes;
two add a -10 mm lateral offset. No nonzero-forward setting captured in this grid.

## Exact settings and fresh results

All selected boards have center height **380 mm** above the table. Tilt is from
horizontal. Gap is **vertical ball-bottom clearance above the nominal drop-ray
point on the board center plane**, not clearance above its physical top face.
Release H is the ball-center height above the table. Forward follows the board's
downward surface tangent toward the bucket; lateral runs across the board.

| Family | Tilt ° | Gap mm | Forward / lateral mm | Release H mm | Nominal point-to-bucket mm | Capture64 | Capture128 | Paired350ms drift mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 00 | 27.5 | 391.8 | 0 / 0 | 809.3 | 650.000 | 3/3 rim | 0/3 | 0.0865 |
| 01 | 30 | 410 | 0 / -10 | 827.5 | 650.077 | 3/3 rim | 0/3 | 1.5730 |
| 02 | 30 | 390 | 0 / 0 | 807.5 | 650.000 | 3/3 rim | 0/3 | 1.6993 |
| 00 | 27.5 | 391.8 | 0 / -10 | 809.3 | 650.077 | 3/3 rim | 0/3 | 0.1043 |
| 01 | 30 | 410 | 0 / 0 | 827.5 | 650.000 | 3/3 rim | 0/3 | 1.5768 |

All thirty fresh repeats pass physics validity and independent-oracle agreement.
All five pairs satisfy the 5 mm early-position-drift bound but fail capture
agreement: 15/15 captures at 64 versus 0/15 at 128, with no whole-ball entry at
128. Three identical repeats per resolution are nominal replay checks, not noise
robustness or a statistical success-rate estimate.

Board-center-to-bucket distance remains **650 mm** throughout the prescribed
135-point grid. Actual nominal point-to-bucket horizontal distance varies with
impact offset and is separately published for every request. This distinction
prevents a shorter off-center shot from being mislabeled a 650 mm ball flight.
The finite sphere's measured contact point is not the nominal center-plane point.

## What this establishes

This is a bounded development search around three fixed H/tilt/gap families,
not exhaustive coverage or new RL training. The admitted off-center shots did
not produce a timestep-stable capture. Stop the search at its fixed budget;
inspect the outgoing-flight and first-rim-contact evidence before deciding on a
new geometry/contact experiment or another PPO run. No noise experiment or PPO
run was started. A negative result here does not prove every possible 650 mm
configuration impossible.

Six recent calibrated 650 mm studies now total **1,501 requests / 809 native
releases**, including 144 controls and 180 verification requests. Discovery
totals are 1,177 requested / 485 native releases, with 37 rim-assisted captures
and zero clean captures. These are attempted configurations, including repeated
historical targets, not unique state-space coverage or policy test performance.

## Evidence and boundaries

- Frozen executed source: `a96d410da3be7c5f0ed52e1462c8d8af6ffdf2b6`.
- Source archive SHA256: `a4419fb6fe8f251a33bf08584951e35b670a4d64ce0908802a6d8655397ce186`.
- Contract fingerprint: `1c59ce4342e6064b52f43ab64856a2245019de574ae563c88894f179840a0f31`.
- Independent proof SHA256: `ee7331edec9580fed7cdea0b9b28ab5a40cc6337034fa1f7e8b8d5cae17b4883`;
  proof version `sircus-impact650-independent-audit-v1`.
- [Curated scalar summary](summary.json), [all 189 requests](requested-shots.csv)
  and [checksums](checksums.json) preserve geometry, source lineage, exact result
  denominators, audit bindings and separate capture/clean gates.
- The original XY650 source, its separate corrected external audit, earlier
  rectangular/bucket results and all historical definitions remain unchanged.
- Board-arm joints stay fixed within each family. Actual off-center release
  positions and matching dropper FK are checked; fresh endpoint pairs require
  5 mm collision clearance. Held endpoints do not validate grasping, arm transit,
  actuator tracking, perception, generalization or physical deployment.
- Bucket geometry remains the original tapered open container: inner top/bottom
  diameters 177/150 mm, outer 200/158 mm. Its reported 175 mm height still has
  unresolved overall-height versus interior-depth meaning; the existing explicit
  interior-depth approximation plus 8 mm base is preserved.
- Native source, raw joints/poses/contact traces, logs and host configurations
  remain outside Git. Only the designated 4090 worker ran; 4070 and Isaac
  workloads/configuration were not changed. The existing viewer was not restarted.

[Results and actual saved native flights in the UI](http://127.0.0.1:18781/#impact650Study).
