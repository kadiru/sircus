# Bounded verification of the remaining 30° recipes

This follow-up closes an explicit gap in the translated XY650 study. It is a
fixed verification experiment, not another adaptive search or a PPO update.
It preserves all original geometry, calibrated scoring, native controls,
materials and historical fingerprints. The bucket's 175 mm reported height
remains ambiguous; its existing interior-depth approximation is unchanged.

## Five exact sources

From the independently audited XY650 parent, select these literal coarse records
in this order. Each was a native-valid, rim-assisted capture at 64 substeps and
was absent from the parent's selected verification targets.

| Parent grid suffix | Shared ΔX mm | Shared ΔY mm | Board center H mm | Tilt | Ball-bottom gap mm | Ball center release H mm |
|---|---:|---:|---:|---:|---:|---:|
| 0043 | -10 | 60 | 380 | 30° | 410 | 827.5 |
| 0070 | -10 | 80 | 380 | 30° | 410 | 827.5 |
| 0096 | 0 | 40 | 380 | 30° | 390 | 807.5 |
| 0151 | 0 | 80 | 380 | 30° | 410 | 827.5 |
| 0177 | 10 | 40 | 380 | 30° | 390 | 807.5 |

Full source IDs are `coarse-64-xy-grid-NNNN-r0`. Board and bucket share the
translation, preserving the original 650 mm vector and bearing. Copy each exact
requested joint pair, source record hash and query fingerprint. Check the frozen
pair afresh with the existing strict native FK/collider gate and 5 mm clearance.
No resampling, new IK branch, changed action or rescue is permitted. A rejected
request remains counted and cannot be simulated.

## Budget and gates

Freeze all sources, groups, geometry and plans before any new release:

1. Original positive/negative controls: three copies each at 64 and 128 = 12.
2. If initial controls pass, five exact recipes × three repeats × two resolutions = 30.
3. Repeat the original control block = 12.

Maximum **54 requests**, with a 30-minute native-worker wall limit and a separate
bounded queue wait. Use only the 4090 UUID and the shared scene lock. Do not
start or signal any 4070 or Isaac workload. One external CPU audit rescans all
native dense data and exact source/asset/package/option identities.

Capture retains measured board rebound, downward whole-ball entry and one
continuous second inside. Rim assistance is allowed but stays distinct from
clean entry. Fresh capture qualification requires 3/3 captures at each
resolution, valid physics and independent oracle agreement, both control
blocks passing and at most 5 mm paired displacement at 350 ms. Fresh clean
qualification additionally requires all six new captures to be clean.

Historical parent sources remain rim captures. They are never relabeled clean,
and their historical non-clean label does not make a fresh clean test impossible
by construction. The new explicit parent-source gate requires a valid parent
capture, not a fictitious `source_clean=True` value. Three identical nominal
repeats are a numerical replay check, not a statistical noise-reliability claim.

## Exploration boundary

Before this follow-up, four calibrated 650 mm studies made 1,258 total requests
and 609 native releases, including controls and verification. Discovery alone:
1,042 requests, 393 native releases, 32 rim-assisted captures, zero clean.
The recorded native-worker elapsed times sum to 48.6 minutes; this excludes
queueing, software development, independent audits and UI publication.

This finite check should take minutes of native work after initialization, not
an open-ended overnight run. If none of the five recipes captures consistently
at both resolutions, stop repeating small height/tilt/gap grids. Inspect the
first rim contact, outgoing board velocity and contact-resolution behavior;
prespecify at most one structural impact-point/ballistics experiment before
deciding to change the scene or model. No extra releases are hidden in this
54-request protocol. Noise and further PPO stay deferred.

## Run and evidence

Run from a new frozen source/output namespace in the pinned Ubuntu environment:

```bash
python -m sircus.mjlab.tilt650_pipeline \
  --parent-study XY650_PARENT --parent-audit XY650_EXTERNAL_V2_PROOF \
  --assets ASSETS --output NEW_OUTPUT --gpu-lock SHARED_4090_LOCK \
  --display EXISTING_INTERFACE_REPORT --batch 32 \
  --queue-timeout 1800 --wall-time-limit 1800
```

The full audit is external: `NEW_OUTPUT-audit.json`. The original study and its
passed/failed audit history remain untouched. The additive read-only panel is
`#tilt650Study`, separate from `#xy650Study`; completed counts and actual saved
traces require a matching full independent audit before qualification is shown.
Raw joint arrays, observed poses, dense flight caches, logs and host paths stay
outside Git. This validates held articulated endpoints only, not pickup,
arm transit, actuator tracking, perception, physical deployment or generalization.

## Completed result

[Audited results and scalar contact diagnosis](../../baselines/mjlab-tilt30-650-20261105/README.md):
54/54 native releases, no rejects/invalids. Candidate captures are 15/15 at 64
and 0/15 at 128; all candidate captures are rim-assisted, with zero clean. Both
control gates passed; independent audit passed 14,410 checks. No recipe qualified.
The next recommended impact-point study is a separate decision, capped at
200 requests and one worker-hour; it has not been launched.
