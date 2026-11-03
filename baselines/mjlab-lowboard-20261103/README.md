# Completed lower-board far-target study — November 3, 2026

**No tested setting captured at 650–770 mm.** Each of the four targets had
52 admitted native releases out of 160 requested shots; all 208 far releases
missed the strict route. The independent CPU audit passed **2,435 checks**,
including scalar rescoring of all 232 native releases. The other 24 releases
were exact parent controls, and all 24 captured. Control success is separate
from far-target success.

The 4090 run took 498 seconds (8.3 minutes), using frozen execution commit
`bb68e42ca9fc7786788df0420d58ea3816587003` and fingerprint
`2581c867b89056977d7085e3be115da6b5bbe9562b24d816ddb2a1b219bbca74`.
This is finite DEVELOPMENT search, not a new learned policy or proof that all
far shots are impossible. The [parent distance study](../mjlab-distance-20261103/README.md)
and its reliable 450/472.826 mm settings remain separately identified.

## Requested and executed settings

[Protocol and launch instructions](../../sircus/mjlab/LOWBOARD_STUDY.md): fixed
board XY 470.875779/−277.893401 mm; four exact parent corner-ray targets at 650,
700, 750 and 770 mm. Requested board heights 220–360 mm by 20 and independent
28/32/36/40° tilts. Ball-center release levels were 5/15/30/50/80 mm below the
unchanged cheap-prior ceiling 819.324338 mm; gap=release−board height−37.5 mm.
No clipping, resampling or guard relaxation.

| Bucket distance mm | Requested far shots | Native releases | Geometry rejects | Strict captures |
|---:|---:|---:|---:|---:|
|650|160|52|108|0|
|700|160|52|108|0|
|750|160|52|108|0|
|770|160|52|108|0|

There were 128 development H/tilt contexts; 52 admitted a board pose. The
640 development requests retained 380 workspace-prior rejects and 52 strict
clearance rejects. The top 814.324 mm release failed dropper links 4/6
self-clearance; 804.324 mm was the highest tested admitted release. That is a
sampled endpoint result, not an exact reach boundary or a motion-path guarantee.

Actual admitted board settings per target were H260/T40; H280/300/320 at 36/40°;
and H340/360 at 32/36/40° (mm/degrees), each with 15/30/50/80 mm headroom. The
admitted gaps span 341.824–506.824 mm and ball-center releases 739.324–804.324 mm.
All H220/H240 and all 28° requests remain rejected under the unchanged prior.

## Why the far shots missed

The [post-hoc contact diagnostic](completed/far-contact-diagnostics.json) found
zero whole-ball entries in all 208 far native flights. Every flight's first
external contact was the table, although bucket-exterior contacts can precede
that table contact. Best recorded unassisted descending rim-center crossings
reached about 507 mm at the 650 mm target and 551 mm at the other three targets.
These are descriptive sampled crossings before any non-board contact, not
whole-ball entry tests or global flight-range limits. The 650 mm subset includes
seven requests with contact before crossing or no observed crossing; its smaller
maximum is not comparable as a universal maximum range. Material and scoring definitions were unchanged.

## Repeatability and the next experiment

No actual capture was available for recipe selection. Therefore no refinement
or fresh paired repeat was run, and no 650–770 mm configuration can be called
reliable. Only 64 substeps were used natively in this study. Declared 32/128
contracts are not evidence of timestep robustness; old paired results retain
their original scene and scope. [Recipe CSV](completed/recipes.csv) intentionally
contains only its header, rather than invented successful settings.

A separate [post-hoc CPU-only boundary diagnostic](posthoc-tilt-boundary-cpu.json)
tested H380/400/420 at 24/28/30/32° and four headrooms for the 650 mm target.
It admitted 36/48 endpoints; 28/30/32° passed, 24° failed the prior. This locates a
promising next family: slightly higher boards that permit a shallower tilt.
For example H380/T28 with 15 mm headroom gives G 386.824 mm and ball-center
release 804.324 mm. **These are geometry-only proposals; no ball flight or capture
was measured for that diagnostic, and it is excluded from this frozen study.**

## Evidence and interface

- [Summary](completed/summary.json), [target counts](completed/target-search.csv),
  [all context aggregates](completed/context-search.csv), [settings](completed/settings.json),
  [curated scene](completed/scene-geometry.json), [provenance](completed/provenance.json)
  and [checksum manifest](completed/SHA256SUMS).
- [Best observed development examples](completed/best-observed-development.json)
  list exact settings and failed outcomes. Their post-hoc selection prioritizes
  route capture, whole entry, retention, closest opening and stable ID. They are
  illustrations, not a new policy score or proof of maximum first-flight range.
- `http://127.0.0.1:18781/#lowBoardStudy` shows the matching full audit, explicit
  far-target counts separate from controls, H/tilt maps and four actual saved
  failure traces. Existing solver/policy routing and historical panels remain
  unchanged. Static/derived files were updated without restarting the viewer.
- Immutable source/evidence: `studies/lowboard-v1` /`output/lowboard-v1`. The native
  and CPU-audit workers exited normally. Raw joints, trajectories, forces, logs,
  queues and credentials remain outside Git. The task used 4090 only and issued
  no 4070 process commands, Isaac deployment changes, hardware/Viam/audio commands
  or policy activation.

The bucket keeps its reported 175 mm height uncertainty and provisional 175 mm
interior plus 8 mm base. Strict endpoint checks and these model-specific flights
do not establish arm transit, actuator tracking, perception or physical accuracy.

---

The following preflight is retained as separately labeled historical CPU evidence.

# Lower-board far-target study — CPU preflight

Status: **strict CPU geometry preflight passed; new ball-flight results pending**.
The [preflight proof](cpu-preflight.json) records 1,769 passed checks, with no
failures. It reused the existing immutable parent `distance-v1` geometry and
native CPU models unchanged. It did not validate new study orchestration, run
GPU dynamics, measure a capture, train a policy, or test an arm path.

The fixed board XY is 470.875779 / −277.893401 mm. All four exact parent corner
targets are retained: 650, 700, 750 and 770 mm. The grid requests board heights
220–360 mm in 20 mm steps, tilts 28/32/36/40°, and five release levels near the
existing spherical workspace prior. Eight original geometry hints are assigned
by nearest height/40 mm and tilt/4°, with stable source-ID ties, before outcomes.

## Reach ceiling and requested releases

The unchanged proposal guard is

`Zball ≤ 0.364 + sqrt(0.842² − X² − Y²) − 0.095 − 0.09`.

At this fixed XY, its ball-center ceiling is **819.324338 mm**. This is a cheap
workspace prior, not the exact URDF reach envelope or native collision-free
ceiling. Requested ball-center heights are 814.324338, 804.324338, 789.324338,
769.324338 and 739.324338 mm. Drop gap equals release height minus board height
minus the 37.5 mm ball radius. No gap is clipped or retried.

## Endpoint results

- 136 contexts were checked: 128 development contexts and eight exact source
  control contexts. All 24 repeated source-control queries retained the original
  joint arrays and passed strict paired admission.
- Each far target has 32 requested H/tilt contexts. Thirteen pass board admission;
  19 are conservatively rejected by the existing board workspace prior before
  IK. This does not establish that those rejected configurations are globally
  impossible.
- Each target retains all 160 gap requests. Fifty-two pass strict paired endpoint
  admission; 95 fail the board workspace prior and 13 fail strict clearance.
  Across four targets: **208 admitted endpoints /640 requested attempts**, with
  380 workspace-prior and 52 strict-clearance rejections. No ball was simulated.
- The 814.324338 mm release level fails the 5 mm self-clearance requirement
  between dropper links 4 and 6 in all 13 board-admitted contexts per target.
  Checked branch lower bounds are about 2.03–2.18 mm. The highest tested
  paired-admitted release is **804.324338 mm**; the exact boundary between those
  tested levels remains unknown.
- Minimum admitted clearance lower bound: 5.409159 mm. Maximum independent
  native full-pose position error: 8.304 nm. These endpoint checks do not prove
  arm transit, actuator tracking, or physical accuracy.

The admitted board contexts are H260/T40; H280/T36 or40; H300/T36 or40;
H320/T36 or40; H340/T32,36 or40; and H360/T32,36 or40 (mm/degrees).
Each admits the 15/30/50/80 mm release-headroom levels. All H220/H240 and all
28° requests remain recorded rejections under this unchanged prior.

## Provenance and limits

The proof includes source/package/asset hashes, the audited parent fingerprint,
geometry branch-proof hashes and the raw preflight checksum. Raw joints and
pose/trajectory arrays, credentials and host calibration are excluded.

The new protocol was imported privately in memory and its exact source
assignments, H/tilt/gap values, bucket coordinates and 24 control requests matched
the preflight plan. Only provisional IDs and ordering differed. This is protocol
geometry alignment, not validation of a new native runner or ball dynamics.

The run took 64.35 seconds with CUDA hidden and bytecode writes disabled. The
immutable source was not edited. No GPU jobs, 4070 workload changes, Isaac
changes, service restarts, hardware or cloud actions were performed. The bucket's
reported 175 mm height interpretation remains unresolved, as in the parent.
