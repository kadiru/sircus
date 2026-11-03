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
