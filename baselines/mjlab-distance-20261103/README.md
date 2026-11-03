# Seeded distance study — November 3, 2026

Status at source freeze: **CPU endpoint preflight passed; native results pending**.
This study uses the eight perfect-capture configurations from the completed
[height/tilt parent](../mjlab-height-tilt-20261102/README.md). It does not replace
that study, activate a policy, or relabel its results.

- [Protocol and launch instructions](../../sircus/mjlab/DISTANCE_STUDY.md): 14
  placements on two separately reported rays, seeded gap sweeps and conditional
  farther-target height/tilt expansion. Maximum 5,589 requested attempts.
- [Pinned Ubuntu CPU preflight](cpu-preflight.json) passed 238 checks: 112 contexts,
  88 admitted board poses, 24 conservative table-footprint rejections; all 24
  exact seed controls retained source joints and strict clearance. Twenty-six of
  48 corner-direction gap queries at 650/750 mm had admitted paired endpoints.
  This is geometry evidence, not ball-flight success.
- Each selected recipe gets 3 nominal plus 30 fresh joint H/tilt/gap variations,
  paired at 32/64/128 substeps. Prespecified capture/fine-pair gates and all
  rejected requests remain explicit. Raw observations/logs stay outside Git.
- Planned immutable source: `~/sircus-mjlab/studies/distance-v1`; planned raw
  evidence: `~/sircus-mjlab/output/distance-v1`. The bounded supervisor acquires
  only the shared 4090 lock, runs an independent CPU audit, and publishes a
  separate `#distanceStudy` section in the existing viewer without restart.

The original ray is limited to about 648.8 mm by the bucket's full outer radius
plus 5 mm table-edge margin; the second ray reaches 772.8 mm. Original 650 and
diagonal 800/950 mm requests are deliberately retained as outside that declared
footprint constraint. Table geometry is not enlarged to make them appear valid.
Support, IK, strict paired endpoints and observed capture are different checks.

This is bounded DEVELOPMENT search and selected-repeat engineering evidence;
no RL, held-out layout result, exhaustive coverage, transit/actuator tracking,
perception or physical validation. Bucket reported-height uncertainty and other
provisional assumptions remain unchanged. The 4070 and Isaac setup are untouched.
