# Clean fixed-650 mm height/tilt/gap study

This bounded development search seeks clean board-to-bucket capture under the
frozen current-ball calibrated scorer. The bucket and board XY stay at the exact
prior650 layout; board orientation faces the bucket. Only board height, tilt
from horizontal and vertical ball-bottom gap change. Heights are above tabletop.
The measured300×200×5mm board, ball properties, contact materials and provisional
bucket interpretation are preserved: reported175mm remains ambiguous, with the
existing175mm interior plus8mm base approximation.

## Prespecified requests

- Initial controls: fixed positive/negative source configurations, three nominal
  copies each at64 and128 substeps:12 requests. Positive must capture cleanly3/3
  at each resolution; negative must miss0/3 with valid native releases. All require
  valid physics and independent scorer agreement before search can start.
- Coarse: H220/260/300/340/380/420mm × tilt16/20/24/28/32/36° ×
  gap200/240/280/320/360/400/440/480/520/560mm =360 requests at64, plus the exact
  original380mm/28°/386.8mm reference =361.
- Local refinement: the top three native valid coarse records each receive the
  full27 combinations of H±20mm, tilt±2°, gap±20mm, including the original center
  again. All81 requests count; these are deterministic nominal repeats, not noise.
- Verification: select up to five distinct target settings from coarse+refinement and
  replay their exact joints/actions three times each at64/128, at most30 requests.
- Final controls repeat the initial12. Maximum total496 requested attempts.

Sources rank by clean capture, capture, whole entry without external contact,
larger mouth clearance, longer retention, smaller opening distance, then stable
record ID. Native failures and endpoint rejects are retained in complete counts.
The conditional selections and exact derived plans are saved before execution;
there is no hidden resampling, clipping, successful fallback or test-set claim.
Verification selection keeps the highest-ranked record for each target rounded
to nine decimals solely for duplicate detection; executed values are unchanged.

A verified clean setting requires a clean source record,3/3 clean captures at
both resolutions, valid physics, independent scalar/oracle agreement, at most5mm
paired ball-position difference at350ms, and passing initial+final controls.
Capture retains board rebound, downward whole-ball mouth entry, and one continuous
second inside. A rim-assisted capture stays distinct from a clean capture.

## Geometry and scope

Query-stable CPU IK uses16 starts/180 iterations with outcome-independent cached
mathematical target pools. Every selected joint pair receives fresh native FK,
joint-limit, actual bucket-collider and independent5mm clearance checks. The
whole outer bucket needs5mm tabletop margin. The old cheap reach sphere does not
veto requests. Exact controls and verification joints are preserved; changed
targets are polished instead of silently reusing source poses.

These are prescribed held endpoints during ball flight. The study does not
validate grasping, actuator tracking, moving-arm collisions, transit paths,
perception or physical performance. No noise study, PPO update or active-policy
replacement is included. Prior failed650 replay and historical scoring results
remain in their original namespaces and interface panels.

## Run

Use the pinned Ubuntu environment, a new immutable source/output directory, the
shared4090 GPU lock and the existing native assets. The pipeline selects only the
4090 UUID; it neither restarts services nor changes the4070 or Isaac setup.

```bash
python -m sircus.mjlab.clean650_pipeline \
  --old-study OLD_SHALLOW_STUDY --positive-study POSITIVE_CONTROLS \
  --positive-audit POSITIVE_AUDIT --replay-study PRIOR_CALIBRATED650_REPLAY \
  --calibration-report FROZEN_CALIBRATION --verification-report FROZEN_VERIFICATION \
  --assets ASSETS --output NEW_OUTPUT --gpu-lock SHARED_4090_LOCK \
  --display EXISTING_INTERFACE_REPORT --batch 32 \
  --queue-timeout 14400 --wall-time-limit 7200
```

After native execution, `clean650_verify` reconstructs the literal/adaptive plans,
fresh CPU endpoints, dense contacts/scoring, timing, controls, selections and UI
metadata independently. Its full passed check list and exact report/manifest
bytes gate the read-only display's qualification claims. Raw joints/trajectories,
logs and generated runtime state remain outside Git. Additive `web/clean650.html`
supports inline CSS/script insertion into the existing viewer without a restart.
