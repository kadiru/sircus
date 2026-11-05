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

## Completed coarse search and frozen fine follow-up

The v1 search completed496 requested attempts/155 native releases. Among442
discovery requests,101 native releases produced0clean and2rim-assisted captures,
both the exact original reference. Its matched verification remained3/3rim at64
versus0/3capture at128. Both pre/post control sets passed; the independent audit
passed64,075 checks. [Curated evidence](../../baselines/mjlab-clean650-20261104/README.md)
keeps control counts distinct from650 discovery. This is not proof of impossibility.

The separately versioned `fine-v2` profile keeps the same scene/calibration and
fully binds that audited v1 result. It uses H375/380/385mm × tilt27/27.5/28/28.5/29°
× gap380/386.8/390/395/400mm:75 combinations plus the exact reference. The top3
native results receive27 refinements each using ±5mm height, ±0.25° tilt and
±2.5mm gap. Initial/final controls and top5 matched64/128 verification are
unchanged; maximum211 requests. Selection rules are unchanged and all rejects
remain counted. Only literal profile/version pairs receive a verified UI badge.

Run the pipeline below with `--profile fine-v2 --coarse-study AUDITED_V1_OUTPUT`
in a new frozen source/output namespace. Do not modify a prior producer or its
raw evidence. The UI preserves the previous completed result while showing the
latest profile. Noise and further PPO remain deferred.

Fine-v2 completed211 requests/106 native releases with0clean650 discovery or
verification captures; both control sets passed and auditv2 passed33,824 checks.
The [fine evidence](../../baselines/mjlab-clean650-fine-20261104/README.md) is separate.
A [CPU reach diagnostic](../../baselines/mjlab-clean650-20261104/reach-diagnostic.json)
found a necessary full-pose workspace bound at H380:28° has0.109mm wrist-reach
slack, while27.5° exceeds the model's bound by1.349mm. Higher boards admit lower
tilts with the same804.3mm nominal release center; eight of nine CPU probes pass.

The separately frozen `raised-v3` profile tests H400/420/440mm ×T26/27/28° ×
G326.8/346.8/366.8mm (27 combinations plus reference). It retains the same XY650
scene, scorer, controls and verification; refinement is H±5mm/T±0.5°/G±2.5mm.
Maximum163 requested attempts. Use `--profile raised-v3 --coarse-study AUDITED_V1_OUTPUT`.
This profile is motivated by the CPU endpoint probes; no flight success is assumed.
For inline publication, wrap `clean650.js` in `<script type="module">` so its
exports parse correctly. The existing viewer service need not restart.

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
