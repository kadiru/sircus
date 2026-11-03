# Corrected-contact spatial development map

This is a finite, headless **development search**, followed by independent CPU
verification. The native study and both CPU audits completed. Seventeen sources at 12 of 35
positions qualify; full results and exact settings are in the dated baseline. It does not train PPO, replace an active policy, restart a
service, or change Isaac, the 4070 workload or physical hardware.

The read-only viewer section is `#correctedGrid`. Curated results belong in
[the dated baseline](../../baselines/mjlab-corrected-grid-20261102/README.md).
Historical contact, rectangular-container and bucket results retain their original
definitions and fingerprints.

## Physics and scoring

The reference model is articulated, with passive gravity compensation only:
every arm body already has `gravcomp=1`, so the duplicate applied `qfrc_bias`
force is removed. It uses **64 native substeps per 240 Hz scoring interval**,
or 15,360 Hz integration. The original `solref=(.004,.15)`, friction 0.35, geometry,
materials and scorer remain unchanged. Effective runtime options are checked:
Newton solver, 100 iterations, tolerance 1e-8, pyramidal contact cone, implicitfast
integration and gravity −9.81 m/s². This is a new explicit physics version; an old
scene description is not evidence that old policies match the new runtime.

Joint endpoints are prescribed before each native step. This is a held-posture
bounce study, not full grasping or actuator tracking. Higher resolution addresses
the contact sensitivity found in [the repeatability study](CONTACT_REPEATABILITY.md);
it does not establish physical calibration or bitwise deterministic trajectories.

A qualified route still requires a board rebound, whole-ball entry and one
continuous second retained. Partial exits reset retention. Bucket rim-assisted
captures count; intervening table, floor, robot or unattributed assistance does
not. Clean entry is reported separately.

The board remains 200×300×5 mm with its landscape mount rotation. The bucket is an
open straight-sided taper: inner top/bottom diameters 177/150 mm, outer 200/158 mm.
**175 mm is reported height; overall height versus interior depth is unresolved.**
The retained provisional model assumes 175 mm interior depth plus an 8 mm bottom,
so the modeled rim is 183 mm above the table. Ball properties and mount compliance
remain provisional.

## Requests, admission and qualification

- **35 DEV layouts:** 7 X coordinates from 0.10 to 1.24 m and 5 Y coordinates from
  −0.65 to 0.25 m. Every requested position remains in the map.
- **336 geometry queries per layout:** 17 preserved relative templates × 16
  rotations, plus 64 seeded geometry proposals. Old source joints are continuation
  hints only; historical captures and qualification labels do not become new
  successes. There are 11,760 initial geometry queries, not 11,760 native throws.
- **Strict endpoints:** deterministic 16-start/180-iteration IK, joint limits,
  native FK, self-collision, both arms, carried ball, board and world checks with
  a 5 mm modeled clearance requirement. Eligible native candidates are selected
  for geometric diversity, without using old outcomes, at most 64 per layout.
- **One fixed common home:** its exact joints and hash are frozen before native
  outcomes. All 35 layouts receive a home-clearance preflight. Actual negative
  bucket/base contacts are separated from movable-link staging obstruction and
  unproven clearance. These categories are not interchangeable with IK failure.
- **New capture sources:** at most two per layout must connect from that common
  home to their exact selected joints and back. Low board routes and high dropper
  routes are serialized and sampled. A round trip around a source's own endpoint
  is insufficient evidence of common-home connectivity.
- **Confirmation:** each admitted source gets 3 nominal attempts and 20 fresh
  perturbations. Qualification requires 3/3 nominal captures and at least 18/20
  fresh captures. Fresh requests require strict endpoint admission and checked
  local paths composed with the source's common-home route. Geometry and motion
  rejects remain failures within the 20-attempt denominator.

Perturbation standard deviations are 1 mm board X/Y, 2 mm board height/drop gap and
0.35° tilt. Counts distinguish rejected geometry, eligible-but-not-sampled
requests, native releases, search captures and confirmed sources. A finite search
failure does not prove a bucket position impossible.

The contract freezes **30 future validation and 100 future test layouts** before
this map's native outcomes, separated by at least 5 mm from the supplied mjlab
historical coordinates and one another. The grid is DEV, then potentially TRAIN.
No outcome search or policy evaluation on those future splits occurs here.
This does not claim disjointness from every experiment ever performed.

## Reproduce without deploying

Use a new immutable source snapshot and fresh output path with the existing pinned
mjlab environment. Set `CUDA_VISIBLE_DEVICES` to the authorized 4090 UUID recorded
as `GPU_UUID` in `gpu_slot.py`; expose no other GPU. Paths below are placeholders.

```bash
python -u -m sircus.mjlab.corrected_pipeline \
  --parent-study /path/to/learning-v2 \
  --bank /path/to/expanded-learning-v2/proposal-bank.json \
  --contact-reference /path/to/contact-confirm-v1/report.json \
  --source-study /path/to/learning-v2 \
  --source-study /path/to/expanded-search-v2 \
  --source-study /path/to/expanded-refine-v2 \
  --source-study /path/to/stable-ten-v1 \
  --historical-study /path/to/bank-learning-v1 \
  --historical-study /path/to/expanded-learning-v2 \
  --assets /path/to/native-assets \
  --gpu-lock /path/to/shared/scene.lock \
  --output /path/to/new-corrected-grid \
  --queue-timeout 21600
```

`corrected_pipeline` runs the finite grid job and then `corrected_verify` with
CUDA hidden. It never starts training. It waits on the shared 4090 lock for at
most six hours without interrupting another job. The wrapper bounds the grid
process by queue timeout plus two hours, then bounds the CPU audit to 20 minutes.
The study stops on its 512 MiB evidence cap or a 512 MiB free-space floor; it never
deletes another experiment's data.

The contract binds source, asset, dependency, geometry-bank, reference-contact and
common-home identities. Compressed 240 Hz positions, velocities and categorized
contact forces support independent scalar rescoring. Preserve raw evidence and
runtime snapshots outside Git; publish curated summaries and their checksums.

The completed run also used a separately reviewed supplemental CPU verifier.
Reuse the recorded pipeline `audit_command` from its JSON status, run from the
reviewed audit-only snapshot with CUDA hidden, retain `--source` pointing to the
original executing snapshot, and change `--output` to
`STUDY/audit-corrected-supplement.json`. This validates original native sources
without replacing them. The display's training assessment requires both this
supplemental output and `audit-corrected.json` to match and pass; the basic finite
pipeline alone creates only the latter. Their overlapping checks are displayed
separately. Neither audit launches training.

Review the completed map and audit before proposing a corrected-version training
study. Any later learned-versus-uniform comparison needs frozen shared candidate
masks/joints, an actual saved checkpoint and outcome-free inference on the frozen
test split. The existing viewer's old policy routing stays unchanged meanwhile.
Sampled geometry does not certify continuous collision avoidance, dynamics,
grasp stability, perception or continuous-table coverage.

## Completed checkpoint and training readiness

The study completed 460 search releases, 87 nominal repeats and 580 fresh requests
(567 releases; 13 rejected requests remain failures). Captures were 349, 83 and 495,
respectively. The 17 qualified sources cover 12/35 development layouts; only five
layouts have two qualified sources. Both audits independently rescore the same
1,114 releases and pass 3,115 and 3,165 overlapping checks. Neither rightmost grid
column (X≥1.05 m, ten positions) produced a search capture.

The evidence supports a **limited pilot after a matching training adapter and
smoke checks**, not immediate reuse of the old trainers. `bank_study` and
`expanded_study` still instantiate legacy environments/masks; their policy
mathematics can be reused, their recorded physics contracts cannot. A corrected
pilot must use `ContactEnvironment(substeps=64, board_mode="articulated",
gravity_mode="passive_only")`, `CorrectedGeometry`, fixed-home admission and
shared frozen masks/joint branches for learned and uniform choices. Path rejects
stay failures or are excluded by the same geometry-only mask for both methods.

Deduplicate repeated relative geometry prototypes before sampling: the 17 absolute
source poses/joint pairs represent 12 oriented relative prototypes. Preserve every
source/branch provenance; identical endpoint geometry does not imply identical
arm clearance or dynamics. If branch choices remain separate actions, include
features that distinguish them. Do not let duplicate source records artificially
weight the uniform baseline.

A proposed fresh-init 2,048–4,096-throw terminal-PPO pilot is a centralized
contextual shot selector with privileged bucket geometry; IK and geometric path
planning remain external. Freeze the geometry-derived candidate bank and existing
35/30/100 split, select checkpoints using validation only, then evaluate frozen
learned and uniform policies without test rollout search or resampling. Keep all
unsupported layouts in headline denominators. This pilot has not been launched.
