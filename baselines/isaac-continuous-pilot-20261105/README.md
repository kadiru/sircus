# Continuous cooperative shot pilot — November 5, 2026

A fresh continuous-action PPO policy improved capture on its known training
layouts, while held-out capture performance remained unchanged. This is a
single-seed exploratory Isaac Sim study, not a table-wide or physical success
claim. The earlier categorical policies and completed benchmarks stay separate.

## Results

| Frozen layout subset | Initial | After 1,600 training attempts |
|---|---:|---:|
| 25 training sites | 8/200 (4%) | 32/200 (16%) |
| 5 validation sites | 8/40 (20%) | 8/40 (20%) |
| 5 test sites | 8/40 (20%) | 8/40 (20%) |
| All 35 sites | 24/280 (8.6%) | 48/280 (17.1%) |

Each site received eight nominal repeats of one deterministic prediction per
checkpoint. Those repeats are not independent new layouts or noise robustness
trials. Sites are held out from this optimizer, but their coordinates are
inherited from earlier project development. Initial capturing sites were
`r1c4`, `r3c4` and `r4c4`; the final model added training sites `r1c3`, `r1c5`
and `r2c3`. Twenty-nine final sites remained 0/8. Clean captures stayed 8→8.

Training comprised 64 batches × 25 sites, 16 PPO updates, 62 qualified captures
and 1,214 rejected requests. Of 386 released shots, all rebounded from the board;
324 missed, with first external contact table 297, robot 16 and unobserved 11.
Some table failures followed bucket contact; they are not all proven undershoots.
Seventeen retained balls were excluded because of an intervening table contact.
All observed robot contacts on valid misses followed the board rebound.

No training/evaluation capture had board-to-bucket radius ≥600 mm; 176 admitted
training releases in that range missed. No captured target exceeded the
conservative general dropper-only XY reach bound. Far-target cooperation and
broad spatial generalization remain unproven.

## Feasibility is separate from capture

The geometry-only 512-Sobol-per-site survey retained all 17,920 requests. It found
155 IK/FK/proxy-qualified endpoint candidates at 33/35 bucket sites; 153 passed
sampled path proxies from canonical home. These are not successful-shot labels.
`r1c7` and `r2c7` remain unresolved, not proven impossible. Native release guards
were applied anew, including to every rejected PPO attempt.

The policy chooses bearing, radius, board height, vertical drop gap and tilt.
Heading faces the bucket; release XY lies above the board. No menu projection,
rejection resampling or per-scene ballistics optimization occurs at inference.
The actor receives privileged geometry, not camera/point-cloud estimates.

Observed training requests spanned board height 217–492 mm, gap 118–403 mm,
tilt 15.7–45.8°, radius 283–1066 mm and bearing offset about −72–65°. Allowed bounds
are broader; this was sampled learning, not an exhaustive sweep.

## Example final predictions

These exact model predictions captured in 8/8 nominal repeats. Coordinates are
table-local and heights are above the table. Full precision, all 35 predictions,
and rejected cases are in [settings.csv](settings.csv) and
[predictions.json](predictions.json). They are simulation settings, not physical
calibrated poses.

| Site | Subset | Board X/Y mm | Board H mm | Gap mm | Ball-center release H mm | Tilt ° |
|---|---|---:|---:|---:|---:|---:|
| r1c3 | train |549.0 / −109.9|348.2|237.8|623.5|24.8|
| r1c4 | validation |567.8 / −61.2|351.2|242.3|631.0|24.9|
| r1c5 | train |614.3 / 14.3|351.3|241.6|630.4|25.4|
| r2c3 | train |658.2 / −147.9|343.2|229.8|610.5|25.4|
| r3c4 | train |634.1 / 35.2|333.4|214.2|585.1|28.4|
| r4c4 | test |619.7 / −132.2|334.5|212.2|584.2|28.3|

Gap means ball-bottom release height minus board-center height. Ball-center
height adds the provisional 37.5 mm radius. It is not a measured gripper pose.
The board is 200×300×5 mm in landscape orientation. Bucket inner top/bottom are
177/150 mm; outer 200/158 mm. Reported height 175 mm still has unresolved overall
height versus interior-depth interpretation; interior 175 mm plus 8 mm bottom
remains the provisional model. Contact properties and rigid mounting are fixed
provisional assumptions in this study.

## Recording and verification

Twenty scheduled batches produced 19,929 actual native 1080p/30fps frames
(about 11:04): four geometry-coverage batches and sixteen early/middle/late PPO
batches. Coverage outcomes never entered the optimizer. The chosen fixed camera
and silver-gray stage are preserved. No ball-following or generated replacement
imagery is used. Episode resets occur before recording; all subsequent shot
setup is driven and measured, with no scored-endpoint teleport. Grasping, paths
and ball loading are not learned.

The nearest table has six recorded releases and no captures. Some visible
motion includes unnecessary IK joint winding; it is not learned orbital motion.
The longer film honestly retains held releases and unsuccessful trials.
Full clips and a setup-dwell-trimmed alternative remain outside Git with raw
RGB, 240 Hz ball/arm/board/contact arrays and logs.

Independent audits checked fresh seed initialization, 66 checkpoints, 84 native
batches, 16 update links and every active attempted shot. Native outcomes were
reconstructed from 2,733,850 all-cell observations; 2,393 recorded release-frame
readbacks matched robot/ball arrays. CPU/GPU log-likelihood error was ≤6.68e−6.
Canonical evaluation homes matched within 1 mrad. Maximum observed board
movement through scoring was 0.529 mm. Mesh/self-collision certification and
physical calibration remain outside this study.

## Artifacts and reproduction

- [summary.json](summary.json): complete counts, action ranges and scope.
- [study.json](study.json), [map-contract.json](map-contract.json): separate fingerprinted learning and geometry-survey protocols.
- [policy-initial.pt](policy-initial.pt), [policy-final.pt](policy-final.pt): fresh initial and final continuous policy/critic weights, guarded by study/domain/geometry fingerprints.
- [checkpoint-audit.json](checkpoint-audit.json), [native-audit.json](native-audit.json), [feasibility-audit.json](feasibility-audit.json): independent receipts and source digests.
- [failure-summary.json](failure-summary.json): aggregate per-site, surface and radius findings; raw observations are excluded.
- [film.json](film.json): encoded movie/clip digests, measured movement and edit provenance; media stays outside Git.
- [manifest.json](manifest.json): file digests and native/video provenance.

Native producer: `c0e5dfd5172cce27be76e22797d39af15cf7506a`; auditors: `1cb55b2927f9b2584998dc8b77f83dc99d87484c`; video/report exporter: `d4076510041c65844a197584a6909ef38521c89b`.
The 61.34-minute native runtime includes filming and assessment; it is not a
headless-training compute benchmark. Method, commands, pinned dependencies and
read-only viewer are documented in [CONTINUOUS_PILOT.md](../../sircus/isaac/CONTINUOUS_PILOT.md).
The committed older farm bundle supplies geometry/camera provenance without
requiring its raw footage. Preserve scene locks and the dedicated 4090 allocation.

Final Mac regression: 34 physical-controller mocks, five simulation tests and
383 Isaac tests completed successfully (46 dependency skips), plus all 36 JS
checks and the historical two-arm physics/scoring/study checks. All 90 continuous-pilot CPU Torch tests also passed on Ubuntu with no skips,
including the policy, likelihood, independent audits and film-caption paths.

No physical robot, Viam, audio, existing dashboard or 4070 configuration was changed.
