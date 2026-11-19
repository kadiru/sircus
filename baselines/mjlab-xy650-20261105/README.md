# Translated XY650 study — November 5, 2026

The [prespecified protocol](../../sircus/mjlab/XY650.md) completed 388 requests,
228 native releases and 160 geometry rejections. No clean 650 mm setting was
discovered or qualified. Both initial/final control gates passed. Corrected
independent CPU audit v2 passed all 80,138 checks against unchanged observations.

Board and bucket were translated together by ΔX−10/0/+10 mm and ΔY 40/60/80 mm,
preserving their original 650 mm vector and bearing. The arms, board, ball,
materials, calibrated scoring and provisional bucket dimensions were unchanged.
This is a family of translated layouts, not a new result for the original fixed XY.

| Phase | Requested | Native | Captures | Clean | Geometry rejects |
|---|---:|---:|---:|---:|---:|
| Initial controls |12 |12 |6 |6 |0 |
| Coarse grid and references |253 |105 |15 |0 |148 |
| Local refinement |81 |69 |9 |0 |12 |
| Five selected settings, 64/128 |30 |30 |15 |0 |0 |
| Final controls |12 |12 |6 |6 |0 |
| Total |388 |228 |51 |12 |160 |

All 12 clean captures are positive controls. Discovery only: 334 requests,
174 native drops, 24 rim-assisted captures and 0 clean. It covered 331 distinct
requested targets, 171 distinct native targets and 21 distinct capturing targets
(configured target keys rounded to nine decimal places). These are development observations, not
held-out policy performance or a complete feasibility map.

## Matched selected recipes

All selected settings used a 380 mm board center. At 64 each captured 3/3,
rim-assisted; at 128 each captured 0/3 and never achieved whole-ball entry.
All physics/oracle checks passed. The small early-flight displacement did not
establish capture agreement near the rim.

| ΔX mm | ΔY mm | Tilt ° | Gap mm | Ball-center H mm | Capture 64 | Capture 128 | Δ350 ms mm |
|---:|---:|---:|---:|---:|---:|---:|---:|
|−10 |60 |27.5 |391.8 |809.3 |3/3 rim|0/3 |0.0865 |
|0 |80 |27.5 |391.8 |809.3 |3/3 rim|0/3 |0.1081 |
|−10 |40 |28 |386.8 |804.3 |3/3 rim|0/3 |1.7582 |
|0 |80 |28 |386.8 |804.3 |3/3 rim|0/3 |1.7360 |
|−10 |60 |28 |386.8 |804.3 |3/3 rim|0/3 |1.7261 |

The five 30° coarse captures—H 380 mm, gaps 390/410 mm at several translations—
were **not selected** by the frozen first-flight ranking and remain unverified
at 128. Their 128 outcomes are unknown, not failures. The three selected refinement
seeds were distinct XY targets but shared the old H/T/G recipe. Future selection
should consider diversity in H/T/G families as well as distinct geometry.

All captured discovery shots encountered bucket contact before a descending
ball-center rim-plane crossing in their first flight. Consequently that
diagnostic crossing margin stays missing. Later positive scored mouth clearance
does not turn a rim-assisted capture into clean entry. No capture definition,
material, calibration or compatibility guard was relaxed.

## Audit and provenance

- Native source 39a50292c7226ec2afc5d7527f63d095b3cdfc53; contract
  `7db7440c12005c39808ee4dfbf831d8b81000cda048a8d4eddc896d9b1d31137`.
- Static scene fingerprint
  `c6dc22893328aebf7d58a0c314cbd1643edc83a305e9efbc1fc80ba11988fff6`;
  per-query and execution fingerprints retain exact target/source/joint lineage.
- Original audit v1 failed because it expected a calibration field in the old
  shallow contract. That historical schema predates calibration metadata. The
  failed proof, logs, native producer, report and 41 evidence artifacts remain
  preserved; no physics was rerun for a checker correction.
- Corrected checker source 211346870497c38f5da0bae625e6785da1842af6 compares
  unchanged scene/packages/assets against all ancestors, retains explicit matching
  calibration on all calibrated parents and the full byte-bound positive-control
  calibration identity. It ran from a separate snapshot against the original
  producer source and saved an external audit v2 proof.
- The audit verifies all 388 requests, exact original sources/assets/packages,
  initial/frozen adaptive plans, unique selections, first-flight metrics from
  dense caches, fresh 5 mm endpoints, actual 64/128 options, independent calibrated
  scoring, native initialized joints, identities, controls and displayed traces.
  All 133 original source files and 25 imported core dependencies remained unchanged.
- Publication source ae28d42d352d40a61cc1cc5fc11e637063d612c3 reads the explicit
  external proof. The read-only viewer was not restarted; historical display
  fields and the original failed audit were preserved. Hash-only publication
  provenance is in `xy650-v1-publication-v2.json`.

`summary.json`, all 388 rows in `requested-shots.csv`, `checksums.json` and
`SHA256SUMS` retain configured coordinates/parameters, scalar diagnostics,
outcomes and hashes. Observed crossing XY, dense trajectories, joints, raw poses,
logs, credentials and host paths are excluded. Raw observations stay on Ubuntu.

This is privileged-geometry shot search with held articulated endpoints, without
PPO, noise qualification, policy activation, grasp/transit, actuator tracking,
perception or physical validation. The bucket's 175 mm reported-height ambiguity
remains explicit. The 4070 workload and Isaac setup were untouched; all finite
experiment/audit workers exited. [Read-only results](http://127.0.0.1:18781/#xy650Study).

Next bounded options: verify the unselected 30° family at both resolutions;
diversify refinement by H/T/G family; allow board-local impact offsets or tool
yaw to explore energy/reach/clearance limits; use a separately labeled no-arm
ballistics control to distinguish flight feasibility from robot constraints.
Noise and further PPO remain deferred.
