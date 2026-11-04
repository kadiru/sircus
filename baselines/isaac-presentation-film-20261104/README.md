# Fixed-camera native farm film — November 4, 2026

A fresh Isaac Sim 5.1 / Isaac Lab 2.3.2 recording on the RTX 4090: **35 tables,
70 articulated arms and 35 distinct local bucket positions**, using the exact
user-selected camera and unchanged silver-gray stage. The final movie is
**60 seconds, 1920 × 1080, H.264, 30 fps**. Camera position and aim stay fixed.

The edit retains the first 24 complete chronological foreground trials: **22
qualified captures and two table misses**, with concurrent background drops.
It keeps setup, release, board bounce and the immediate outcome, removing quiet
tails. Its 1,827 action frames become 1,800 unique native frames at 1.015× playback.
No outcome-selected omissions, looping, padding or generated replacement imagery.
The only overlay is a small provenance caption. Scripted setup timing is compressed;
this is not a real-time recording of the controller's full execution.

## Scope and evidence

These are **saved-configuration demonstration trials**, with fresh PhysX ball motion
and actual articulation drives. They do not contain new RL updates, learned grasping,
learned joint paths, held-out evaluation or physical validation. Each table covers
24 nearby qualified geometries before repeating six. The source PPO study remains
separate and frozen. The 175 mm bucket height still has its original reported-height
uncertainty; geometry, calibration assumptions and scene fingerprints are unchanged.

- [Verification](verification.json): pinned source/exporter commits, movie/report hashes,
  source bindings, timing, checks and stable final state.
- [Per-table summary](cell-summary.json): 30 attempts per table, 29–30 admitted releases
  and 24 distinct requested configurations per table. All 35 cells released.
- Full recording: 1,050 attempts, 1,049 releases, **329 qualified captures**, 163 clean
  captures and 296 single-board-bounce captures. Clean and single-bounce overlap.
  One `r5c7` attempt was withheld by the existing settling-correction IK guard.
  These filming counts are not held-out learned-policy performance.
- All 30 cycles were audited against full-rate 240 Hz native observations: 1,008,000
  all-cell samples and 1,049 valid scalar outcome reconstructions. The rejected
  attempt remains rejected. Whole-ball entry, board bounce and one continuous second
  of retention remain required, even when a quiet retention tail is cut visually.
- Exact camera, source hashes, stage values, contact/rim/isolation checks, native
  joint/ball readbacks, frame map and complete movie decode passed. Background contact
  visibility uses camera projection; selected RGB frames were also visually reviewed.
- The native contact sheet and encoded frames 0, 450, 900, 1410, 1649 and 1799 were
  inspected on Mac. Movie and full report transfer SHA256 values match Ubuntu.

The recorder exited successfully after 1,663.93 seconds. The 4090 was idle afterward;
the existing 4070 workload and dashboard PIDs were preserved. About 80 GiB remained.
No service restart, hardware connection, Viam configuration or audio change occurred.
243 local regression tests completed successfully, with eight dependency skips;
13 private launcher tests passed. The stage audit normalizes JSON list/tuple
serialization while rejecting any actual renderer-value or frozen-source change.

## Artifact locations and reproduction

Ubuntu raw evidence: `~/sircus-isaac/output/presentation-farm-20261104-v2`.
Export: `~/sircus-isaac/output/presentation-farm-20261104-v2-export`.
Mac copy: `.runtime/media/farm-presentation-20261104/` beneath the project directory.
`farm-one-minute.mp4`, `film.json`, poster/contact sheets, original RGB, native NPZ,
logs, source archives and failed startup receipts remain outside Git.

Recorder snapshot: `studies/presentation-a6f1578-v1`.
Exporter snapshot: `studies/presentation-export-e9d9f57-v1`.
Source menu/evidence: `output/expanded-training-20261102-v4`.
See [method and commands](../../sircus/isaac/PRESENTATION_FILM.md).
