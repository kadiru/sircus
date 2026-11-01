# Four-cell orthographic replay — November 1, 2026

Synthetic, bounded visualization run `overhead-replay-20261101T211407Z` on the RTX4090. Frozen settings come from `parallel-world-20261101T205548Z-n32`; this is not additional search or a new RL policy. The previous study and its results are unchanged.

| Cell | Bucket XY (mm) | Horizontal distance (mm) | Board-center Z (mm) | Ball-bottom release Z (mm) | Tilt | Replay |
|---|---|---:|---:|---:|---:|---|
| 00 | 470, −400 | 360.6 | 350 | 590 | 24° | Clean capture |
| 02 | 670, −400 | 300.0 | 350 | 550 | 24° | Clean capture |
| 03 | 870, −400 | 360.6 | 350 | 590 | 24° | Clean capture |
| 06 | 670, −442.9 | 342.9 | 350 | 590 | 24° | Clean capture |

All four endpoint checks passed. Each shot had exactly one board-contact episode, no detected external intermediate contact, whole-ball entry and one continuous second retained. All six contact/isolation probes passed. Bucket displacement remained below 0.000104 mm. These four selected nominal replays are visualization checks, not a robustness estimate.

`replay-summary.json` preserves source scene fingerprint, executed source hashes, source-report hash, the measured endpoint checks, each scored shot and hashes of the raw evidence. `video.json` records the 18.6-second movie's hash, exact playback speeds and contact timings. `SHA256SUMS` covers the compact evidence files.

Rendering snapshot on Ubuntu: `~/sircus-isaac/studies/overhead-replay-v3`. Source subsequently received formatting-only dict-literal cleanup and failure-handler lint annotation; physics/runtime logic is unchanged. Composition snapshot: `~/sircus-isaac/studies/overhead-video-v3`. Raw output: `~/sircus-isaac/output/overhead-replay-20261101T211407Z`. Credentials and raw observations are not included.

Six orthographic RGB views captured at 120 Hz from 240 Hz physics. The video shows 0.5× and 0.25× speed, with labelled simulator-position trails and fixed introductory/final holds. No interpolation. The first camera-framing attempts are retained separately; the earlier low camera included a neighboring table, corrected by tightening its far clip before the final replay.

The board is a provisional rigid attachment, bucket reported height175mm still has unresolved meaning, and ball properties are uncalibrated. Arms initialize at solved endpoints; there is no pickup or transit demonstration. See [rendering instructions and limits](../../sircus/isaac/OVERHEAD_REPLAY.md).
