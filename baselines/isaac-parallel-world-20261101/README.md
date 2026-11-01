# 32-cell Isaac world — November 1, 2026

One PhysX scene on the RTX 4090: **32 tables, 64 articulated arms, 32 distinct table-local bucket targets**. Each bucket stays fixed across trials. Board/release XY remain (670, −100) mm, and the board faces its own bucket. This is endpoint-initialized parameter exploration, not a new RL policy, arm-path validation or physical evidence.

Run `parallel-world-20261101T205548Z-n32` completed 1,090 attempts in **197.3 seconds** after application launch: 64 controls, 896 search attempts (28 per cell), 130 fresh confirmations. Search: 133 qualified captures, 102 single-board-bounce captures, 26 clean entries, 123 IK rejections. Confirmation: 107/130 qualified captures, all 107 single-board-bounce, 67 clean, 40 assisted; 17 IK rejections and 6 released misses.

Six targets passed 10/10 fresh perturbed shots each. Thirteen cells had a qualifying searched setting; 19 had none in this finite search. The farthest selected target is 440 mm away; no qualifying setting was found farther than 440 mm. Ten repeats are a small sample, not a reliability guarantee.

| Cell (UI table) | Bucket X/Y mm | Distance mm | Board H / gap mm | Tilt | Fresh captures | Clean | Invalid |
|---|---|---:|---|---:|---:|---:|---:|
| cell-00 (1) | 470 / -400.0 | 360.6 | 350 / 240 | 24° | 10/10 | 10 | 0 |
| cell-01 (2) | 570 / -400.0 | 316.2 | 350 / 240 | 24° | 10/10 | 1 | 0 |
| cell-02 (3) | 670 / -400.0 | 300.0 | 350 / 200 | 24° | 10/10 | 6 | 0 |
| cell-03 (4) | 870 / -400.0 | 360.6 | 350 / 240 | 24° | 10/10 | 10 | 0 |
| cell-04 (5) | 470 / -442.9 | 396.9 | 400 / 240 | 24° | 7/10 | 0 | 3 |
| cell-05 (6) | 570 / -442.9 | 357.1 | 350 / 240 | 24° | 10/10 | 10 | 0 |
| cell-06 (7) | 670 / -442.9 | 342.9 | 350 / 240 | 24° | 10/10 | 10 | 0 |
| cell-07 (8) | 870 / -442.9 | 396.9 | 400 / 240 | 30° | 7/10 | 7 | 3 |
| cell-09 (10) | 570 / -485.7 | 398.5 | 400 / 240 | 30° | 7/10 | 7 | 3 |
| cell-10 (11) | 670 / -485.7 | 385.7 | 350 / 240 | 30° | 8/10 | 6 | 0 |
| cell-11 (12) | 870 / -485.7 | 434.5 | 400 / 240 | 30° | 8/10 | 0 | 2 |
| cell-13 (14) | 570 / -528.6 | 440.1 | 400 / 240 | 30° | 7/10 | 0 | 3 |
| cell-14 (15) | 670 / -528.6 | 428.6 | 400 / 240 | 24° | 3/10 | 0 | 3 |

Ball-bottom release height = H+gap; ball-center adds 37.5 mm; provisional dropper-flange offset adds 95 mm. All heights use tabletop Z=0. Fresh trials perturb H/gap ±2 mm and tilt ±0.5° with frozen nominal settings and new declared seeds. IK rejects count in the denominator. At H400/gap240 mm, slight positive height perturbations can fail dropper IK; do not call those released misses.

The bucket retains 177/150 mm inner and 200/158 mm outer diameters. **Reported height 175 mm remains ambiguous**; the model retains 175 mm assumed interior depth plus 8 mm bottom. Board 300×200×5 mm, landscape; rigid mount/material/ball parameters remain provisional.

Validation:

- All 32 buckets remained at assigned positions: maximum displacement 0.000469 mm (floating-point readback scale). Five per-cell contact probes and the deliberate foreign-cell isolation probe passed.
- All 1,090 compact records matched raw trial rows; declared actions/seeds, completed-batch counts, settling gates and score consistency passed. Three artifact hashes and 48 executed Python source hashes matched. Raw observations and generated assets stay outside Git.
- One-cell and four-cell reference runs passed their contact checks. Cell 0 control H350/gap240/tilt24 captured in 1/4/32-cell runs. Borderline H350/gap200/tilt24 captured in 1/4 cells but missed in 32. Translation/batching sensitivity has not been isolated; do not claim bitwise clone equivalence.
- The first search batch has 121 matched overview/close-camera frames, encoded at 30 fps (4.03 s). It includes successes and misses; it is not a best-shot montage. Video SHA256 is recorded in summary.json; media stays in the local run directory.
- UI selection, stale/mismatched-media handling, loopback route allowlist, raw-file/traversal rejection and blocked control requests were tested. GPU jobs exited naturally; 4070 workload was untouched.

Files: [summary](summary.json), [all 32 target settings CSV](settings.csv), [reference checks](reference-checks.json), [method and launch instructions](../../sircus/isaac/PARALLEL_WORLD.md).

Executed source snapshot: `parallel-world-v4`; the final task branch adds report validation, UI polish, video tooling and documentation without changing those completed physics results. Existing baselines remain unchanged.
