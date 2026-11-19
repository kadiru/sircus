# Clearance-triggered wider search: 50 fresh positions

Completed November 16, 2026 Pacific / November 17 UTC. Synthetic mjlab evaluation at 64 substeps; no model retraining.

| Planner | First-drop captures | Feasible setups | Flight misses | Invalid physics | Robot-contact shots |
|---|---:|---:|---:|---:|---:|
| Current planner | 40/50 (80%) | 50/50 | 9 | 1 | 7 |
| Wider clearance search | 42/50 (84%) | 50/50 | 7 | 1 | 6 |

The wider search gained **two captures and lost none** on the same 50 positions. Both captured at 40; neither captured at eight. The 38 identical-command pairs had identical outcome classifications. One of those pairs was numerically invalid: the trace was flagged `through_wall_after_entry`. It is counted as a failure in both denominators.

The new search triggered at 13 positions and changed 12 commands. It prepared 416 extra geometric proposals (98 admitted), then 52 fixed-target board-arm branch searches (108 admitted branches). Nine final commands came from alternate board-arm branches; three from expanded geometry. Both planners used exactly one final native drop at each position.

## Where it helped

| Bucket X, Y (mm) | Current result | Wider-search result |
|---|---|---|
| 951.0, −591.7 | Table-bounce flight missed after rim contact | Captured after a table bounce |
| 991.5, −133.4 | Ball hit the robot after the board bounce | Direct capture; no robot contact |

The first gain selected an alternate joint branch of a different geometric shot. This comparison does **not** isolate the effect of changing joints alone. The second gain came from expanded geometry. The unchanged numerical-invalid case was at (734.2, −245.3) mm.

## Cost and limits

Median CPU planning time increased from **33.7 to 39.5 seconds**. The wider planner's 90th percentile was **275.2 seconds** and maximum **383.0 seconds**, versus 97.3 and 207.6 seconds for the current planner. These are per-position worker wall times in this run, including candidate checks, rather than guaranteed interactive response times. Four CPU workers prepared the comparison. The native GPU lease lasted **269.6 seconds**.

The gain is modest: six robot-contact shots, one additional flight miss, and one invalid-physics result remain for the wider planner. This single comparison does not establish improvement everywhere. The live interactive planner has not been switched or restarted. Results are visible at `http://127.0.0.1:18781/#clearanceExpansionStudy`.

Both nine-shot control sets passed. The independent CPU audit passed **40,635 checks**, binding the scene, source, fresh layouts, frozen commands, arm paths and native scalar scoring. The 4070 and Isaac setup were unchanged.

## Reproduction and evidence

- [Protocol and runner](../../sircus/mjlab/CLEARANCE_EXPANSION.md)
- [Aggregate results and all 50 synthetic position outcomes](summary.json)
- Producer implementation: `40d55ee893661ce656e76a84c0df38f5f33bf6df` (planner/runner bytes match the frozen source manifest).
- Private Ubuntu source snapshot: `~/sircus-mjlab/studies/clearance-expansion-v1`.
- Native evidence: `~/sircus-mjlab/output/clearance-expansion-v1`; independent receipt beside it as `clearance-expansion-v1-audit.json`.
- Choices SHA-256: `54fefc386aacffda9a155226061cd2897b538afb5c7888a3e8e5d6067485ab2e`.
- Report SHA-256: `caab4df627a5d03d04f8609a7116c58a9f49dcbec0cf08056af23582b81b577d`.
- Evidence-manifest SHA-256: `c793e371be1b594cd3758c41f6109e0f89368bffa77f991b42ebfcd6e085e377`.
- Audit SHA-256: `2bf16609f5d3e404235a05431908a5d356509c308b0dbf8fdaa3431e54a14eb1`.

Raw trajectories, generated runtime configurations, logs, hardware calibration and credentials remain outside Git. Historical results and successful saved recipes were preserved. Native physics holds the selected arm endpoints; CPU path proofs do not establish a dynamically executed robot transit.
