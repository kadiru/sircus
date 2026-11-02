# Completed 32-target workspace feasibility study

Run `feasibility-20261102T012536Z-n32` completed on RTX 4090 in Isaac Sim 5.1.0 / Isaac Lab 2.3.2. The run began November 1 local time (November 2 UTC). It made **6,212 attempts**, including **3,043 accepted releases** and **3,169 endpoint/clearance rejections**, in 1,557.5 seconds after application setup. Contact probes are additional preflight checks, outside those counts.

This is a finite development parameter search, **not a new RL policy, held-out generalization result or exhaustive workspace coverage**. Arms start at solved joint endpoints and settle. Grasping, transit, self-collision-safe paths and physical deployment are not validated.

## Main findings

- 15 of 32 targets had a qualifying selected setting. All 15 captured 3/3 nominal repeats (45/45 total). Six targets passed 30/30 fresh local perturbations; the other nine ranged from 6/30 to 27/30. The remaining 17 targets are unresolved, not proven impossible.
- Confirmation across the 15 selected targets: **364/450 board-only captures**, including 338 single-board-bounce and 276 clean entries. Ten endpoint rejects and all other failures remain in the denominator. This is conditional on selected development targets, not an 80.9% success claim over the whole table.
- Six 30/30 targets: UI targets **13, 18, 20, 21, 28 and 29**. All 180 of those captures had one board bounce. Each 30/30 result has a Wilson lower 95% bound of 88.6%, not guaranteed reliability. Only narrow geometric perturbations were tested.
- One selected target lies beyond the dropper's conservative geometric reach bound: **target 23**. It captured only **6/30** perturbations, with zero single-bounce or clean captures. Its nominal replay has two board-contact episodes and measured rim contact. It establishes a fragile modeled possibility, not a dependable far-shot solution.
- Keeping selected heights/tilts but returning to original board XY produced 18/45 nominal captures, with 21 endpoint/clearance rejects and six external-contact failures. Heading was recomputed by the same bucket-facing rule. This is a small paired development ablation; unequal search budgets and selection effects prevent broad causal or policy-superiority claims.

## Exact robust settings

All positions/heights are millimetres in the dropper-base/table frame; tilt is degrees from horizontal. Ball-bottom release Z = board-center H + gap; ball-center Z adds the provisional 37.5 mm radius. Heading is prescribed toward the bucket. Each row below captured 30/30 fresh perturbations, all with exactly one board bounce.

| UI target | Bucket X/Y | Board X/Y | Board H | Ball-bottom Z | Tilt | Board–bucket distance | Clean / 30 |
|---|---|---|---:|---:|---:|---:|---:|
| 13 | 700/-470 | 670/-100 | 325 | 565 | 30 | 371.2 | 29 |
| 18 | 100/-220 | 520/-150 | 325 | 625 | 32 | 425.8 | 30 |
| 20 | 500/-220 | 620/150 | 325 | 585 | 32 | 389.0 | 30 |
| 21 | 700/-220 | 620/150 | 300 | 560 | 30 | 378.5 | 30 |
| 28 | 500/230 | 670/-100 | 325 | 565 | 30 | 371.2 | 29 |
| 29 | 700/230 | 670/-100 | 400 | 600 | 32 | 331.4 | 14 |

[All 32 targets and all selected settings](settings.csv), including unsuccessful targets, are preserved separately from historical baselines. UI target numbers are one-based; artifact IDs are zero-based (`target-22` is UI target 23).

The fragile beyond-reach example uses bucket (1100, −220), board (620, −150), board H400, gap180, ball-bottom Z580, ball-center Z617.5 and tilt16°. Board-to-bucket distance is 485.1 mm. The bucket center is 1121.8 mm horizontally from the dropper pivot versus the conservative 1026.6 mm ball-center reach bound. Even the closest whole-ball entry point is 1070.8 mm away (inner radius88.5 − ball radius37.5 =51 mm opening allowance). Thus the entire vertical whole-ball entry region lies beyond that outer bound in this model. Throwing the ball horizontally is not included in this direct-drop comparison.

## Search and scoring

See the [complete frozen protocol](../../sircus/isaac/FEASIBILITY_MAP.md). Attempt counts: 896 fixed-XY, 3,912 expanded-XY, 864 refinement, 45 nominal, 45 fixed-XY ablation and 450 perturbations. The action is board X/Y, height, drop gap and tilt; heading faces the bucket. Selected settings were frozen before all repeats, with no confirmation-driven reselection.

Perturbations: board X/Y ±3 mm, board H/gap ±2 mm, tilt ±0.5°. Bucket positions and contact parameters stay fixed. Capture requires board rebound, whole-ball entry and one uninterrupted second inside. `sircus-board-route-v1` rejects detected table, floor, robot or unattributed contacts; rim-assisted capture is allowed. Exactly one board bounce and clean entry are separate measures.

Across all stages, 335 legacy captures were excluded by the external-contact rule, including 119 at beyond-bound targets. Several apparent far successes measurably bounced off the table first. These are not counted as board-only success. No historical result has been relabeled.

Endpoint rejects: 1,674 dropper IK, 1,332 board IK, 33 settling-correction IK, 37 not settled, and 93 board/bucket separation guards. The largest accepted board-center error was 1.994 mm, normal error0.0244°, joint speed0.0000914 rad/s, and dropper-center error0.496 mm, within unchanged gates. Sampling failure or numerical IK failure does not establish impossibility.

## Evidence and verification

- [report.json](report.json): compact per-target counts, selected actions, representative measured trajectories, contract, source hashes and direct-reach audit.
- [verification.json](verification.json): 6,212 raw/summary records matched, 63 immutable executed-source hashes and four evidence hashes verified; exporter checksum and aggregate settling checks. Local checks deliberately corrupted the scene fingerprint, repeat seed, frozen setting and retention; all four were rejected.
- [input-assets-verification.json](input-assets-verification.json): 80 generated input assets matched by hash, size and modification time. The initial audit was captured during execution, not before launch; every mtime predates the run, and a post-run check found no changes.
- All five per-cell attribution probes and foreign-cell collision isolation passed. Maximum bucket drift was 0.000597 mm.
- Raw JSONL, full trial histories, generated USD/assets and logs remain outside Git. The exporter verifies those on the experiment host. Representative trajectories are measured PhysX samples, not generated visual footage.

The verifier preserves the exact original scene fingerprint. Derived heading/distance comparisons allow only 1e-12 numerical tolerance before verifying that original hash, handling a one-ULP Linux/macOS `atan2` difference. This does not alter poses or runtime acceptance gates.

Geometry remains measured/provisional as documented: landscape board200×300×5 mm, base spacing1337.31 mm, bucket177/150 mm inner and200/158 mm outer diameters. The **175 mm value is still only reported height**; overall height versus interior depth is unresolved. Current depth175 + bottom8 mm is an approximation. Contact, mass, rigid mounting and tool proxy remain provisional.

## Interface and next step

The existing read-only preview serves `/feasibility.html`: clickable 32-target map, exact settings, target-specific failure reasons, confirmation rates/intervals, paired original-XY comparison and half-speed replay of measured top/flight-plane trajectories. The existing native Isaac videos remain separate. No robot command API is exposed.

Do not begin another long PPO run from these results alone. Next, run a bounded robust far-target search around target23 and neighboring far targets, using expected single-bounce capture under perturbations, reachability-conditioned proposals and measured outgoing ball trajectories. Avoid refining table-assisted legacy success merely because its shaping reward is high. Then create fresh spatial train/validation/test layouts and train a scene-conditioned board-XY/H/gap/tilt policy; evaluate a frozen checkpoint without per-scene search. Quasi-perception and grasp/path learning remain later stages.

The finite 4090 job exited normally. The read-only preview remains running, and the existing 4070 process was untouched. No physical hardware, Viam, audio or other services were changed. The final headless suite ran111 tests (110 passed; one Torch case skipped on Mac and covered by five passing Ubuntu CPU farm tests). All five feasibility DOM tests passed; the complete map and replay were inspected in Chrome. The earlier full physical-mock/numerical/JS regression also passed. See verification.json and the accompanying PR.
