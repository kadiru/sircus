# 32-table PPO pilot — November 1, 2026

Actual Isaac Lab 2.3.2 / Isaac Sim 5.1 / PhysX training on RTX 4090. One shared actor/critic chooses board height, drop gap and tilt for 32 simultaneous, collision-isolated tables with 64 articulated UF850 arms. Each table's bucket XY remains distinct and fixed during training. This is a new learning run, separate from the earlier parameter search and four-cell frozen replay.

Run: `farm-ppo-20261101T212820Z-n32`; immutable source snapshot: `farm-ppo-v2`. The finite job completed in 518.24 seconds after scene initialization, including recorded rollouts and evaluation. There were 2,048 training throws, 16 PPO updates, 160 validation throws and 128 frozen test throws, plus separate contact probes. This bounded single-seed pilot did not establish a successful general policy.

## Frozen test comparison

All policies use the same 32 test coordinates, disjoint from the 32 training and 32 validation coordinates. Each coordinate is tested once per policy. These are nearby spatial interpolation checks within the development region, not physical or broad out-of-distribution validation.

| Policy | Captures | Clean entries | Near (<400 mm) | Middle (400–500 mm) | Far (≥500 mm) |
|---|---:|---:|---:|---:|---:|
| Initial engineering bias | 7/32 | 3 | 5/7 | 2/11 | 0/14 |
| Final, update 16 | 7/32 | 1 | 6/7 | 1/11 | 0/14 |
| Validation-selected, update 12 | 6/32 | 1 | 5/7 | 1/11 | 0/14 |
| Fixed H350 / gap240 mm / tilt24° | 9/32 | 1 | 7/7 | 2/11 | 0/14 |

All 128 test endpoints passed IK/settling. All reported test captures had exactly one board bounce. Capture requires a board bounce, whole-ball entry and one continuous second retained, without detected external intermediate contact; rim-assisted captures count. The selected policy improved validation from 8/32 to 14/32 but did not improve the frozen test. Do not promote it over the fixed baseline or infer superiority from the small samples. No policy achieved a far test capture.

## Video provenance

`sircus-32-table-ppo.mp4` records the first training batch of iterations 1, 8 and 16. The corresponding policies had completed 0, 7 and 15 updates, with 0, 896 and 1,920 previous training throws. Those 96 filmed actions were subsequently consumed by PPO. Batch capture counts were 6/32, 8/32 and 8/32; unreachable endpoints and misses were retained.

The wide orthographic view shows all 32 tables, alongside simultaneous near/middle/far close-ups selected by cell index rather than outcome. Each camera recorded 241 samples at 60 Hz. The 2560×1600, 30 fps movie plays the trajectories at half speed with one-second endpoint holds. White trails come from recorded simulator ball centers. Its 30.1-second duration includes those holds. Raw frames, episode logs and the movie remain outside Git; `video.json` identifies the verified artifact.

This is **high-level shot selection**, with known simulator geometry and endpoint IK initialization. The video does not demonstrate learned arm transit, pickup, gripper actuation, perception-based control or real-world transfer. Bucket-facing heading is prescribed; the policy learns only height, gap and tilt. Provisional ball/material/mount values and the unresolved meaning of the bucket's reported 175 mm height remain unchanged.

## Published evidence

- `summary.json`: contract, all optimizer/validation summaries, frozen test comparison, distance bins and provenance checks.
- `splits.json`: exact 96 seeded coordinates and features; training distances span 300–632 mm.
- `initial.pt`, `latest.pt`, `selected.pt`: small synthetic checkpoints, preserving the initial, final and validation-selected policies. `latest.pt` also includes the final optimizer; `selected.pt` is evaluation-only. Load trusted files with `weights_only=True`, and require a matching contract fingerprint.
- `checkpoint-audit.json`: all 96 filmed sampled-action log probabilities recomputed from stored weights; all 96 neural-policy test actions recomputed on CPU. The fixed baseline is checked separately by the report.
- `verification.json`: source/evidence verification and runtime boundaries.
- `video.json` and `SHA256SUMS`: media identity and published-file integrity.

The audit verified all 2,048 training rows, the full weight-update hash chain, actual actor changes, exact training membership of the filmed batches, disjoint splits, validation selection, checkpoint hashes and contact probes. All 55 executed source-file hashes matched the immutable snapshot. Maximum bucket drift was 0.000611 mm. Five contact attribution probes passed in every cell; the foreign-cell isolation probe passed. The one-/four-cell parent controls are historical; this run does not newly establish strict equivalence across clone counts.

Tests: 101 local Isaac tests ran successfully, with the PyTorch-specific test skipped on the Mac; all five new tests passed in the Isaac container, including an actual gradient update. A four-cell physics/PPO smoke and the full 32-cell run passed their evidence/checkpoint audits. Video decode and frame verification are recorded in `verification.json`.

Reproduce with [FARM_PPO.md](../../sircus/isaac/FARM_PPO.md). The existing 4070 workload and preview service were left running; no physical, Viam or audio changes were made. No long training run is left active.

## Next learning boundary

Far test shots all received the external-contact penalty. Before a longer run, distinguish insufficient ballistic range from action/reachability limits, then investigate the reward's flat penalty on failed far shots. Preserve the strict success rule; use development data for any controller/reward change, and a fresh final test set afterward. Perception-conditioned policies and full joint trajectories remain separate work.
