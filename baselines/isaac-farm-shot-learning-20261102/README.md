# Broad 35-target search and longer PPO study

Native Isaac Sim 5.1 / Isaac Lab 2.3.2, RTX 4090. The original 35 distinct bucket coordinates, two arm bases, measured/provisional geometry and capture scorer are unchanged. This study expands shot search and learns a shared candidate-ranking policy. It does not establish unseen-position generalization or learned grasp/joint-path control.

**Decision: do not promote the policy as a dependable deterministic controller.** Stochastic selection improved substantially against its initial policy on the same candidate bank, but the highest-probability choice still captured at only 10/35 targets. The historical policy captured at 14/35 in its separate greedy comparison. Nineteen search-selected recipes are stronger nominal controls; a searched recipe is not a learned policy.

## Completed results

| Stage | Captures / attempts | Interpretation |
|---|---:|---|
| Broad development sweep | 306 / 1,607 | 1,236 releases, 371 rejected endpoints/settling checks; not policy evaluation |
| Search-selected repeats | 76 / 140 | 19 sites each passed 4/4; the remaining 16 did not capture |
| Uniform initial policy, sampled | 96 / 560 (17.14%) | Same broad native bank as the trained policy |
| Trained policy, sampled | 183 / 560 (32.68%) | Frozen final weights; 16 fresh episodes per site |
| Historical policy, sampled | 169 / 560 (30.18%) | Its original narrower bank; not a controlled architecture-only comparison |
| Initial / trained policy, greedy | 10 / 35 → 10 / 35 | One episode per site, not a reliability estimate |
| Historical policy, greedy | 14 / 35 | One episode per site |

The 8,960 on-policy training attempts produced 8,959 releases, one rejected release and 2,503 captures. There were 128 PPO updates. Training counts are separate from the frozen tests above. The initial versus final sampled comparison uses the same candidate bank and paired sampling seed; this is one training seed at the same 35 development layouts, not a replicated or disjoint-layout benchmark.

Final sampled quality: 35 clean captures and 166 exactly-one-board-contact captures. Historical sampled quality: 76 clean and 151 exactly-one-board-contact captures. These overlap with total captures and must not be added. The new policy's small aggregate advantage over the historical sampled score does not establish overall superiority; clean capture and greedy choice remain weaker.

## Per-table frozen final results

Every entry is captures out of 16 sampled episodes. Row/column labels identify bucket grid sites, not camera display-slot order.

| | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 | 6 | 7 | 15 | 8 | 15 | 0 | 0 |
| R2 | 1 | 15 | 10 | 12 | 0 | 0 | 0 |
| R3 | 0 | 16 | 5 | 12 | 5 | 0 | 0 |
| R4 | 0 | 12 | 15 | 16 | 12 | 0 | 0 |
| R5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |

Search discovered captures at four previously zero-scoring sites: `r1c1`, `r2c1`, `r2c3` and `r2c5`. All four selected settings passed 4/4 nominal repeats. The trained sampled policy subsequently captured 6/16, 1/16, 10/16 and 0/16 respectively. The last case is a concrete failure to select an available successful shot.

| Previously unscored site | Board H, mm | Ball-bottom gap, mm | Board tilt | Horizontal distance, mm |
|---|---:|---:|---:|---:|
| r1c1 | 321.99 | 451.11 | 28.42° | 560.06 |
| r2c1 | 368.12 | 479.55 | 39.13° | 545.73 |
| r2c3 | 404.82 | 207.29 | 38.80° | 354.37 |
| r2c5 | 320.00 | 135.00 | 24.00° | 400.00 |

Ball-center release height is H + gap + 37.5 mm; the flange is a further 95 mm above that center. Exact XY, headings, every stage's per-table counts and selected settings are in [per-table.csv](per-table.csv) and [the native report](farm-study-result.json). [Parameter coverage CSV](parameter-coverage.csv) and [JSON](parameter-coverage.json) distinguish available native-qualified ranges from the configurations actually sampled during training at every table; circular bearing span handles the ±180° wrap. Repeating four identical simulated shots is not a guarantee under placement, material or calibration noise.

## How variety and learning work

Five task variables replace blind exploration of twelve arm joints: bucket-relative board bearing, board-to-bucket distance, board-center height, drop gap and tilt. The board faces the bucket, release XY stays above its center, and the dropper points downward. Original bounds are H 140–540 mm, gap 80–560 mm, tilt 12–52°, distance 280–1200 mm and bearing offset ±180°. Reach and clearance make the admissible set much smaller than this numerical box.

CPU search mixes inverse-ballistic proposals with independent seeded Halton samples, screens up to 512 IK checks per site, and retains up to 128 candidates. Board centers range across the original table with a 15 mm center margin; a held board may overhang it. Actual FK checks its center and bottom clearance despite unconstrained in-plane spin. This produced 1,607 distinct per-site proposals, of which 1,236 passed native endpoint tests. All 930 accepted misses remain in the training bank alongside 306 successful sweep cases; outcome labels do not filter the action bank.

The shared categorical PPO actor scores 11 features: five privileged bucket features plus six shot features, including sine/cosine of bearing. A masked distribution samples among valid candidates. It starts uniformly, learns from terminal shot rewards, and uses a separate critic. There are no table-ID embeddings or point-cloud inputs. No capture optimizer runs during frozen inference, but a geometrically qualified candidate bank is still required. This is a ranking relationship over a finite bank, not an unconstrained continuous inverse-physics solution.

The new versioned reward preserves graded proximity/retention feedback for misses and penalizes external routes. Capture bonuses still require the unchanged board bounce, whole-ball entry, uninterrupted one-second retention and no external-contact route. Rim-assisted captures count; partial exits reset retention. The historical flat −0.5 external-contact reward and its old results remain intact. Restitution priors in proposal generation do not randomize native materials: simulation still uses the provisional 0.65 ball restitution and 0.2 friction.

## Reach and energy diagnosis

[Compact CPU reach diagnosis and reproduction evidence](reach-audit-summary.json) investigate why simply widening height limits may not help the far sites. The downward tool requires the wrist 90 mm above the flange, in addition to the 95 mm flange-to-ball offset. At extended horizontal reach, that restricts available release height and energy. The other arm must simultaneously hold the tilted board.

These are finite-grid, frictionless-model diagnostics with strict IK checks, not impossibility proofs. A residual at the rim-plus-radius approach plane is not a whole-ball capture criterion; the diagnostic separately considers three heights and the opening region. Some far poses still need better coverage, and changing wrist orientation or the task's static-board assumptions would require a separate versioned experiment.

A separate [109-option probe bank](reach-probe-bank.json) preserves 70 historical controls and adds 39 analytically derived configurations for the ten C6/C7 sites. The first `r2c6` proposal uses H260 mm, gap412 mm, tilt24° and distance517.5 mm. It is an IK-valid proposal, not a claimed capture. Its no-retraining native test **did not execute**: another agent held the shared 4090 lock for the entire 900-second queue wait, which exited before launching Isaac. No queued probe remains. [Bound status](transfer-probe-status.json), [verification](verification.json). Its source and CPU audit tests are ready; native validation remains outstanding. No main-study metric includes these untested proposals.

## Evidence, reproducibility and limits

- [Study contract](farm-study-contract.json), [scene/action contract](contract.json), [native bank](native-qualified-study-bank.json), [proposal protocol](sweep-proposal-protocol.json).
- [Initial checkpoint](candidate-policy-initial.pt), [experimental final checkpoint](candidate-policy-final.pt): load with `weights_only=True`, CPU first, using the candidate-policy architecture. Historical Gaussian checkpoints are incompatible. Do not promote this checkpoint on the basis of training reward or sampled mean alone.
- [CPU checkpoint audit](candidate-checkpoint-audit.json), [independent audit](independent-audit.json): all 439 batch and per-cell summaries, 8,960 training likelihoods, 128 update links, 1,190 categorical frozen-evaluation likelihoods and 70 greedy choices checked. Maximum likelihood error was 1.91e-6. Historical evaluation did not save Gaussian latents, so its likelihoods were not independently reconstructed.
- [CPU proposal audit](cpu-bank-audit.json): exact workers, hashes, assignments, original buckets and strict IK/FK for all 1,607 proposals. [Native probes](probes.json), [rim rays](rim-rays.json), [original camera](selected-camera.json), [binding](camera-binding.json), [verification](verification.json), [manifest](manifest.json).

The main run used immutable source `studies/farm-shot-study-20261102-v4`, matching all 97 Python hashes in source commit `d2a9b70e1fb1ac087f311d1a05ee774b54e82a64`. Native source archive SHA256: `177f2187c33102a0c4405f8f3d4451044e1cc537dfc11c3fe00836763be480ad`. CPU search used v3; historical partial searches remain separately preserved. The later transfer-probe adapter and report additions do not rewrite the executed main source.

Study fingerprint: `dde5280aedf1f41a441f62f87abfdb9e3d43430c85469d27323f04044d92a35c`; unchanged scene/action fingerprint: `63362ef37a93881f769b39d1ee7456335f1cc7bf5db04ffa7e548ce9697455bf`.

The bounded native loop completed in 2,110 seconds, excluding application startup and CPU proposal generation. Raw stage records, intermediate policies, source archives and logs remain outside Git under the configured Ubuntu output/studies directories; the portable baseline keeps compact verification and initial/final checkpoints. No raw camera observations or credentials are included. Final storage check found about 3.8 GiB free; preserve evidence and plan storage before a substantially larger run.

Repository mock/fake-only regression passed, including 266 Isaac tests with ten expected Mac Torch skips; actual Ubuntu policy and frozen-transfer CPU tests passed. All 14 task Python files passed Black. Native endpoints, captures and data identity were tested; optimizer arithmetic and full contact traces were not independently reconstructed. No physical hardware, Viam configuration or audio work occurred. The existing 4070 workload and unrelated services were preserved. The read-only Mac report runs separately on loopback port 18785.

[Detailed methodology and commands](../../sircus/isaac/FARM_SHOT_LEARNING.md). Before claiming generalization, freeze a policy and test disjoint bucket locations without outcome-based optimization at inference. Broader physics/material randomization and perception-derived inputs remain separate work.
