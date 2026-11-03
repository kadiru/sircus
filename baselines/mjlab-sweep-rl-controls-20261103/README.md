# Current-ball sweep/RL engineering controls — November 3, 2026

These completed native controls gate the separate matched-budget benchmark.
They are engineering checks on TRAIN recipes, not learning results or a held-out
performance estimate. Historical baseline files remain unchanged.

- All 17 exact shared source recipes captured under the corrected native scorer.
- 50 requested decisions produced 48 native releases; a deliberate invalid pose
  was rejected before native flight. Rejected decisions remain in the ledger.
- 32 discarded PPO smoke decisions produced finite, varying rewards and changed
  the actor. Their model and outcomes are not reused for benchmark training.
- Independent CPU audit passed 6,881 checks, including native scalar scoring,
  evidence/proof identities, likelihoods and optimizer replay.
- Controls took 412.5 seconds. The separate independent audit took 1.5 seconds.

Execution-critical source matches commit
`b92b5a14ae113452e03f797e60139a017e52e14e`. Source/scene/asset hashes are
recorded in the JSON evidence. Full native arrays and local operational logs
remain outside Git; this compact bundle alone does not rerun their audit.

The experiment uses fixed current-ball materials, privileged geometry and held
articulated endpoints with sampled setup paths. It does not validate driven arm
motion, grasping, impact compliance, perception or real hardware. The bucket's
175 mm reported height remains uncertain; interior depth is still provisional.

`controls.json` and `audit.json` are immutable outputs. Verify this bundle with
`sha256sum --check SHA256SUMS` (Ubuntu) or `shasum -a 256 -c SHA256SUMS` (Mac).
The benchmark's learning curves and final test are separate evidence; see
[the protocol](../../sircus/mjlab/SWEEP_RL_BENCHMARK.md).
