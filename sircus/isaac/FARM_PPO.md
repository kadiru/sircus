# 32-table PPO training farm

This task adds actual neural-policy updates to the cloned Isaac Lab / PhysX workcells. One centralized cooperative policy chooses **board-center height, vertical drop gap and board tilt** for each of 32 distinct table-local bucket targets. Every workcell has two articulated UF850 arms, one board, one ball and one fixed bucket. Heading toward the bucket is prescribed by the simplified plane task.

The prior parallel-world study searched parameter combinations. This new run samples actions from a Gaussian actor, executes the throws in PhysX, scores the results, and updates actor/critic weights with clipped PPO. The movie records actual on-policy training rollouts that are subsequently used by the optimizer. Misses and unreachable endpoints are retained. It is not a montage of successful replays.

The [completed pilot, checkpoints and verified results](../../baselines/isaac-farm-ppo-20261101/README.md) are published separately. The selected policy scored 6/32 on the frozen test versus 9/32 for the fixed baseline; this pilot is not promoted as a successful general policy.

## Learning problem

- Inputs: five normalized privileged geometric features—relative bucket XY, horizontal distance, and sine/cosine of target-facing heading. No point-cloud or camera input yet.
- Outputs: height 300–450 mm, vertical ball-bottom gap 80–250 mm, tilt 12–40° from horizontal. A tanh transform bounds sampled Gaussian latents. Release XY stays over the board center; the supplied IK/settling controller implements each action.
- Episode: one decision followed by a throw, ending on capture, falling below the table, or four seconds. Arms are initialized at solved endpoints and must pass existing measured pose/velocity/drift gates. Pickup, gripper actuation, joint-trajectory learning and motion-clearance validation are separate.
- Algorithm: shared 5→64→64→3 actor and separate critic; tanh hidden activations. PPO ratio is computed for the stored latent; the common invertible action transform cancels from the ratio. Exploration uses a learned diagonal Gaussian scale. The entropy regularizer is latent-Gaussian entropy.
- Budget: 16 updates, four simultaneous 32-shot batches per update = 2,048 training throws. Four full-batch optimizer epochs per update, learning rate 0.0003, clip 0.2, critic weight 0.5, entropy coefficient 0.001, gradient clipping 1.0. Terminal one-step returns use observed reward, without bootstrapping/GAE. Seed 20261103. This is a bounded pilot, not a convergence claim.
- Initialization: a common H 350 mm / gap 220 mm / tilt 24° bias, zero actor output-layer weights and latent logstd −1.2. It uses engineering knowledge from earlier development; it is not learning from an uninformed random pose.

## Reward and unchanged success criterion

Invalid IK/settling earns −1; detected table/floor/robot/unattributed contact earns −0.5; no board bounce earns −0.25. Otherwise the reward is 0.10 plus a proximity term up to 0.25, retention up to 0.25, capture 1.0, clean-entry 0.10 and exactly-one-board-bounce 0.05.

Capture requires a board bounce, whole-ball entry and one continuous second retained, with no detected external intermediate contact. Rim-assisted captures count. The original capture, entry-quality and board-route fields are preserved separately; `ppo_reward` is new. These synthetic definitions are unchanged by the camera/learning adapter.

## Splits and evaluation

All 32 training bucket XYs remain fixed across throws. Validation/test each have 32 separate seeded coordinates, offset 18–36 mm in each XY axis from corresponding grid cells. Exact coordinates are disjoint; these are nearby spatial interpolation checks, not broad out-of-distribution or physical validation. Earlier studies already informed the region and action range.

Evaluate deterministic actor means initially and after updates 4/8/12/16. Select by validation captures, then mean reward, with earlier checkpoints winning exact ties. Freeze before evaluating on the test split. Report initial, final, selected and fixed-H 350 / gap 240 / tilt 24 baselines on the same test coordinates. There is no per-test-scene optimization. IK solves geometry only and cannot look at outcomes. A selected update 0 remains explicitly untrained rather than being relabelled as a successful learner.

## Training video and evidence

The wide orthographic camera records all 32 tables; three simultaneous insets show cells 00, 13, 31 (near/middle/far) without selecting examples by outcome. Record the first rollout of iterations 1, 8, 16 at 60 Hz from 240 Hz physics. Play at 30 fps (0.5×), with labelled holds. White trails use simulator ball centers. Counters report the iteration and throws collected before the filmed batch. **Arm endpoint resets occur between throws; the video does not demonstrate learned trajectories.**

`farm_report.py` checks the contract, disjoint splits, all episode actions/rewards, policy-update hash chain, checkpoint hashes, validation selection, final test rows, frame continuity and exact membership of each filmed rollout in the training data. `farm_checkpoint_audit.py` reloads the filmed policies and recomputes their sampled-action log probabilities and all frozen test actions on CPU. Tolerances account only for float32 CPU/CUDA/NumPy differences, never for robot pose acceptance. `farm_video.py` refuses a run that fails provenance checks.

## Run

Use the existing isolated 4090 Isaac Lab 2.3.2 / Sim 5.1 environment and generated assets. Mount a new immutable source snapshot and leave other GPU jobs alone. The run requires 3 GiB free storage and the existing scene lock.

```bash
export SIRCUS_ISAAC_SOURCE=/workspace/studies/farm-ppo-v2
# --smoke runs four cells, one update and four training throws first.
timeout 3600 flock -n /workspace/scene.lock \
  "$SIRCUS_ISAAC_SOURCE/python.sh" "$SIRCUS_ISAAC_SOURCE/farm_training.py" \
  --headless --enable_cameras --device cuda:0 --updates 16 --batches-per-update 4
```

The container's only visible GPU is the 4090. All work is synthetic; there is no Viam import or hardware connection. The separate 4070 workload is untouched. Existing dashboard/study pointers are not overwritten; new progress is `farm-training.json`.

After completion, use the matching source on `PYTHONPATH`:

```bash
python farm_report.py /path/to/farm-ppo-run /path/to/compact-export
# Run with the Isaac PyTorch interpreter, CPU-only:
python farm_checkpoint_audit.py /path/to/farm-ppo-run
# Host NumPy, Pillow>=9, DejaVu fonts and ffmpeg:
python farm_video.py /path/to/farm-ppo-run
```

Checkpoint loading uses `torch.load(..., weights_only=True, map_location="cpu")`, followed by `farm_policy.Policy.load_state_dict`. Require matching contract/version. Numerical/source evidence and small policies can be published under a dated baseline; raw frames, episodes, logs and video remain outside Git.

Provisional ball/material/rigid-board properties and the bucket's unresolved 175 mm height interpretation remain as recorded in `scene.json`. This pilot does not establish real-world deployment readiness. The PPO implementation follows the [clipped surrogate objective](https://arxiv.org/abs/1707.06347); the one-shot task and reward are specific to Sircus.
