# Fixed-camera training and action-only video

The selected camera can record a new PPO run directly, rather than replaying selected historical actions. Every batch is filmed from the same saved eye, aim and lens, including driven arm setup and the ball's release, rebound and immediate outcome. A shorter export removes settling waits and quiet tails, while the full scoring rollout remains in the evidence.

[Completed ten-batch run, videos and experimental checkpoint](../../baselines/isaac-action-training-20261102/README.md): 23.33-second action edit and 52.33-second complete captured sequence. The foreground has ten rebounds and two captures; no batch was discarded for its outcome.

## Why some arms remain parked

The previous 35-table source run found geometry-qualified board/dropper settings at 17 grid sites. The other 18 sites had no accepted option in its finite 60-proposal geometry screen. This is a failure to find a supported setting within that menu, not proof that the complete tabletop is unreachable. Repeated menu choices also produce little or no visible joint motion. Arms hold their poses during ball flight.

This filming pass keeps those unsupported sites parked with releases disabled. All 35 sampled slot actions and their failure reasons remain in the training records; enabled attempts and actual releases are counted separately. It does not invent successful endpoints or move buckets just to animate every table. The model still receives the original invalid-attempt penalty. All samples remain in the original two-batch PPO update buffer.

## Training scope

One shared Gaussian actor/critic samples four high-level shot parameters: board bearing around the bucket, board height, release gap and board tilt. A fixed geometry-only menu projects its latent proposal to a qualified endpoint configuration. The original geometry, reward and whole-ball/one-second capture evaluator remain unchanged. A fresh run uses ten batches of 35 sampled slots and five PPO updates. This is a bounded training recording, not a convergence or held-out evaluation study.

The controller drives both articulations between configurations, with original effort limits during transit and the existing final measured pose/speed/drift release guards. These transitions and ball initialization are scripted. PPO does not learn pickup, gripper actuation or joint trajectories here. The selected camera does not observe or steer the policy; policy inputs remain privileged scene geometry, not point clouds.

## Record on the configured 4090

Freeze the source into a fresh `studies/` directory before launch. Use the existing Isaac Sim 5.1 / Isaac Lab 2.3.2 container, generated assets and scene lock; do not restart another service or GPU workload. Copy the original chosen camera JSON into the workspace. With a fresh output directory, inside that container:

```bash
python.sh film_cylinder_farm.py --headless --enable_cameras --device cuda:0 \
  --appearance silver \
  --training-camera-json /workspace/selected-camera.json \
  --geometry-menu-source /workspace/output/cylinder-film-35-v5 \
  --output /workspace/output/action-training-new
```

The menu source must match the original source fingerprint and its recorded geometry-only preflight checks. No capture outcomes from that source select training actions. The complete original camera file is preserved and checked. Whole-cell origins use the same right-front permutation as the chosen camera scout; targets remain attached to their source cells.

## Editing without changing scores

Each batch retains arm setup, ball descent and rebound. A bucket contact alone is not a miss: a rim-assisted capture may follow it. The editor uses full-rate simulator telemetry and the final scorer result to choose a short visual ending after entry, subsequent table/floor contact, explicit workspace exit or timeout. The complete retention scoring remains available even if a visual cut removes its stationary tail. No batch is discarded for failure, and no trial is repeated to obtain a better outcome.

The action edit and full recorded sequence use the same native RTX frames. Per-batch metadata records retained source intervals, camera poses, measured joints, training-policy hashes and observed outcomes. Physics/release sampling and compressed setup timing must be reported separately; neither file is labeled real-time arm control or evidence of policy improvement.

After capture, verify the sampling checkpoints on CPU with the matching snapshot, then encode on the Ubuntu host with NumPy, Pillow and FFmpeg:

```bash
python.sh action_training_audit.py /workspace/output/action-training-new
python3 action_training_export.py output/action-training-new \
  output/action-training-new-export --untrimmed
```

The audit reloads each sampling checkpoint with `weights_only=True` on CPU and recomputes all 350 stored latent log probabilities. It binds ten clips, eleven checkpoint digests and the five logged weight updates. This verifies the sampling chain; it is not an independent optimizer reconstruction or held-out evaluation. The exporter binds this audit, checks every batch's actions/rewards/native camera, reconstructs foreground scoring at 240 Hz, and decodes the resulting videos to check frame and chapter counts.

`action-training.mp4` is the action-only edit, retaining every batch. `action-training-untrimmed.mp4` preserves every captured frame, including stationary retention tails, while setup/settling/optimizer waits that were never filmed remain omitted. Release frames sample every eight 240 Hz physics steps and play at 30 fps (1× ball-flight time). Setup uses 36 measured joint frames scheduled across each scripted interpolation, so its timing is compressed rather than real-time control. Camera sampling does not require relative joint error to vanish; final release still requires the unchanged measured readiness and settling checks. `film.json` records the exact retained source intervals and terminal-cut reason for every batch. Raw force/position telemetry and checkpoint audits remain with the full local run.

The earlier one-minute replay, twelve-drop silver export, training checkpoints and historical scoring results remain separate. Reported bucket height remains 175 mm with unresolved overall-height/interior-depth interpretation; the provisional model is unchanged.
