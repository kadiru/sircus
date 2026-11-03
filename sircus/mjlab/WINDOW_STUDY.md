# Conditional release-gap windows and terminal SAC

This opt-in simulation experiment fixes one previously successful development
bucket and board XY. It maps release-gap windows while conditioning on board
height and tilt. This is deliberately smaller than the ultimate cooperative
two-arm task: the action is **gap only**, not joints or board orientation. The
board's exact joint branch is frozen before outcomes for each context.

The corrected articulated-board/passive-gravity contact contract uses 64 native
substeps per 240 Hz observation (15,360 Hz), unchanged bucket geometry and scorer.
A capture requires a board rebound, whole-ball entry and one continuous second
inside. Rim assistance counts; external assistance does not. The reported
175 mm bucket height remains unresolved; the provisional model still uses
175 mm interior depth and an 8 mm bottom. No physical calibration is claimed.

## Frozen experiment

- 24 training, six validation and eight test height/tilt contexts; no overlap.
  Bucket/board XY are identical throughout, so this is not unseen-bucket testing.
- Gap bounds 20–650 mm; board-center height 40–560 mm and tilt 8–64 degrees
  from horizontal. All actions retain strict 5 mm endpoint clearance, self/cross
  arm, tool, carried-ball, board, bucket, floor and table checks. Some low poses
  will be rejected. Failed requests consume budget and remain in denominators.
- Shared seed: 21 gaps ×24 contexts =504 requests, including the exact robust
  development gap. Each learner is charged these504 requests.
- Adaptive search adds504 requests: three rounds of seven/context, combining
  transition refinement with broad remaining intervals. A supervised RBF logistic
  model predicts binary capture. Its probabilities are not calibrated guarantees.
- SAC adds504 different requests:21 rounds of24. A Gaussian continuous actor and
  twin critics learn the existing shaped terminal reward. Every shot terminates,
  so there is no next-state bootstrap. Alpha=.02,400 seed-data updates then32
  updates/round, batch64, CPU network training. Native physics uses only4090.
- Different learning objectives are explicit: surrogate predicts captures; SAC
  optimizes shaped reward plus entropy. Both are evaluated by capture counts.
- Four matched data milestones504/672/840/1008; six validation contexts ×three
  paired repetitions each. Select captures, then mean reward, then earliest.
- Freeze selected checkpoints/actions before test: eight contexts ×three nominal
  +20 fresh Gaussian gap perturbations (sigma2 mm), per method. A frozen bounded
  uniform baseline has eight independent base actions, repeated by the same
  protocol. Out-of-bounds perturbations fail; no clipping, retry, projection or
  native search during inference.

Total:1,512 unique training requests (1,008 charged to each learner),144 validation
and552 final test requests, plus three quarantined native anchor controls =2,211.
Validation/test/control outcomes never enter replay or supervised fitting.
One seed and correlated repeats cannot establish universal method superiority.

Sampled successful gap runs may be disconnected. Their observed endpoints and
neighboring failures do not certify every value between them, conditional
min/max ranges do not form a safe Cartesian box, and2 mm gap perturbations do
not validate board-placement noise. Raw 240 Hz position/velocity/category-force
traces support independent rescoring and approximate under/overshoot diagnostics.

All native shots hold prescribed endpoints; they do not learn grasping, motion,
or actuator tracking. Selected successful actions receive a separate sampled
common-home geometric path audit where supported by the existing transit bounds.
Other paths remain unknown. No active policy/server/hardware changes are made.

## Reproduce

Use the separately pinned Ubuntu Python3.10 environment and read-only assets
listed in `requirements.txt`/the source manifest. Select4090 by its UUID, not an
assumed index. The runner waits on the cooperative Isaac/4090 lock and never
interrupts another process. It refuses an existing output directory, records
source/assets/packages before labels, caps output at512 MiB, requires512 MiB free
space and stops after a finite four-hour execution budget.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4
"$HOME/sircus-mjlab/venv/bin/python" -m sircus.mjlab.window_study \
  --assets "$HOME/sircus-isaac/assets" \
  --scene baselines/mjlab-contact-transit-20261102/scene.json \
  --anchor baselines/mjlab-contact-transit-20261102/nominal-source.json \
  --output "$HOME/sircus-mjlab/output/window-sac-NEW" \
  --gpu-lock "$HOME/sircus-isaac/scene.lock"
```

After completion, run `window_verify` on CPU against the executing immutable
source snapshot. Use `prepare_window_report` to append read-only results to the
existing combined viewer report; it does not switch inference or restart a server.
The live viewer's **Conditional gap study** panel preserves older results.
Raw native arrays/logs remain outside Git; curated reports, CPU preflight and
selected small checkpoints may be published with matching fingerprints.
