# Direct seed-22 continuation: 30,000 requests or eight hours

This explicitly authorized continuation of the original continuous PPO model
completed and passed its independent native audit. The time budget stopped
TRAIN at **8,960 additional requests and 67 updates**. On the same 70 new TEST
positions, parent + checker caught **26**, continued + checker **28**; both
had **eight clean captures**, while rejections increased **nine to 20**.
This is a mixed single-continuation result. See the [audited comparison,
complete scalar curves, costs and saved model identities](../../baselines/isaac-seed22-continuation-20261110/README.md).
The completed six-run benchmark and the separate V4 hybrid remain historical.

The starting checkpoint is original seed `20261122`, SHA256
`57d7658e9ed2f72ad09cf4e8523ab497ccbe0708364f1b8ec4a4c5c1aef1fe9f`.
Both its weights and saved Adam state are restored: 10,000 inherited requests,
73 update rounds and 292 optimizer steps. A separately seeded continuation
exploration stream is explicit; this is not a bit-for-bit replay of the old RNG.

## Fixed experiment

- Stop at **30,000 additional reserved TRAIN requests** or **28,800 seconds**,
  whichever is reached first. All geometry and native declines remain charged.
  Assessments have a separate 525-request maximum inside the same native wall
  limit. Startup is included. The final 1,200 seconds are reserved for final
  assessment/checkpoint/cleanup, so TRAIN can stop before the absolute ceiling.
- Use the original 35 TRAIN bucket locations with fresh stochastic proposals.
  More attempts increase shot-setting experience, not unique TRAIN target count.
- Keep original actor/critic, five inputs, five continuous action bounds,
  physical scene, material assumptions, reward, controller and release checks.
  No bank selection, imitation updates, outcome search or old-shot replay.
- Apply the already tested continuous geometry layer with unchanged 24-function,
  512-residual-evaluation and 10%-of-span component adjustment limits. It retains
  admitted settings, attempts one local repair, or declines. Native movement,
  settling and release checks remain necessary after mathematical admission.
- Fresh, outcome-blind positions: 35 monitoring and 70 final TEST, with original
  stratification and 25 mm separation from discovered historical coordinates and
  each other. No reachability or capture filtering selects easier test targets.

Monitoring is prescheduled at 0, 1,000, 2,000, 5,000, 10,000, 20,000 and 30,000
additional requests, plus a final time-stop reading if needed. Monitoring never
selects a better earlier model or changes training. The last completed model is
saved before final TEST outcomes.

Final TEST compares three fixed methods on the same 70 locations: original
parent directly, original parent with geometry wrapper, and continued model
with the same wrapper. The last two isolate the learning effect. The first two
separate the wrapper's effect. Report captures, clean captures, releases,
rejections and causes using all requested positions, plus conditional-on-release
results clearly labelled. One continuation is not a repeated-seed reliability
or real-hardware result.

## Learning and evidence

The unchanged PPO update receives each **original sampled latent and original
log probability**, including every failed request, and the reward observed after
the fixed execution wrapper. The adjusted setting is never substituted into
that likelihood. Proposed settings, executed settings, measured starts and
geometry proofs remain separately recorded. This is continuous PPO with a fixed
environment-side transform, not a categorical policy over a shot library.

Up to four fresh native batches enter an update, normally 140 requests. Four
full-buffer passes use original Adam rate 0.0003, PPO clipping 0.2, value weight
0.5, exploration weight 0.005 and gradient limit 1. Buffers flush at prescribed
milestones or the final stopping boundary; they are then discarded.

Reservations are fsynced before geometry or physics. Partial paid work stays
charged and unresolved; no paid request is repeated or refunded. New output,
checkpoints, native 240 Hz traces, complete ledger/history, original-helper
identities and independent reader results stay separate from original evidence.
The independent CPU reader checks likelihoods, membership, geometry, native
settling, full-rate outcomes, reward and update/checkpoint joins. It does not
claim a bitwise CPU replay of CUDA Adam.

## Runtime bounds

The existing 4090 container is used without restarting it. The production CLI
uses only the existing `/workspace/scene.lock`; the 4070 and other services are
outside this task. The external pidfd watchdog owns one child, independently
enforces the wall/counter/storage limits, and signals only verified descendants.
Native execution also checks 6 GiB disk free, 1 GiB host RAM available and
512 MiB GPU free, plus periodic input/source/asset integrity.

CPU-only actual-input preflight verifies the exact parent and optimizer before
App startup. A zero native exit remains **audit pending**. A failed or partial
run is not silently restarted, extended or promoted. No model promotion,
physical-arm, Viam, perception, cloud or audio work follows automatically.

Source verification distinguishes the original producer from the already
reviewed geometry adapters: `joint_feasibility.py` was added later; the film
entry gained CLI dispatch, and `SettledTransitShots.prepare` delegates its
existing qualification/path calculation to the shared preflight helper. These
three exact adapter hashes are pinned separately. Old full source bytes and
unchanged physical/controller function bodies are independently compared; every
other original physical/scoring/policy helper must retain its original hash.

Resources separate proposal, geometry, native episodes, updates, checkpoints,
assessment and independent audit. Geometry time is already inside episode wall;
sampled GPU occupancy and lifetime RSS are not policy-only or phase-local peaks.
Prescheduled native TRAIN recordings are retained regardless of outcomes.

The completed run charged 8,960 TRAIN plus 420 assessment requests, all terminal.
Its native wall across all attempts was 28,773.736 seconds, inside the 28,800-second
limit. After all results and the final model were saved, default Kit shutdown
waited without exiting; an identity-verified graceful TERM request returned zero
before the existing watchdog deadline. No producer, scoring or audit guard changed.
The subsequent unchanged CPU auditor passed; a private standard-library `os.wait4`
collector measured its resources because `/usr/bin/time` was absent. The native
producer remains commit `77c57cf0921c694a3321a4d97faae3d7af345d27`; later report
commits do not replace its frozen source or evidence.
