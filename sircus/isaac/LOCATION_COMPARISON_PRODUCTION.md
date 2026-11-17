# Location comparison: process ownership and independent audit

Latest status: one fresh corrected attempt is approved but still unstarted. A
further review caught missing original raw-row audit metadata while waiting for
the GPU. The recording repair passed 149 comparison tests and independent review;
its new source/input CPU preflight and fresh resource checks remain pending.
The private launch hold stays active until those checks pass.

Earlier, the [authorized native start closed before any task requests](../../baselines/isaac-seed22-comparison-startup-20261117/README.md).
The startup guard/recorder composition is repaired in `706c3ef`, with actual-input
CPU preflight, 164 focused tests, 44 producer-environment mocks and full repository
checks passed. The failed attempt is preserved; a separately authorized fresh
start and fresh GPU/resource checks remain required. No comparison performance
result exists yet.

Earlier status: the [actual-input CPU preflight](../../baselines/isaac-seed22-comparison-preflight-20261117/README.md)
passed on November 17 using source `89b0afd`. It restored the original model and
optimizer state and matched all 35 saved TRAIN predictions. No simulator,
training or new shots ran; native execution is still a separate decision.

The initial implementation added the production connection for the
[prepared comparison](LOCATION_COMPARISON.md) and its
[runner](LOCATION_COMPARISON_RUNNER.md). That implementation task exercised only
mocks and invented evidence; the later real-model CPU check is recorded above.
The original benchmark, prepared coordinates, parent checkpoint, original policy,
physics, geometry/release guards and learning budgets remain unchanged.

## One bounded application, then a separate CPU audit

`location_comparison_production.py` has no default execution action. Its explicit
supervisor route verifies the actual manifest and launches one owned child through
the byte-bound original producer wrapper. The shared scene lock must already exist;
the watchdog holds it through native execution and verified cleanup. There is no
container/service restart, remote command, automatic retry or resume route.

The native child checks its watchdog ancestry, process start identity, session
cookie, exact source/input bytes and sole prescribed 4090 visibility. A bounded
ownership handshake avoids a race between spawning the child and recording its
identity. The parent model and its original 35 TRAIN predictions are checked on
CPU before the only AppLauncher call. The original scene and finite startup
probes are constructed and independently checked before any task request.

The application drives the two independently restored copies through the prepared
schedule: 8,960 TRAIN requests and 64 four-pass PPO updates per copy, 315 shared
monitor requests, and 285 final requests across the parent and two final models.
The same 70 broad and 25 nearby TEST positions are used for each model. Every
reject is charged. Exact completion remains 18,520 logical requests.

The separate watchdog observes its own clock and storage rather than trusting
child heartbeat timestamps. Startup and shutdown use the cumulative shared hour;
each training group has its cumulative eight hours. A blocked native call cannot
pause a deadline. A paid reservation and phase change are published immediately;
unchanged per-tick heartbeats alone are throttled. The independent storage floor
remains 6 GiB. The child additionally checks available host/GPU memory.

Cleanup uses the existing reviewed Linux pidfd operations and rechecks process
identity before signalling. It never sends a signal solely to a numeric PID or a
whole process group. An uncertain cleanup retains the lock while its owner is
alive and records that limitation. The new receipt distinguishes a child start
attempt from a returned child process; its legacy application-start field is
explicitly an alias for child-start intent, not proof that the simulator opened.

A successful native process exit is **audit pending**. Only after verified native
exit and cleanup does the supervisor launch the separate CUDA-hidden audit child,
with its own two-hour limit and the same storage floor. The audit child must prove
live ownership and the exact preceding native exit. Its report is bound to the
manifest, prepared comparison, session, trace bytes and child-exit evidence.
An exit code, a `passed` string or synthetic fixture cannot substitute for this.

## What the independent reader checks

`location_comparison_verify.py` reads saved evidence without simulator steps or
optimizer updates. It combines the accounting auditor with the unchanged original
geometry, motion, settling, release and full-rate scoring readers. It checks:

- Exact parent/source/scene/asset bytes, inherited passed audit, and the reviewed
  original helper/adaptor relationships.
- Original parent weights and Adam state; all initial, monitoring and final
  checkpoints; and the model state used for every proposal. New sampling-state
  snapshots close the previous gap between monitoring checkpoints.
- The original five input features, Gaussian/tanh action likelihoods, direct
  deterministic assessment means, saved optimizer/RNG identities and counters.
- Four passes over each unchanged 140-request PPO buffer, retained rejections,
  before/after weight and actor identities, and update artifact membership.
- Request-to-native-slot joins, padding, geometry adjustment evidence, measured
  motion/settling/release, rewards, captures and clean captures from the original
  full-rate physics observations.
- Scheduled native frame hashes and contiguous frame coverage, without extra
  recordings or shots; missing/partial/unlisted evidence fails the audit.

Sampling snapshots are additional evidence files only. They do not change
proposals, randomness, training examples, updates or final model selection.
Their write time is included in the measured proposal-call wall time; the
separate checkpoint-call total covers the explicit initial/milestone/final saves.
Full-rate NPZ evidence lives under the newly owned `native` directory, alongside
the existing frame root, so the original independent readers retain their safe
path rules.

The audit does **not** replay CUDA Adam arithmetic bit for bit, prove an entire
continuous motion collision-free, or validate physical robot performance. It
verifies saved model states and the recorded simulated behavior under the pinned
implementation. A mock pass does not establish native readiness or improved
catch rate.

## Execution stays off

The later CPU preflight bound the final source and actual inputs in a new private
manifest and verified the original producer's CPU environment. It did not approve
a native launch. Before a real run, recheck those bindings and current
GPU/lock/storage ownership. The session cookie must exist in the
supervisor's process environment before interpreter startup; setting it later
cannot stand in for `/proc` ownership. The child wrapper receives the verified
source directory explicitly. The old continuation manifest is not reusable.

A real launch remains a separate user decision. The CLI requires explicit mode,
manifest SHA, session and execution approval; imports and mock tests cannot start
it. The code does not launch, retry or alter the completed historical benchmark.

See the [mock validation receipt](../../baselines/isaac-seed22-comparison-production-20261117/README.md).
