# Independent native audit and bounded launch contract

This adds read-only native TRAIN verification and a mock-only one-start
supervisor to the [candidate continuation](CANDIDATE_CONTINUATION_PLAN.md).
No simulator, training, diagnostic release or new shot was started. The frozen
benchmark, checkpoint, bank, scene, rewards, guards and interfaces stay unchanged.

## Independent native evidence audit

[`audit_native_train`](candidate_continuation_native_audit.py) takes externally
trusted journal/source/native-input identities. It reads a complete TRAIN run
without constructing a policy, optimizer or simulator, and checks:

- Exact ledger replay to the first 1,400-release or 1,750-request cap. Every
  reserved request needs one independently verified terminal result; rejected
  requests remain charged. Missing, duplicate, partial or unlisted native files
  fail closed. It does not resume or refund work.
- The durable pre-dispatch claim, compressed and uncompressed packet hashes,
  physics bytes, fixed candidate/action/latent/context mapping, current policy
  hash and inactive padding. Native labels and rewards must exactly match the
  learning journal's terminal rows, including split-prefix outcome writes.
- Actual URDF, poses and camera bytes against the native binding; arm chains
  are reconstructed from those same verified bytes. Real evidence also requires
  the original frozen identities and parent checkpoint bytes. Synthetic fixtures
  are explicitly labelled and cannot establish real-parent verification.
- The [guard auditor](candidate_continuation_native_guards.py) reconstructs
  canonical home association, saved-fit FK and 16-sample path checks, original
  settling windows, all **14 joint-velocity channels**, correction commands and
  their saved fits/path checks, and first-cause rejection labels. Six rotary-axis
  positions/targets are recorded; finger positions and motion between readbacks
  are not independently certified. No new IK search occurs during audit.
- The [score auditor](candidate_continuation_native_score.py) reuses the unchanged
  original numerical scorer and reward on saved 240 Hz traces. It verifies clock,
  trace shape, contacts, captures, clean captures and shaping terms. Parked slots'
  null rewards are adapted only in an in-memory copy for the legacy whole-farm
  scorer. Active rewards and original packets are never changed.

A native TRAIN pass is **one audit component**, not model promotion or a complete
pilot audit. The [existing saved-state audit](candidate_continuation_verify.py)
still independently checks the final checkpoint, structural learning history and
seeded schedule; neither auditor replays intermediate Adam arithmetic. The new
native auditor does not certify the 420-request fresh assessment or the actual
startup initialization/step counts. Original startup contact/rim outcomes are
rechecked separately; bounded diagnostic schedule accounting belongs to the
supervisor contract below.

Reported native batch wall/CPU costs are sums of saved per-batch producer timers.
They exclude shared initialization, separate learning and audit costs. They do
not establish GPU memory, device occupancy, full phase resource costs or physical
validation. Sampled capsule paths remain proxies, not mesh collision certificates.

## Bounded launcher, tested through an injected mock runtime

[`LaunchPlan` and `supervise`](candidate_continuation_launcher.py) implement an
immutable, new-only ownership and resource contract. **No production process,
GPU-query or subprocess backend, executable command, or launch CLI is included.**
Only an explicitly mock-only runtime is accepted. Test success is labelled
`mock_complete_independent_audit_required`; it never means native audit passed.

The fixed limits are:

| Boundary | Limit |
| --- | --- |
| Existing 4090 and shared scene-lock wait | 7,200 s |
| Native application starts | 1, with no retry or restart |
| Shared initialization / diagnostics | 600 s |
| Imitation | 600 s |
| Native TRAIN | 7,200 s |
| Frozen assessment | 1,800 s |
| CPU-only audit | 600 s |
| Absolute native application wall time, all phases | 10,800 s |
| Storage floor | 6 GiB; no evidence deletion |
| TRAIN | First of 1,400 releases or 1,750 requests |
| Final assessment, separately accounted | 420 requests |
| Startup diagnostics | At most 14 cohorts / 490 initialized trajectories / 1,824 physics steps |

Startup diagnostics retain the original ordered rim-ray early stop, five contact
cohorts and one foreign-cell isolation cohort. All 35 initialized ball states
must be hash-bound per cohort, including the 34 parked balls during isolation.
Parking resets and diagnostic costs stay separate from task requests. Startup
records cannot change after initialization, and no smoke release is admitted.

The supervisor rechecks source/storage at each poll, enforces monotonic clocks,
phase order and request caps, and writes start intent before the sole start call.
It can terminate only the exact owned PID, start token, session, container and
4090 identity. The runtime must atomically recheck ownership at signal time.
Foreign/reused PIDs are never signalled. Uncertain ownership preserves evidence
and retains the mock scene-lock lease for inspection. Failures and interrupts
cannot create another start in the same output directory.

The synchronous mock contract cannot preempt a blocked runtime callback. A future
production backend needs a separately reviewed external hard watchdog, actual
dependency/source/ownership verification, and actual resource measurement. No
4070, service/container restart or physical command is implemented here.

## Verification and remaining execution gates

[Compact software verification](../../baselines/isaac-candidate-native-gates-20261109/README.md)
contains test counts, scopes and source hashes, not raw traces. Tests use generated
arrays, synthetic bookkeeping and fake process/clock/storage/GPU responses only.
One composed test runs both real numerical guard and score auditors on the same
35-request synthetic cohort; it remains fabricated evidence, not shot execution.

Before any separately authorized real run, assemble and mock-test the final-only
assessment dispatcher, integrated entry point, combined learning/native/assessment
audit, production watchdog/runtime and phase resource receipts. These additions
must preserve the fixed fresh splits, one-start limits and no-retry accounting.
There is no training authorization or performance improvement claim in this task.
