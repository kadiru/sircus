# Bounded continuation launch attempts — November 9, 2026

The first authorized seed-22 continuation attempt stopped before the simulator was
started. No training, task shots, startup diagnostic trajectories, held-out
requests or new checkpoints were produced. The original frozen benchmark,
parent weights and candidate bank remain unchanged.

The 4090 was idle. With a specific user exception, the existing stopped container
was started with its existing `sleep infinity` command. Static archive/input
verification and CPU-only production-input/dependency checks passed. Source
commit `f81304c8172ffd0f85752f373c1a2270630d8e0f` was archived and staged separately;
782 source files, 14 input files and all 80 assets were byte-bound.

The watchdog then refused startup because the container's Python 3.11.13 exposes
neither `os.pidfd_open` nor `signal.pidfd_send_signal`. It failed during process
preparation, before the native-child creation call. The launch receipt's one
`native_application_starts` is a **start attempt**, not an observed simulator
initialization. All charged counters and diagnostic counts are zero. The
supervisor exited; a subsequent read-only check found only the existing
container's `sleep infinity`, no GPU compute owner and the scene lock free.
The conservative failure receipt still records an uncertain child identity and
retained-lock intent; that historical receipt is not rewritten.

The attempt is `native_failed_no_retry`; its directory and logs are preserved.
No automatic retry, resume, model promotion or performance claim is allowed.
The independent learning/native/startup/assessment audits did not run. Resource
costs were not measured by native phase, and there are no new videos.

An independently recorded self-only kernel check subsequently opened a pidfd
for the checking process, sent signal zero through that fd and closed it. Both
operations succeeded. Installed Ubuntu x86_64 headers define `pidfd_open` as
syscall 434 and `pidfd_send_signal` as 424. This establishes a Python binding
gap; it does not authorize another application attempt or relax ownership checks.

The future-only compatibility adapter now prefers CPython bindings and, only on
the verified 64-bit Linux x86_64 ABI, provides the same pidfd operations through
typed libc syscalls. PID/start/session/ancestry checks and fd cleanup remain
unchanged. All 60 focused mock cases passed (13 adapter, 23 watchdog, 24 runtime),
and independent review passed. A separate exact-module self-only capability
probe also passed inside the existing container, with no application start,
training or retry. The failed executing source snapshot was not modified.

[Compact evidence](verification.json) contains aggregate facts and artifact
digests only. Raw logs, runtime plans/configuration, positions, native data and
process receipts remain in private runtime storage. The interface adds a
separate stopped-attempt panel alongside the preserved historical 70-request
comparison; its earlier passed audit cannot certify this continuation.

## Separately authorized v2 attempt

The user approved a fresh attempt from source
`50085bf50583f76b1b6b569ea32f5eedfa1b4f9a`, with the same caps and a new output
directory. Its simulator started and recorded **9 startup cohorts / 315
diagnostic trajectories / 1,584 physics steps**. No imitation or PPO update,
TRAIN request, final assessment request or new checkpoint occurred.

Two integration failures stopped this attempt during initialization. The host
supervisor could not read a container-owned `0600` ownership receipt. Separately,
the native producer compared unchanged float32 effort caps with float64 URDF
numbers and rejected their representation difference. The actual limit values,
controller and native dynamics were unchanged. No process remained afterward;
the 4090 had no compute owner and the shared scene lock was free. The original
conservative failure receipts remain intact, and there was no automatic retry.

The original independent startup auditor subsequently passed all saved startup
placements, clocks and contact gates on CPU: **4.280 s wall / 4.262 CPU-seconds**,
without another application or shot. This is a component audit of existing
diagnostics; the full learning/native-TRAIN/assessment audit did not run.

The failure receipt measured initialization at **23.248 s wall / 45.108 aggregate
CPU-seconds**, with 6,728.4 MiB process-lifetime RSS high-water and 6,189.4 MiB
sampled device occupancy. These are startup-failure measurements, not an audited
final policy cost comparison; child RSS and device/Torch scopes are not additive.
[Exact compact v2 evidence](attempt-v2.json) preserves those distinctions.

The exact-value guard fix requires the original float32 dtype and exact bits,
rejecting one-ULP changes, malformed/nonfinite limits and wrong dtype. A CPU
check now precedes AppLauncher; actual GPU cap construction remains unchanged.
Private receipt transport is now read-only without relaxing evidence permissions
or process-ownership checks. Its 57 focused mock cases and independent review
passed. An actual read-only check fetched both receipts and ten progress events
from the closed v2 attempt, while host direct reads still failed and all 12 files
retained root ownership and `0600`. The corrected cap helper also passed a real
CPU Torch check against the exact original URDF, rejecting one-ULP and dtype
changes. These checking processes started no native application or learning.

Repository verification covered 1,136 Isaac cases (121 optional dependency skips),
34 physical mock cases, five simulation cases, 50 Node cases and all nine two-arm
check scripts. Two loopback-binding tests initially failed under the sandbox;
both passed with the required socket permission. All remaining checks passed.
Any further application attempt requires separate authorization; v1 and v2 remain
closed.
