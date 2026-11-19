# Bounded continuation launch attempts — November 9, 2026

## Latest v4 outcome: saved trained checkpoint, incomplete assessment

The separately authorized v4 used immutable source `9a49a0d`. It completed the
first stopping cap: **1,400 releases from 1,470 charged TRAIN requests**, including
70 rejects. Recorded exploratory outcomes were **804 captures / 225 clean**;
their denominator is all 1,470 requests. Imitation reused the existing examples
for 240 steps / 30,680 presentations, then PPO committed 11 updates / 44 optimizer
steps over all 1,470 new requests. These are extra-budget updates to one seed,
not an equal-cost multi-seed comparison.

The final checkpoint was saved and frozen before assessment, with SHA
`15020a948d295fe5e7258a732145b89aa61fda2c385a0d6966a9e4cc7e81bd41`
(186,571 bytes). The original checkpoint/learning reader passed its saved bytes,
journal/ledger and learning-counter checks; a private Mac backup has the same
SHA. The original startup reader also passed nine cohorts / 315 trajectories /
1,584 physics steps. Neither component pass is a completed pilot audit.

Assessment stopped in the first validation-bank cohort. Its empty-support ball
remained parked in all 960 preserved physics samples, but the native wrapper's
`no_selected_setting` row label differed from the adapter's expected
`inactive_split`. The mock had fabricated the latter label. The new regression
executes the unchanged original batch body with fake physics; it reproduced the
old failure and checks the strict correction without new shots or relaxed guards.

There are **140 charged assessment requests: 105 terminal and 35 unresolved**.
Only 96 releases / nine declines were committed; the fourth cohort performed
uncommitted physical work. No final capture counts are inferred from that cohort
and **no TEST request was reached**. No batch is refunded, replaced or resumed.
The model is not promoted, and no held-out improvement claim is possible.

The original native TRAIN reader also rejected a first-cause label: 35 unpaid
padded slots in the last 34-request / one-request cohorts have the same native
outer-label distinction. No active-row first-cause label mismatch was found in
the saved metadata. That original failed reader remains failed and preserved.
A separate strict revised CPU reader from `c0bd3d6` passed all 43 native TRAIN
batches, independently confirming **804 captures / 225 clean and 70 rejections**.
It verifies raw packet/reservation/physics joins and exact inactive padding
invariants before mapping only the 35 unpaid labels in a private guard copy.
Paid rows, saved packets, original guard and scorer remain unchanged. Its native
receipt SHA is `2fa58d3a95502143bfe9304627eeab42ba1b2944852ae9551e5fce68617eb204`.
The repeated learning/checkpoint and startup checks passed as well. No simulator,
optimizer, new request or held-out outcome reading occurred in this CPU check;
its 281.80-second reader loop / 469.21 process-CPU seconds exclude imports/input
validation, and its 1,394.1 MiB RSS is process-lifetime high-water. These checks
certify completed TRAIN components only, not the missing assessment/combined
audit. [Compact v4 evidence](attempt-v4.json) records each status and the untouched
source/checkpoint/input identities.

| Producer phase | Wall seconds | Aggregate CPU seconds | Lifetime RSS high-water, MiB |
| --- | ---: | ---: | ---: |
| Initialization | 24.56 | 46.92 | 6,888.3 |
| Imitation | 33.10 | 47.62 | 6,888.3 |
| TRAIN | 3,862.30 | 8,282.63 | 7,226.8 |
| Partial assessment | 296.26 | 635.50 | 7,346.3 |

These producer costs describe a stopped attempt, not completed per-method
assessment. RSS is process-lifetime high-water; sampled device occupancy was
6,189.4 MiB and includes PhysX/renderer. Torch GPU allocation peaked at 0.279 MiB;
the neural updater uses CPU. Host supervisor/watchdog and separate CPU reader
costs are excluded; earlier attempt/preparation costs are not erased or added
as if they were non-overlapping phases. No new videos were recorded by protocol.

The finite publisher exited, owned native job exited and scene lock released.
The existing GET-only viewer shows the TRAIN curve, precise partial accounting
and stopped producer costs without restarting its server. All prior attempts,
raw evidence, original seed-22 model/bank and frozen benchmark remain unchanged.
No additional training, simulation start, assessment or retry is authorized by
this report. A future evaluation can reuse the saved model; it requires its own
reviewed bounds and treatment of the incomplete assessment.

## First authorized attempt: before simulator startup

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

## Separately authorized v3 attempt

V3 used source `03d31ff85a3201bce6a2bd7339e80c13a354a308`. Startup ownership and
effort checks passed, and **62 actor-imitation updates / 7,926 presentations of
existing examples** were committed. No PPO update or TRAIN/assessment request
occurred. The watchdog stopped its owned child after 64.923 seconds on a timestamp
check. Additional in-flight optimizer work is unknown; no final checkpoint is
committed or present in the observed checkpoint inventory. This partial history
cannot be resumed or promoted as a usable trained model.

A deterministic regression reproduced the race: the writer can publish a valid
receipt while the watchdog hashes source/reads progress, after the watchdog's
earlier clock sample. The corrected guard samples after reading the frozen event
list, and again after validation. It retains event-based phase starts, finite
nondecreasing clocks, original phase/absolute caps and exact future-timestamp
rejection, with no epsilon. All 68 focused cases and independent review passed.

This run's existing startup traces also passed the original CPU-only reader:
nine cohorts, 315 initialized trajectories and 1,584 physics steps, taking
4.327 seconds wall / 4.327 CPU-seconds. Full learning/native-TRAIN/assessment
audits did not run. The original checkpoint and all source/input bytes verified
unchanged after termination. [Compact v3 evidence](attempt-v3.json) keeps those
component checks separate from policy performance.

The viewer shows recorded counters separately from committed updates and unknown
in-flight work. Its optional TRAIN curve uses completed requested slots,
including rejections, with unresolved charged slots separate; it remains
unaudited exploration. The v3 curve is empty because no task was requested.
Current full repository checks passed (1,147 Isaac cases with 121 optional skips,
56 Node cases, 34 physical mocks, five simulation cases and nine two-arm scripts).
V1, v2 and v3 remain closed and preserved; no automatic retry was performed.
