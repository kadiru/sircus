# Location comparison: third start authorized and running

The user approved the prepared third attempt once the 4090 and shared scene lock
were free. It started on November 17 at 22:15:51 UTC from source `300ab5c`, after
source/input/storage/ownership checks. Its startup audit passed. At 22:19:05 the
initial 35 monitoring requests were reserved but not yet terminal, with zero TRAIN
requests and no failure receipt. This is a dated launch observation, not a final
performance result.
Both failed attempts and the original benchmark remain preserved. The same
limits and one-start/no-retry rule remain active.

Second attempt: the approved second start also closed before any TRAIN or assessment
request. Its watchdog reported `Native child ownership changed` after 0.233 s;
only the owner and exit receipts exist. The container exited, its owned processes
stopped and the shared lock was released safely. Both failed attempts remain
preserved. The one-start approval is consumed; no automatic retry is permitted.

A CPU-only test of the unchanged wrapper ran 30 inert Python children, all exiting
zero. In 24 runs, Linux briefly returned an empty environment cookie for a live
process during an exec transition; an immediate read recovered the exact same
PID, start time, parent, session and cookie. Ten runs also demonstrated an exit
between polling and the identity read. This reproduces watchdog race conditions,
but the failed native receipt did not capture its mismatching snapshot, so its
specific transient cannot be proved from that receipt alone. No simulator,
model update or task request was part of these diagnostics. All 101 input and
217 source files matched afterward. The bounded identity-read repair is now CPU
verified; all process and experiment guards remain required. A fresh prepared start remains held for explicit approval.

The following sections preserve the earlier attempt and preparation history.

The authorized comparison used its single simulator-start attempt, then closed before
any TRAIN or assessment request. **No new training, evaluation or capture-rate
result exists.** The original benchmark, model, prepared positions and budgets
remain unchanged. The failed source was `89b0afd`; exact input and receipt
identities are in [`summary.json`](summary.json).

## What the native evidence shows

The new container exposed only the prescribed 4090. Its actual-input, saved-model,
GPU and self-only pidfd checks passed before entering the bounded supervisor.
The original simulator container stayed stopped; historical experiments and
source were mounted read-only. New outputs used a separate persistent parent,
with the exact container-to-host mapping retained in private launch receipts.

Startup created an empty diagnostic directory. The hash-bound progress and watchdog
receipts show **zero TRAIN requests, zero assessment requests and no pending paid
reservation**. No learning loop or startup diagnostic record was reached. The
application's close did not return normally. An identity-checked pidfd sent one
graceful TERM to only its owned Python process; the application and watchdog
then exited. The watchdog recorded `stopped_no_retry` with reason
`Native exit lacks final progress`, and confirmed safe lock release.

The native child returned zero, but there was no final child receipt or completed
startup audit. It therefore did **not** pass. The watchdog measured 599.618 s;
the enclosing container measured 603.374 s. These are failed-start costs, not
training costs. All bound 101 input files and 217 source files matched afterward.
The failed container, files and logs are preserved. A later unrelated job took
the 4090 and was left untouched.

The initial host launch check also encountered Docker's `CAP_DAC_OVERRIDE` name
normalization while the container was still in `created` state. Correcting that
host-only comparison allowed the same never-started container to start once;
it did not consume or repeat a simulator attempt.

## Diagnosis and repair

Source inspection and a CPU mock reproduction found a deterministic composition
error: the production startup service installed its own `env.tick` wrapper before
the real startup recorder checked that movement methods were genuinely inherited.
The recorder rejected the override before taking any diagnostic step. The old
composition reproduced the exact guard error, zero ticks, zero guard calls and
an empty startup directory; it also left an instance method override afterward.
This is **source/mock diagnosis**, not a preserved native traceback: the old
exception handler held the first cause only in memory while closing the app.

The fix passes a guard callback into the recorder. It keeps the strict inherited
method check and calls the guard before every original tick, inside the recorder's
existing instrumentation. The service no longer overrides the method. Cleanup
continues to remove the recorder's temporary overrides. Probe placement, physics,
scoring, motion/release guards, targets and learning budgets do not change.

The child also writes a durable `child-failure.json` before potentially blocking
cleanup. A failure to write that receipt still proceeds through cleanup and
remains an error; it cannot become an audit pass.

**147 comparison tests and 17 recorder tests passed.** The genuine recorder and
independent startup reader were exercised together with a fake environment:
1,488 synthetic ticks, 245 initialized trajectories and one guard before each
tick. Regression tests retain rejection of pre-existing method overrides,
guard denial before motion, exception cleanup and failure-receipt ordering.
These tests start no simulator and do not measure native performance.

## Corrected source verified on CPU

Fixed source `706c3ef` passed its actual-input CPU preflight in the unchanged
producer image: Python 3.11.13, Torch 2.7.0+cu128, NumPy 1.26.0 and SciPy 1.15.3.
The original seed-22 model and Adam state restored; all 35 saved TRAIN predictions
matched, with maximum normalized error 1.78814e-7 against tolerance 5e-6.
The CPU check took 2.603 s wall / 2.479 CPU-seconds and 669.3 MiB peak RSS.
These are preflight-process costs, not simulation or training costs.

The same isolated producer container then passed 44 recorder/production mock
tests. It had no GPU or network access, read-only experiment mounts and zero
simulator imports or actual learning steps in the model preflight. Afterward,
all 101 input files and 217 source files still matched; prospective native/audit
outputs were absent, and 15.030 GiB storage remained free. The original stopped
simulator container was unchanged. Exact manifest/archive/receipt hashes are in
[`summary.json`](summary.json); the prepared comparison fingerprint is unchanged.

Full repository checks also passed: 1,665 Isaac cases (1,544 passed, 121 optional
dependency skips), 59 Node cases and all remaining required stages. The focused
164 tests above and the 44 producer-environment tests passed without skips. These
checks verify software and input compatibility, not successful native startup.

Corrected V2 launch files were staged separately but unstarted. At that checkpoint
a fresh native attempt required separate approval: the [bounded launcher](../../sircus/isaac/LOCATION_COMPARISON_PRODUCTION.md)
allows one child start and no automatic retry. It retains 8,960 TRAIN requests
and an eight-hour cap per copy, the shared one-hour phase cap, and the same fresh
70 broad + 25 nearby TEST positions for parent and both final models. Before
launch it must recheck source/input hashes, 4090 ownership, the scene lock and
the 6 GiB storage floor. No second simulator attempt has been started; the
unrelated 4090 workload stays untouched.

## Additional recording repair before the approved fresh attempt

The user subsequently approved one corrected attempt when the 4090 is free.
While waiting, an additional integration review found missing audit metadata:
the comparison stored raw `shot_batch` rows without the five fields attached by
the original producer. The unchanged auditor needs `split`, `qualification`,
`path_rejection`, `path_proxy` and `scoring_board_normal`. This was found before
the fresh attempt started; its one-start approval was still unused at that checkpoint.

The bridge now attaches those exact original fields after `shot_batch` and
before feedback and durable recording. Existing scores, poses and actions stay
identical; no policy, geometry, movement, physics, reward or budget changes.
The 149 comparison tests passed. New regressions execute the original raw-row
construction methods over fabricated arrays, then the unchanged independent
scoring, rejection and padding readers. They cover released misses, geometric
rejections, unpaid padding and a full all-rejected TRAIN batch. Padding retains
`None` feedback features and `-1` rewards, as required by this comparison's reader.
Independent review found no blocker; its 30 focused cases passed.

The later source `90ae3ff` passed a new actual-input CPU preflight: all 35 saved
parent predictions still match (maximum error 1.78814e-7), all 101 input and 217
source hashes match, and prospective native/audit outputs remain absent. Only
`location_comparison_native.py` changed among the 217 production Python files.
All 60 producer-environment recorder/production/native mock tests passed. The
model preflight measured 1.979 s wall / 1.998 CPU-seconds / 669.7 MiB peak RSS;
these are CPU-check costs only. Postcheck free storage was 13.317 GiB.

Full repository checks passed again: 1,667 Isaac cases (1,546 passed, 121 optional
dependency skips), 59 Node cases and the remaining required stages. Exact new
archive, manifest and receipt hashes are retained in `summary.json`. The prior
`706c3ef` preflight remains historical; the approved attempt now binds `90ae3ff`.
The preparation hold can be released, with the same unused one-start approval;
the launcher still requires a fresh idle-4090, source/input, scene-lock and
storage check. No second native application, model update or task shot has
occurred at this checkpoint.

## Ownership repair verified; third start held

Final source `300ab5c` retries only incomplete or unreadable child-identity samples
for at most 100 ms / 21 reads. It never accepts a blank cookie or unreadable
identity as ownership. PID, start token, cookie, parent and session must all
match; changed identities and persistent unreadability still fail closed.
Deadline, storage and source checks remain active during rereads. A confirmed
child exit now reads the final atomic progress again before assessing completion.
Original identity and pidfd signaling helpers are unchanged.

The actual producer-wrapper check passed **60 inert children**, 30 under the
file-owner UID and 30 with the native container's root/capability setup, with
GPU visibility disabled in both. The helper handled 37 empty-cookie reads,
14 unreadable reads and 19 absent reads; all children exited zero. These are
process-observation tests, not simulator starts or task shots. Earlier CPU checks
remain preserved: one private probe mistakenly checked the small temporary
filesystem; the next exposed the PermissionError case now covered by the fix.

The final actual-input CPU preflight again matched all 35 saved parent predictions
(maximum normalized error 1.78814e-7). It measured 1.995 s wall, 2.017 CPU-seconds
and 670.2 MiB peak RSS. All 101 input and 217 source hashes matched afterward; only
`location_comparison_watchdog.py` changed among production Python files relative
to `90ae3ff`. The producer environment passed 116 mock tests. Local verification
passed 174 comparison cases, including 56 watchdog cases, plus 39 unchanged
original-watchdog cases and independent review.

The broad repository run exercised 1,686 Isaac cases with 121 optional skips;
two loopback tests were blocked only by sandbox binding. Their two files passed
all four cases with loopback permission. The 34 physical mocks, five simulation
mocks, 59 Node cases and nine remaining numerical scripts passed. The final
unreadable-snapshot follow-up added six cases and passed the complete comparison
suite. No native performance is established by these software checks.

A fresh third package binds source `300ab5c` and manifest SHA
`7de349d890ca032d2f8802b987350de55ca5314a52fbcfab6fdfc7d6c8109a5b`.
It is prepared behind an explicit HOLD with zero starts and requests. The same
8,960 TRAIN requests / eight-hour per-copy limits, 70 broad + 25 nearby TEST
positions per model, shared-hour cap and 6 GiB floor remain. The two failed starts
are closed; the [one-start/no-retry rule](../../sircus/isaac/LOCATION_COMPARISON_PRODUCTION.md)
required a fresh exception before this package could start. The later approval
and dispatch are recorded at the top of this report.
