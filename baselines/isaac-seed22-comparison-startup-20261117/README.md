# Location comparison: closed startup attempt and guarded-recorder repair

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
the fresh attempt started; its one-start approval remains unused.

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
