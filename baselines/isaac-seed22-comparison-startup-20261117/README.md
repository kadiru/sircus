# Location comparison: closed startup attempt and guarded-recorder repair

The authorized comparison started one simulator application, then closed before
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

Startup created an empty diagnostic directory. The signed progress and watchdog
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

The fixed source still needs its new actual-input CPU preflight. A fresh native
attempt requires separate approval: the [bounded launcher](../../sircus/isaac/LOCATION_COMPARISON_PRODUCTION.md)
allows one child start and no automatic retry. No second simulator attempt has
been started.
