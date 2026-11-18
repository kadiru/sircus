# Location comparison: recorded requests, failed shutdown, no final audit

The third attempt stopped at the original watchdog's shared-phase deadline.
**The original final independent audit did not run or pass.** Saved accounting
reports all scheduled requests terminal, but this does not yet establish model
integrity, captures, clean captures, rejection rates or improved generalization.
No such performance results are published here.

## Observed terminal state

At the November 18, 13:46:42 UTC check, the owned container had exited with code
1 and PID zero; its OOM flag was false. The watchdog reported
`stopped_no_retry`, reason `independent watchdog wall deadline reached`,
`termination=stopped` and `lock_release_safe=true`. A subsequent unrelated job
held the 4090 and shared scene lock and was left untouched. These are dated
operational observations, not a claim that either resource is currently free.

| Last producer accounting observation | Reported value |
|---|---:|
| Original-location copy TRAIN requested / completed | 8,960 / 8,960 |
| Broader-location copy TRAIN requested / completed | 8,960 / 8,960 |
| Assessment requested / completed | 600 / 600 |
| Total requested / completed | 18,520 / 18,520 |
| Reservations | 539 |
| Pending reservation | None |

These counts come from unaudited progress/journal metadata. Assessment includes
315 monitoring requests and the planned 285 final requests: the same 70 broad
and 25 nearby TEST positions for the parent and both final copies. Independent
verification of that schedule, all saved model states and full-rate physical
evidence remains outstanding.

The final trace file exists at 33,055,615 bytes. Its SHA256 is
`5910b73e3e7c671781e356ebb4910630d91bc5303f5b5fbf0403ea9c5aff176a`.
The original watchdog exit file SHA256 is
`c5e54df4c434cc9c5b47a2ac89e46110c4f2b11d342d062787df9e99bfd649ff`.
`child-exit.json` and the final audit are absent. Hashes identify preserved bytes;
they do not substitute for the missing independent audit.

At 13:48:34 UTC, a read-only inventory hashed all 25,059 native-output files
(4,330,228,832 bytes); its receipt SHA256 is
`feb925f5bb05a4417ccc76d4e6efa448cbe2160cfccb52b007dbe3ab14311294`.
All 101 bound input files and 217 source files matched before and after; the
original service was unchanged. Trace metadata ends with
`finish / complete_audit_pending`, reason null, and records 539 reserve/proposal/
terminal events, 128 update events and 13 checkpoint events. The three frozen
checkpoint files each have 136,446 bytes and match the inventory; their exact
hashes are retained in `summary.json`. These are structural/byte checks, not
independent verification of model contents, likelihoods or physical results.

## Shutdown and measured time

The producer saves `comparison-trace.json` before calling `app.close()`, but
writes final progress and `child-exit.json` afterward. Installed Isaac Sim 5.1
closure can wait for Replicator/stage cleanup and invokes framework shutdown;
its default fast-shutdown configuration permits process termination. This is a
source-level explanation for missing post-close receipts, **not proof of which
shutdown subcall blocked in this attempt**. The independent watchdog remained
responsible for deadline enforcement and process cleanup.

The watchdog measured **55,723.455 seconds** for the native child lifecycle,
including startup and failed shutdown. The last child snapshot recorded
25,658.694 seconds for the original-location phase, 26,464.569 seconds for the
broader phase and 1,419.758 seconds for the shared phase. Those group phases
include their associated monitoring and orchestration; they are not pure
training compute. The shared snapshot stopped updating before cleanup finished,
so 1,419.758 seconds is **not** the final shared-phase cost. The watchdog continued
charging elapsed shared time independently and stopped the attempt at its limit.
No audited phase-separated performance or memory report exists yet.

## Preservation and proposed next step

Producer source remains `300ab5c6d161b51577d37d2a70f7499fc3f782e9`; its archive,
manifest, original evidence, failed receipts and earlier attempts are retained.
The historical benchmark and saved parent remain separate. No simulator restart,
retry, resumed learning, new shot, model promotion or media encoding follows
from this failure record. Raw models, frames, observations, logs and runtime
configuration remain outside Git. Compact identities are in
[`summary.json`](summary.json).

The [proposed audit-only recovery plan](../../sircus/isaac/LOCATION_COMPARISON_RECOVERY_PLAN.md)
would examine these saved bytes on CPU, retain every original scientific check
and use a separate, explicit process-completion proof and report status. User
agreement to that changed proof is still required. The proposal cannot turn the
old failed shutdown into a clean completion; it is not implemented or executed.
