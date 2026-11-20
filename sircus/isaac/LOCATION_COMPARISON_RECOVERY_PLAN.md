# Proposed CPU-only audit of the closed third comparison

**Approved on November 18, 2026.** The user authorized implementation, mock tests
and a CPU-only audit of saved data, with publication if it passes. The following
is the preserved proposal; [implementation and execution status](LOCATION_COMPARISON_RECOVERY.md)
are recorded separately. No simulator, training, model update or new shot is
authorized by this recovery.
The [third attempt failed during shutdown](../../baselines/isaac-seed22-comparison-shutdown-20261118/README.md)
after its unaudited accounting reported all 18,520 requests terminal. This plan
would assess the saved data without repeating the experiment. It requires an
explicit change to the process-completion proof, not a change to physics,
learning, scoring, model or evidence-integrity checks.

## Why the original final audit cannot currently run

In producer `300ab5c`, [`verify_exit`](location_comparison_verify.py#L233)
requires `child-exit.json`, `app_closed=True`, no child error and a matching
trace SHA. It also requires the successful original watchdog status, exit zero,
no reason, safe lock release and one child-start intent. The child receipt is absent
and the actual watchdog status is `stopped_no_retry`.

[`audit`](location_comparison_verify.py#L262) calls that exit check before its
model and physics checks. It explicitly rejects native check substitution:
“Injected checks cannot certify native evidence.” The
[`audit supervisor`](location_comparison_watchdog.py#L400) likewise requires:
“Audit requires the exact successful native exit receipt.” Consequently,
calling the unchanged top-level audit cannot certify this closed attempt.
An exit code, complete counters or a newly invented child receipt cannot replace
those missing conditions.

The existing source, receipt schemas, failed exit and canonical audit path must
remain unchanged. A CPU read-only audit is not another native start and does not
consume or repeat shot requests. Agreement is needed here because the proposed
report would use a different process-completion proof, not because reading
saved checkpoints requires a new simulator-start exception.

## Separate recovery proof

Before any implementation or execution, record user agreement to this scope.
Then require a sealed, byte-identified evidence set and an independently checked
recovery receipt containing:

- Original source/archive, manifest/comparison/session, trace and failed
  watchdog identities, plus the observed absence of the original child/audit
  receipts. Preserve every original file and its permissions.
- Identity-bound evidence that the owned native processes exited and the
  original watchdog released its lock safely. A new unrelated lock holder is
  not a recovery target and must remain untouched.
- The final trace's complete request/update schedule, no unresolved reservation,
  and final checkpoint identities. Check the persisted journal against the
  trace, rather than trusting a progress counter alone.
- A complete input/evidence inventory hashed before and after the audit, with
  original failure/closure costs retained separately. Do not assert a clean
  application close or compliance with the shared shutdown limit.
- New audit source identity, CPU-only process ownership, exact command and
  measured resource/exit evidence in a fresh output directory. No retry,
  overwritten evidence, forged `child-exit.json`, patched original manifest or
  substitute successful watchdog envelope is allowed.

## Preserve every scientific check

Use the original producer source and input bytes at their declared paths on
read-only mounts. Recovery code must live in a separate directory because
[`validate_native_inputs`](location_comparison_native.py#L98) checks exact source
membership and hashes; imported original helpers must resolve to that verified
tree. Do not monkeypatch the original module or inject permissive native checks.

The separate implementation must retain the complete scientific body of the
original [`audit`](location_comparison_verify.py#L262), replacing only its exit
proof/report binding with the explicitly reviewed recovery proof. Source/AST
comparison and independent review must demonstrate that the scientific checks
and coverage gates are unchanged. Required checks include:

- Original inputs, parent audit, scene, camera, assets, guards and startup proof.
- Exact accounting and cohort order: 8,960 TRAIN requests per copy; 315 monitor
  and 285 final TEST requests; 18,520 requests and 539 terminal batches total.
- All 128 PPO updates and 144 checkpoint artifacts, inherited weights/Adam,
  saved model/RNG states, original proposal likelihoods, unchanged four-pass
  update accounting and frozen final checkpoints.
- Every requested target/slot, charged rejection, geometry adjustment,
  movement/settling/release gate, original 240 Hz scoring, and matching saved
  full-rate physics. No held-out inference search or model selection.
- Exact artifact membership and hashes, including all ten prescheduled frame
  cohorts; missing, extra, altered or partial evidence fails the recovery audit.

These are required verification targets, not claims that the closed attempt
has already passed them. CPU model reconstruction does not create an optimizer
update or replay any shot. As in the original audit, it does not reproduce CUDA
Adam arithmetic bit for bit or certify physical hardware/continuous collision
avoidance.

## Bounds, output status and publication

The new reader would be CPU-only, with CUDA hidden, no GPU devices, no network,
read-only original inputs/evidence and only a fresh audit destination writable.
An independent process must enforce **at most two hours** and the original
**6 GiB free-storage floor**, plus declared CPU/memory limits and owned-process
cleanup. Preserve a failed audit attempt; do not resume or silently retry it.
No existing container, reader, simulator or unrelated job is restarted.

If every scientific check passes, use a distinct proposed status such as
`data_verified_shutdown_incomplete`, never the original clean-completion status.
Bind that report to the original failed exit SHA and a separate recovery-audit
exit receipt. The original audit remains **not passed**. If evidence or bounds
fail, retain the failure and make no audited performance claim.

Any later interface support must explicitly recognize this recovery status and
display the shutdown failure and unaudited-versus-verified distinction. The
current publication gate must continue to reject it until that separately
reviewed integration exists. Do not promote a model. The ten existing scheduled
clips could be encoded only after a successful recovery data audit; eight are
TRAIN clips and two are parent-only TEST subsets, not final-model TEST videos.

## Required mock tests before a proposed CPU audit

1. Prove original canonical audit/publication gates still reject the actual
   missing-child/failed-watchdog combination and any forged clean-exit substitute.
2. Verify recovery identity joins, failed-receipt/trace/journal hashes, safe
   retirement and complete budget/phase checks; reject mismatches and unresolved
   or extra work.
3. Preserve all existing model, likelihood, geometry, physics, score, checkpoint,
   frame-inventory and source-membership failures. Check the original scientific
   verification body has not been weakened.
4. Prove imports and mocks start no simulator, dispatch no native requests,
   perform no optimizer steps and write nothing into protected evidence.
5. Exercise independent two-hour/storage/memory enforcement, ownership-safe
   cleanup, fresh-output-only behavior, audit failure preservation and distinct
   report/exit schemas; include report-byte verification after child exit.

Only after agreement, implementation, these checks and independent review would
an actual CPU-only audit be considered. Complete recorded data may still support
a scientific comparison, but that remains unproved until this audit passes;
the failed shutdown and its cost cannot be erased by the recovery.
