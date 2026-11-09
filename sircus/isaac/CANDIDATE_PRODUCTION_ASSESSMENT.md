# Production runtime and final assessment integration

This completes the source wiring following the
[native audit / bounded launch contract](CANDIDATE_NATIVE_GATES.md). The initial
implementation was tested with mocks. V1–v3 stopped before task requests; v4
completed TRAIN and saved its checkpoint, then stopped during assessment. The
[preserved evidence](../../baselines/isaac-candidate-live-pilot-20261109/README.md)
distinguishes those attempts from a completed audited pilot. No improved policy
performance is established. The completed benchmark and its checkpoints/bank
remain unchanged.

The original native batch labels a disabled row `no_selected_setting`, while its
preparation qualification stays `inactive_split`. Empty-support assessment
requests retain their charged semantic decline `no_qualified_candidate_support`.
The adapter now requires the exact native label and all no-execution invariants;
the auditor normalizes only its private guard copy after strict raw checks.
Saved packets, original batch code, release/IK/scoring thresholds and failed
attempts are never rewritten. Regressions exercise the original batch body with
fake physics, including all 420 synthetic slots and six empty-support declines.

Unpaid TRAIN padding has the same raw-label distinction. The TRAIN reader
requires exact padding identities, null request/choice/IK/preflight/path/reward
fields, inactive qualification, false success flags and an empty trajectory
before adapting only its copied guard view. The original generic guard and
scorer still run; active rows and saved packets are unchanged. A corrected
post-hoc reader has its own recorded source identity, preserves the original
reader failure, and cannot complete or promote an interrupted assessment. The
separate `c0bd3d6` CPU reader passed all 43 V4 TRAIN batches: 1,470 requests,
1,400 releases, 70 rejections, 804 captures and 225 clean captures. This completed
TRAIN component check is not an assessment or combined audit.

## Process and ownership boundary

The [host runtime](candidate_continuation_runtime.py) verifies the exact Git
archive/commit, source tree, external manifest SHA, every declared raw input,
protected roots and original shared scene lock. It uses only the already-running
4090 container and queries only its declared GPU UUID. There is no container or
service restart, install, arbitrary command, retry or 4070 path.

`ProductionRuntime` is inert on construction. Future execution requires both an
explicitly authorized runtime and `supervise(..., execute=True)`. Configuration,
checkpoints, passed audits and mock tests do not grant execution permission.

The fixed process chain is host supervisor → existing-container
[independent watchdog](candidate_continuation_watchdog.py) → one
[application entry](candidate_continuation_entry.py) → separate CPU audit child.
Host and watchdog imports require only Python's standard library; the learning
contract repeats the policy's fixed probability-law label, checked for equality
by tests, without importing NumPy or Torch. The native child still verifies its
full bound dependency versions before app startup. This bootstrap separation
does not alter the learning protocol hash or provide a CPU simulation fallback.
The watchdog has its own process and clock: a blocked native callback cannot
prevent enforcement. It checks phase order, charged counters, source/storage,
startup schedule and absolute/phase deadlines using append-only session-bound
progress receipts. Host polling preserves phase boundaries even when many fast
imitation receipts arrive between polls.

The watchdog bounds each already-read progress list with a fresh monotonic clock
sample after the read, then rechecks clock/deadlines after validation. This avoids
calling a concurrently published receipt "future" against a stale pre-read time.
Actual receipt timestamps still set phase starts; no tolerance or deadline was
added. Truly future/reversed stamps, delayed transitions and time spent reading
or validating beyond a phase/absolute cap still stop the owned application.

Private container receipts remain `0600`. The host reads only the two fixed
runtime receipts and complete progress metadata through the existing container's
read-only snapshot action. Each response binds a fresh request nonce, exact
watchdog-input bytes/fingerprint, session and output; the host rechecks every
receipt/progress fingerprint, owner join and already-observed history prefix.
Canonical paths, symlinks and the fixed file allowlist are checked on the read
side. This path imports no application/model code and cannot start or stop work,
change permissions or read arbitrary evidence. Direct host reads are not used.

The watchdog uses Linux pidfds and PID/start-token/session/lineage checks to stop
only its owned child and descendants. It does not signal a process group or a
reused/foreign PID. Cleanup grace is clipped to the existing deadline. Uncertain
ownership or cleanup preserves evidence and retains the shared-lock lease rather
than permitting another launch. The original two-hour queue, three-hour absolute
application cap, per-phase limits and 6 GiB storage floor remain fixed.

The [pidfd adapter](candidate_continuation_pidfd.py) prefers CPython's bindings.
For the reviewed 64-bit Linux x86_64 ABI only, a bundled Python missing those
bindings uses the same kernel pidfd operations through typed libc syscalls
434/424, verified against the Ubuntu installed headers. Kernel/API errors still
fail closed; there is no numeric-PID or process-group signalling fallback. All
existing identity/start/session/ancestry rechecks stay around the pinned fd.
An explicit capability preflight opens only the checking process, sends signal
zero and closes the descriptor before subreaper setup or any application start.
This preflight never authorizes starting or retrying an application.

The first authorized launch stopped before native-child creation because its
older source required the missing CPython bindings. Its failed-no-retry history
is preserved in the [launch evidence](../../baselines/isaac-candidate-live-pilot-20261109/README.md).
The compatibility fix passed 60 mock cases and its self-only capability probe in
the existing container before the separately authorized v2/v3 attempts.

## Reviewed inputs and one native application

The manifest schema is `sircus-candidate-continuation-production-inputs-v1`.
It binds canonical **host** paths under `workspace_host`; the entry maps them
read-only into the existing `/workspace` mount without rewriting manifest bytes.
Required metadata includes `source_host`, `source_run`, `assets_dir`,
`frozen_experiment_root`, protected roots and exact expected Python/Torch/NumPy/
SciPy versions. Version mismatch fails before app startup; no environment change
is attempted. Preparation's recorded runtime remains separately disclosed.

`files` entries have exactly `path` and `sha256`. Required roles are listed in
[`ROLES`](candidate_continuation_production.py): frozen scene/camera/poses/URDFs,
parent checkpoint, approved TRAIN inputs/bank/demo evidence, all fresh split/
support/audit/export artifacts, and original source contract/result/menu/camera/
startup metadata. Every generated simulation asset is also byte-bound using
`asset:<relative-path>` roles; undeclared/linked assets fail validation.
`historical_reference` must be the full audited 35-case comparison report,
including bank SHA, coordinates and the original 13/16 capture witnesses. The
compact count-only baseline summary cannot substitute for it.

The pre-start `native_contract` binds scene/assets/poses, fixed TRAIN support and
prepared records, source and unchanged controller/guard identities. Its hash is
the launch plan's `native_binding_sha256`. This is distinct from the full native
binding created **after** actual startup control bytes exist; no future startup
hash or observation is guessed.

The entry validates inputs and source metadata before its sole `AppLauncher`.
It also checks the original CPU `torch.tensor` representation of the six URDF
effort limits. Both this preflight and the producer require the exact original
float32 dtype, shape, finite values and bits; a one-ULP change fails. The actual
GPU limit construction and controller remain unchanged. This avoids mistaking
the original float32 representation for changed float64 limit values.
It compiles only the unchanged film source's definitions and scene-construction
prefix, ending at its origins-equality gate. It never executes that file's
top-level app launcher or old training/filming main loop. Original scene,
materials, selected camera and controller methods remain unchanged. The source
file and startup helper bodies are preserved byte-for-byte.

[Startup observation](candidate_continuation_scene.py) temporarily records original
ball-placement/tick calls without adding either. It saves all 35 initialized
states and per-step contact forces, restores inherited methods before paid work,
and independently rechecks placements, clocks, force gates and original control
files. The original rim early stop, five contact cohorts and one foreign-cell
isolation cohort remain bounded by **14 cohorts / 490 initialized trajectories /
1,824 physics steps**. Parking resets stay separate. No second diagnostic pass
or smoke release is possible after failure.

## Learning and final-only assessment

[Application orchestration](candidate_continuation_application.py) wires the
existing 240-step actor-only imitation and bounded categorical PPO driver to the
unchanged guarded native producer. Tiny neural updates use the existing CPU
engine/dedicated CPU RNG; articulated PhysX and rendering use the 4090. Network,
reward, action/support encoding, release gates and budgets do not change.
Only the 35 TRAIN contexts and their 1,534 existing successful examples enter
updates. Requests are persisted before dispatch; all rejects stay charged.
TRAIN stops at the first **1,400 releases or 1,750 requests**, with no resume.

The final checkpoint is saved and hash-frozen before any assessment outcome.
The [assessment planner/runner](candidate_continuation_assessment.py) loads
separate frozen parent/final models and freezes all **420 deterministic choices**
before dispatch. Both planning and audit recompute readouts through the same
pinned CPU forward path. Prepared parent means/actions/rankings must preserve
the frozen reference within declared numerical tolerances; candidate encodings
and original parent actions are never replaced or clipped.

The three methods are original continuous seed 22, frozen parent bank selector,
and final continued categorical policy. Each receives the same 35 known TRAIN,
35 new validation and 70 new TEST positions. Cohort rotation is fixed:

| Split/cohort | Order |
| --- | --- |
| TRAIN | parent, bank, continued |
| validation | bank, continued, parent |
| first TEST half | continued, parent, bank |
| second TEST half | parent, bank, continued |

The [assessment native adapter](candidate_continuation_assessment_native.py)
reuses the same app, canonical home and original native methods/guards. Parent
actions stay unfiltered; bank and continued choices share immutable supports.
The three empty fresh supports cause six paid bank/continued declines, remaining
in denominators. Empty slots cannot secretly release. There is no held-out flight
search, replacement request, update, checkpoint selection or repeated assessment.

## Independent audit, reporting and compute scopes

After native app close, a fixed child command with `CUDA_VISIBLE_DEVICES=''`
runs the independent reader with a 600-second timeout. It joins four components:
existing learning/checkpoint/seeded-schedule verification, native TRAIN guard/
physics/reward audit, measured startup audit, and the new
[420-request assessment audit](candidate_continuation_assessment_audit.py).
Results, checkpoints, journals, source, support, raw packets and traces are
cross-bound. Missing, partial, altered or repeated evidence cannot yield a
completed audited application result. The structural learning audit still does
not independently replay intermediate Adam arithmetic.

The report's primary result is **TEST captures / 70 requested positions**, then
clean captures, rejections/reasons, releases, capture conditional on release and
paired per-position gains/losses. Paired 95% bootstrap intervals use 10,000 fixed
resamples with reporting seed 20261128. They describe this shared grid and one
continuation, not uncertainty over independent training seeds. Historical 13/16
TRAIN witnesses are retention references, separate from newly paired controls.
Every complete/failed report retains `model_promoted=False`.

The separate [read-only final publisher](candidate_continuation_report.py)
requires external hashes for the closed application, combined audit, host launch
result and pinned manifest, plus the expected session and producer source. It
joins all four passed audit components, exact final checkpoint bytes, complete
request ledgers and the final TRAIN journal before constructing a continuation
report. It recounts comparison denominators and derives the complete exploratory
TRAIN curve; it does not load a model, rerun physics or write a file. The host's
`native_complete_independent_audit_required` status alone cannot establish an
audit pass.

Its additive update preserves the historical 70-request report root and nested
prior attempts. It refuses to replace another active session. Stop the finite
TRAIN-only publisher before atomically publishing the final audited child through
the existing GET-only reader; no reader restart is needed. Final checkpoints
remain separate saved artifacts, and this continuation protocol records no new
trial videos. Existing frozen benchmark videos remain in their original reader.

Entry phase timers separate initialization, imitation, TRAIN, assessment and
audit. CPU is self plus waited/reaped-child deltas; active unreaped child CPU is
not included until reaped. Host supervisor/watchdog overhead is separate. RSS is
process-lifetime high-water; child RSS is the largest child, not a sum. Torch
CUDA peaks include persistent scene tensors; sampled device occupancy includes
renderer/PhysX and is not model-only memory. The CPU reader's own cost overlaps
the parent audit phase and must not be added twice. Previous bank preparation,
inherited 10,000 parent requests / 73 updates and the prior 70-request comparison
remain separate from new work. Unmeasured device values stay null.

[Compact verification](../../baselines/isaac-candidate-runtime-assessment-20261109/README.md)
records mock/synthetic checks and source hashes, not model performance. Before
any actual execution, the source snapshot, manifest, current dependency versions,
GPU/lock ownership, storage and untouched output must pass their implemented
checks under a separately authorized invocation. Source checks and passed
component readers grant no permission to retry or resume a closed application.
