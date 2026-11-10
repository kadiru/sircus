# V4 TEST evaluation: production wiring and independent checks

The evaluation-only production path is implemented and tested with fabricated
models, files, process identities and physics readbacks. **No real V4 model was
loaded, simulator started, training performed or new shot executed in this
task.** Passing these software checks does not grant execution permission or
establish a live native pass. The frozen benchmark and partial V4 assessment
remain unchanged.

The fixed proposal is still [210 untouched TEST requests](V4_TEST_EVALUATION.md):
70 positions each for original seed 22, its bank selector and saved V4. Exactly
four empty-support declines remain in those denominators. The original 140
charged / 105 terminal / 35 unresolved assessment stays `incomplete_no_retry`;
no request is refunded, replaced, repeated or resumed.

## Source, model and evidence boundaries

[Input validation](candidate_v4_inputs.py) keeps two identities: immutable V4
producer source `9a49a0d` / all-Python hash `a8c6955f…358c2`, and the separately
archived new evaluation source. The saved V4 checkpoint remains
`15020a948d295fe5e7258a732145b89aa61fda2c385a0d6966a9e4cc7e81bd41`;
its original training bindings are never rewritten to match the evaluator.

The new manifest binds nine exact file roles: prepared plan, parent and V4
checkpoints, original manifest, original learning result/native binding, old
assessment plan/inventory and completed-TRAIN component summary. Original
manifest roles and assets are independently rechecked as well. Protection must
cover every inherited evidence, bank, asset and source-run root, not only the
new manifest's direct inputs. The original 25-file partial-assessment inventory
must remain exact, with no new TEST dispatch. New output overlaps and symlinks
are rejected.

The evaluated commands are exact copies of original pre-frozen cohorts 6–11.
New request IDs are mapped to original IDs in hash-bound envelopes; original
raw packet IDs, actions, candidate preferences and IK seeds remain unchanged.
No new held-out action search or model selection is introduced.

## Execution path, when separately authorized

1. [The host supervisor](candidate_v4_runtime.py) verifies the source archive,
   manifest, model bytes, original source, protected-root closure, shared scene
   lock, existing container and dedicated 4090 ownership. Its authorization is
   bound to a fresh session, exact output, manifest, source and prepared plan.
   Construction and default supervision cannot start a child. There is no
   container restart or foreign-GPU termination operation.
2. [The independent watchdog](candidate_v4_watchdog.py) starts one entry process
   and enforces limits even if a callback blocks. It continuously checks full
   Python-source membership and bytes, manifest, storage, progress chain,
   counters and phase deadlines. Process cleanup requires PID, start token,
   cookie/ancestry and pidfd ownership checks. A raced foreign PID is skipped
   while other proven owned workers are stopped; uncertainty retains the lock.
3. [The model reader](candidate_v4_model_check.py) starts a fresh **CPU-only**
   child in the pinned producer environment. It verifies exact checkpoint,
   original input/source and runtime identities, imports the original producer
   implementation in a clean module path, loads only the frozen models and
   reconstructs the complete original 420-command plan independently. It then
   checks all 210 wrapped TEST commands. It neither restores an optimizer nor
   takes a learning step. Its durable receipt is frozen before App construction.
4. [The native adapter](candidate_v4_native.py) constructs only the App, original
   farm and startup diagnostics. It creates no learning engine or training
   runner. Original geometry, camera, physics, scorer, control gains, effort
   representation, IK and release guards stay fixed. Each 35-slot reservation
   is durable before its one native call. Empty supports retain the actual
   `no_selected_setting` native label and inactive qualification, with a charged
   semantic decline; they are never replaced by another candidate.
5. [The application](candidate_v4_application.py) runs only initialization,
   evaluation and audit. It preserves terminal prefixes and unresolved charged
   work on any error and never retries. It closes the App before the CPU audit.
6. [The audit entry](candidate_v4_entry.py) binds exact saved result, model,
   native-binding, startup and plan bytes in a new audit request. A bounded
   CUDA-hidden child reconstructs the models again and compares the immutable
   proof with its pre-App counterpart. [The native reader](candidate_v4_native_audit.py)
   then checks startup, the six reservation/envelope/raw-packet joins, physics
   hashes, original guard/scorer reconstruction and all terminal outcomes.
   Real audits cannot substitute synthetic reader callbacks.

The model reconstruction after shutdown is an independent verification of the
same frozen commands. It does not read outcomes to change commands or choose a
checkpoint. Source and input bytes must remain unchanged throughout.

## Fixed bounds and result gates

| Boundary | Limit |
| --- | ---: |
| TEST requests | 210; six cohorts of 35 |
| TRAIN / validation requests / optimizer steps | 0 / 0 / 0 |
| Initialization | 600 seconds |
| Evaluation | 1,800 seconds |
| CPU audit | 600 seconds |
| Absolute application wall | 3,000 seconds |
| Free storage | At least 6 GiB |
| Entry-process start intents | One; no retry/resume |

A process-start intent does not prove that the simulator started: model/input
checks may fail first. Host/watchdog receipts retain this distinction; a full
successful application can certify its one App start. Startup diagnostic
trajectories are separate from paid TEST requests.

A native-complete result remains `native_complete_audit_pending`. Only the new
complete independent model/startup/physics audit permits a separate
`completed_audited` application result. This never completes the old assessment
or promotes V4. Synthetic runs instead use explicit synthetic statuses and
`full_native_audit: false`. A failure file, partial cohort or changed proof
cannot produce completion.

Report captures and clean captures divided by all 70 requested positions for
each method, rejections by cause, releases, and paired gains/losses. New
initialization/evaluation/audit wall and CPU measurements stay separate from
inherited V4 training and earlier failures. RSS is a lifetime high-water mark;
GPU occupancy includes the simulator/renderer. Overlapping audit-child and
application costs are not added together. No new videos are recorded by this
protocol.

## Software verification and current limits

Focused tests use mocks only, including the unchanged original batch method
body with fake physics. They cover model-before-App ordering, raw disabled-slot
labels, complete 210-slot envelopes, corrupt packets/physics, frozen model and
source changes, protected-root closure, partial prefixes, failed audits,
blocked callbacks, reused PIDs, mixed owned/foreign cleanup, source-inventory
changes, stale-clock races, storage/deadline limits and default execution refusal.
Independent reviews cross-check the production, model, physics and watchdog
paths rather than relying only on their producers' claims.

All **110 new mock cases** passed. The complete repository run also passed:
1,324 Isaac cases (121 optional dependency skips), 34 physical mocks, five
simulation cases, 59 Node cases and all nine numerical check scripts. Temporary
localhost socket permission was enabled for the test servers; existing readers
were not restarted. [Compact software evidence](../../baselines/isaac-v4-production-mocks-20261109/summary.json)
contains counts, bounds and source identities, without raw observations or
runtime configuration.

The real simulation host was not deployed to or queried in this task. No actual
checkpoint inference, native startup, physics replay or resource measurement is
claimed by these mock results. A future launch still needs a fresh immutable
package/manifest with the new source identity, actual-input CPU preflight in the
pinned producer environment, current host/GPU/lock/storage checks and separate
execution authorization. These checks must use the original producer container;
CPU-only native simulation is not supported by this protocol.
