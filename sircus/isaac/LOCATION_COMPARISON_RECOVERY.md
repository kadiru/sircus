# Saved-data audit after the comparison's shutdown failure

The user approved implementation, mock testing and a bounded CPU-only audit on
November 18, 2026, followed by publication if verification succeeds. This is the
separate proof described in the [recovery plan](LOCATION_COMPARISON_RECOVERY_PLAN.md).
The original run, failed shutdown receipts, original auditor and historical
benchmark remain unchanged. No training, optimizer step or new shot is part of
recovery.

## Status and scope

Implementation and synthetic verification are separate from a passed native
data audit. Actual execution must still satisfy the original **6 GiB disk floor**;
the preparation check initially found storage below that floor. User-approved
downloaded APT package cleanup reclaimed 911,401,090 bytes and restored
7,089,160,192 free bytes at that observation. Installed environments and evidence
were preserved. No scientific recovery result is established by the mock tests
or the earlier file inventory.

The saved comparison contains producer-reported counts of 8,960 TRAIN requests
per copy and 600 assessment requests, with final checkpoint files present.
Recovery must independently verify all 18,520 requests, 539 terminal batches,
128 updates and 144 checkpoint artifacts before publishing any performance.

## What changes in the proof

- The original clean-exit audit remains unsuccessful. Recovery checks the
  preserved deadline failure, exact retired process/container identity and safe
  historical lock release instead of inventing the missing child-exit receipt.
- `location_comparison_recovery_verify.py` imports the original helper readers
  from producer `300ab5c`. Runtime source and AST checks require its copied
  scientific audit body to match the original, except for the approved exit-proof
  binding and report status. Model, likelihood, geometry, motion, release and
  full-rate scoring checks retain their original code and coverage.
- The complete sealed inventory is checked before and after the scientific
  reader; the independent supervisor checks it again after child exit. Persisted
  journal files must exactly match the final trace. Original inputs/source are
  byte-checked, and original evidence is mounted read-only.
- The resulting report, if successful, says `data_verified_shutdown_incomplete`,
  with `original_audit_passed=false`, `clean_shutdown=false` and
  `model_promoted=false`. It is not an original clean-completion certificate.

The original verifier SHA256 is
`0e196852b48070d423d4db61b894b93b36edc655bc5f92c77adbf6b3815eccdd`.
The unchanged scientific function's source SHA256 is
`6bef2f77aa1800047d4c1d4a0416a2cca1c3d2cb475cb7895fae470a5f1f8947`.
AST comparison is semantic and interpreter-independent; the reported source pin
uses the exact original function bytes rather than a version-dependent AST dump.

## Execution boundary

`location_comparison_recovery.py` is an explicit entry, not an import-time job.
Its three recovery Python modules must live outside the immutable producer
source, with exact membership and externally pinned hashes. The signed private
request binds the original manifest, source, evidence inventory, fresh process
retirement proof, distinct CPU session and fresh audit destination. Original
paths cannot be reused as output destinations.

The entry checks original source bytes before importing helpers and blocks
simulator imports. A separate isolated CPU container must expose no GPU or
network, use a read-only root and original evidence mounts, and allow only its
new audit output to be written. Installed numerical versions must match the
producer environment. No package installation is required.

The independent supervisor enforces one child start, at most two hours, the
actual evidence filesystem's 6 GiB free-space floor, at most four CPU cores and
8 GiB memory. A temporary memory filesystem cannot substitute for the real disk
floor. Original process-identity and pidfd cleanup helpers are preserved.
The child checks live ownership before scientific work and before report
publication. Exact-child `wait4` measures CPU time and peak RSS; cgroup limits
and sampled memory are recorded separately. Failure preserves a separate error
receipt and does not retry.

A second, independent host watchdog bounds the entire newly created CPU
container, including a blocked integrity read or final inventory check. It pins
the complete container ID, name, image, session, entrypoint/command and mounts;
requires the declared CPU isolation; and starts it once only from `created`.
It reserves 30 seconds within the two-hour limit for identity-checked cleanup.
It neither creates nor restarts/removes containers and never acquires the scene
lock. The private creator separately verifies source/control bytes before
creating the new container. Docker logs are size-bounded and retained.

The prepared modes are:

```text
location_comparison_recovery.py --request <pinned-control.json> \
  --expected-request-sha256 <sha256> --producer-source <immutable-source> \
  --preflight-only

location_comparison_recovery.py --request <same-pinned-control.json> \
  --expected-request-sha256 <same-sha256> --producer-source <immutable-source> \
  --execute-approved
```

Preflight validates source/runtime, process-retirement bindings, journal and
inventory without a scientific audit or child start. Production invocation must
be through the reviewed isolated container with the declared resource limits.
It is not a command to run against arbitrary host paths or environments.

## Publication

`location_comparison_recovery_publication.py` has a distinct gate. It requires
externally pinned report, recovery-exit and original failed-watchdog hashes;
the canonical publication gate still rejects this failed native run. Recovery
publication joins report/process/owner/source identities, all scientific
coverage flags, exact checkpoint/trace bytes and the original scheduled media.

The interface displays the shutdown failure alongside TRAIN curves, frozen
monitoring, the shared 70 broad and 25 nearby TEST positions, and measured costs.
The original watchdog lifecycle includes failed shutdown. The stale final
shared-phase snapshot and time after the last progress update have separate
labels; neither is presented as pure training or pure shutdown compute.

Only the ten existing prescheduled native recordings can be exported: eight
TRAIN clips and two parent-only TEST subsets. There were no scheduled final-model
TEST videos. Encoding verifies original ordered JPEG bytes and all frames,
duration, dimensions, chapters and full decode. No simulator replay or new shot
is permitted to obtain additional footage.

This remains a single paired continuation comparison. It cannot establish
repeated-seed certainty, full collision-free path planning or physical hardware
validation, and the CPU model checks do not replay CUDA Adam arithmetic bitwise.

## Prepared implementation verification

The repository run completed 1,803 Isaac cases with 121 optional skips. The two
localhost sandbox errors passed in their four-test file reruns. All 34 physical
mocks, five simulation mocks, 73 Node cases and nine numerical scripts passed.
The independent outer watchdog additionally passed 32 targeted mock tests after
requiring explicit Docker local-log compression disablement with its one-file
limit. Source review found no remaining blocking issue. These checks verify
software behavior with synthetic data; the actual saved-data audit is pending.
