# Seed-22 location comparison: actual-input CPU preflight

**The real-input and CPU model checks passed. The new comparison has not run.**
The source package is pinned to `89b0afd3399b5eb7e5fa335c4d7a16ffbd0994ab`;
the machine-specific manifest and full receipts remain private. See
[`summary.json`](summary.json) for exact hashes, counters and measurements.

- All **101 input files**, **217 Python source files** and the complete
  **80-file asset inventory** matched their bindings. The prepared coordinates,
  original experiment, scene, source helpers and passed seed-22 audit matched.
- The original seed-22 checkpoint restored its weights and complete finite Adam
  state on CPU: **10,000 inherited TRAIN requests, 73 PPO updates, 292 optimizer
  steps**, with learning rate 0.0003.
- Its predictions at all **35 original TRAIN positions** matched the saved final
  readouts. Maximum normalized action error was **1.7881393432617188e-7**, below
  the pre-existing **5e-6** tolerance. No held-out outcome packet was evaluated.
- The successful CPU process took **3.074 s wall**, **2.639 CPU-seconds** and
  **667.5 MiB peak RSS**. Container start-to-exit took 3.813 s. These measure this
  preflight only; they are not training, simulator or policy-performance costs.

## Isolation and preserved history

The existing simulator container was stopped and stayed stopped. The check used
the exact already-local producer image in a new disposable container: no GPU
devices, no network, read-only root/source/experiment mounts, dropped
capabilities and no new privileges. Limits were two CPUs, 4 GiB RAM, 128 tasks
and 300 seconds. CUDA was never initialized; simulator imports and optimizer
steps were explicitly blocked in the private preflight harness.

The first isolated CPU attempt exited 126 before Python could start because the
image's default user could not read the private source directory. Its logs and
0.201-second container cost are preserved. A second isolated container used the
host file owner's UID and a temporary home. No source, dependency, input or
directory-permission change was needed. Both temporary containers were removed
after their identities and exits were verified.

All bound input/source bytes matched again after the check, and the prospective
native and audit output directories still did not exist. Free storage was
17.109 GiB, above the unchanged 6 GiB floor. This task performed **zero simulator
starts, optimizer steps, new TRAIN requests or assessment requests**. The frozen
benchmark and previous experiments remain preserved.

## What is prepared next

The [fixed comparison](../../sircus/isaac/LOCATION_COMPARISON.md) still starts two
copies from the original seed-22 checkpoint. One receives 8,960 requests across
31 admitted original locations; the other receives 8,960 requests at broader,
distinct locations. Four original Isaac positions overlap arm bases and remain
excluded by the already-prepared geometry contract. The two exclusions in the
different mjlab scene are not substituted for these four.

The parent and both final models share the same fresh **70 broad plus 25 nearby
TEST positions**. With 315 monitoring requests and 285 final requests, the
complete proposed comparison totals **18,520 requests**. No budget, target,
sampling or model-selection rule was changed during preflight.

A real run remains a separate user decision. Current GPU/lock/storage ownership
and original native startup/physics guards must pass at launch. This CPU result
does not prove simulator readiness or improved catches. The previous
[142-test mock verification](../isaac-seed22-comparison-production-20261117/README.md)
and this actual-input check cover different evidence.
