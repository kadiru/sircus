# Comparison runner mock validation — November 16, 2026

**71 tests passed. Zero simulator starts, native requests, real policy loads or
real learning steps. No performance improvement was measured.**

The runner connects the existing policy/native helpers to the prepared
original-versus-broader-location comparison. The complete prepared package was
exercised with fabricated results and independently audited:

| Accounting check | Verified synthetic count |
|---|---:|
| Original-location TRAIN requests | 8,960 |
| Broader-location TRAIN requests | 8,960 |
| Simulated updates per trained copy | 64 |
| Monitoring requests | 315 |
| Final TEST requests | 285 |
| Total logical requests | 18,520 |
| Terminal batches | 539 |
| Durable trace events | 1,764 |

Each model's final TEST denominator is 70 broad plus 25 nearby positions.
Rejected TRAIN requests remain in both the budget and the PPO input buffer.
An incomplete group stops the run before final comparison; no paid work is
retried or refunded. Tests also exercise the native bridge class itself with
fake environment/policy dependencies through the full runner and auditor.

The prepared package fingerprint remains
`440cbcca29612820d3b36005010edd90fd4f8172646bd8ea95e57f154775f4e6`.
The original seed22 checkpoint, prepared locations, four excluded bucket
placements, completed benchmark and original physical/reward helpers are intact.

Independent review found and fixed false-return guard handling, assessment RNG
preservation and evidence-directory durability gaps. Source/input tampering,
import shadowing, mismatched model/target identities, omitted/repeated requests,
reward/latent edits, changed test order, wall/storage limits and crash prefixes
are covered by mocks.

Private receipts are under `.runtime/seed22-comparison-runner-20261116/mock-v2/`;
raw traces and synthetic checkpoint files are not in Git. See
[summary.json](summary.json) for compact hashes and counts, and the
[runner documentation](../../sircus/isaac/LOCATION_COMPARISON_RUNNER.md) for scope.

**Production boundary:** the real startup/scene owner, OS-level watchdog and
full independent model/physics verifier still need the outer launcher hookup.
The implemented bounds observer and accounting audit do not replace them.
No native launch command, hardware action, service restart, training or new
shot was invoked or authorized by this task.
