# Location comparison: runner connection and mock validation

The [prepared comparison](LOCATION_COMPARISON.md) now has a runner and an
explicit bridge to the existing PPO and simulator helpers. **This task used
fake results only. It started no simulator, loaded no real policy, performed no
gradient updates and issued no native shot requests.** Original experiments,
checkpoints, prepared coordinates, physics and reward helpers are unchanged.

## What is connected

`location_comparison_runtime.py` drives three separately restored model handles:
the untouched parent, the copy repeating the 31 admitted original locations,
and the copy using broader locations. It checks matching parent weights, Adam
state, inherited counters and paired random streams before the first request.
Each trained copy receives exactly 8,960 requests in 256 batches of 35, with
one four-pass PPO update after each 140 fresh requests. No rejected attempt is
removed from the budget or learning buffer.

The runner saves the assigned coordinates and request IDs durably before
asking for a proposal. It records the original observation, latent, likelihood,
executed result and reward separately. A failed write blocks dispatch. Missing
results or a failed native batch remain charged and stop the comparison; there
is no retry, refund, resume or best-checkpoint selection. A failed group does
not proceed to final TEST.

The single shared parent monitor pass and four monitor passes for each trained
copy cost 315 requests. Final checkpoints are saved and fixed before any final
TEST. All three models receive the same 70 broad and 25 nearby targets, once
each: 285 final requests. Model order rotates by target, while targets with the
same model/rotation are batched to retain parallelism. There are 539 terminal
batches and 18,520 logical requests across the complete comparison.

`location_comparison_native.py` is an explicit, inert-on-import bridge. A caller
must supply an already owned native scene, exact input/source bindings, a new
output directory, guards and separate execution authorization. The bridge:

- Sets each slot's assigned bucket location, calls the unchanged home/reset
  method and verifies actual bucket readback before a shot batch.
- Pads smaller assessment groups to 35 slots with inactive entries. Padding is
  never a paid request, training example or reported outcome.
- Calls the original policy sampling/update and native geometry, motion,
  settling, release and scoring helpers. It changes no model architecture,
  reward or repair limit and adds no lookup table.
- Gives each model independent action and optimizer RNG streams. Deterministic
  assessment and checkpoints cannot consume training randomness. Optimization
  cannot consume the action-sampling stream.
- Keeps full-rate native physics/geometry evidence, raw proposals and partial
  failures, and creates new-only checkpoints. Evidence files and their parent
  directories are synced before they can support completion claims.
- Measures episode CPU/wall time, geometry time, sampled device memory and
  lifetime RSS; orchestration records proposal, restore, update and checkpoint
  wall times. Geometry remains a nested cost, not extra elapsed time.

Production input checks verify actual parent/experiment/scene bytes, declared
source membership and 17 unchanged critical helper hashes. Imported Python
helpers must resolve to the verified checkout. The legacy continuation input
manifest deliberately cannot serve as the new manifest: its source membership
does not include these new modules.

## Bounds and independent accounting

`location_comparison_bounds.py` enforces the prepared request caps, cumulative
eight-hour limits for each training group, a cumulative shared hour, and the
6 GiB storage floor. A reservation consumes its budget before dispatch. A late
terminal result is still counted, but no following request is allowed.
`SnapshotWatchdog` can detect stale progress from its own clock and reject
counter/time rollback. It is an observer, not a process launcher or killer.

`location_comparison_audit.py` independently reconstructs the expected schedule
and checks the event hash chain, model identities, coordinate/slot joins,
original PPO inputs, retained rejects, exact updates, assessment non-learning,
checkpoint freeze, counts and compute scopes. Final reports aggregate all
rotations into 70 and 25 requests per model. Interrupted traces retain unresolved
charges and cannot be relabelled complete. Fabricated evidence cannot be
relabelled as native evidence.

The audit is an **accounting and evidence-binding audit**. It does not by itself
prove checkpoint bytes, model likelihoods, native physics or actual process
ownership. Those require the separate production verifiers. A native trace
without the external verifier remains explicitly physics-pending.

## Mock-only entry and tests

`location_comparison_mock.py --mock --plan <prepared-comparison.json> --output
<new-absolute-directory>` accepts the exact prepared package and runs only the
fabricated engine. There is no native-start CLI in this change. It writes a
synthetic trace, audit and checkpoint fixtures; fake capture rates and clock
values are test data, not robot-performance measurements.

The **71 tests** cover preparation, runtime, native bridge with fake
dependencies, bounds and independent audit. The full prepared package completed
18,520 fabricated requests and 128 simulated update calls, with zero real shots
or learning. A second complete integration test drives the actual bridge class
with an invented environment and policy dependencies, including padded batches,
dynamic bucket readback and independent audit.

Review found and fixed three additional regression risks: ignoring an explicit
false guard return, failing to check assessment RNG preservation, and missing
parent-directory synchronization. Tests also cover changed source/input bytes,
shadowed imports, invalid state restoration, wrong coordinates, altered rewards,
duplicate examples, dropped rows, deadline/storage failures and crash prefixes.

## Remaining native boundary

This completes the requested mock-tested runner connection. It is **not yet a
self-contained production launcher**. Before any native execution, a separately
authorized outer entry must bind the new actual-input manifest, start/own the
scene and lock, run required startup checks, connect an identity-safe OS
watchdog that can stop a blocked native call, and wire the full independent
checkpoint/model/geometry/physics verifier. The observer and injected callbacks
are not substitutes for these components. No claim of native readiness or
improved catch rate follows from the mock pass.

See the dated [validation receipt](../../baselines/isaac-seed22-comparison-runner-20261116/README.md).
