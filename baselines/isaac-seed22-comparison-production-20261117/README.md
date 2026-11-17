# Location comparison production connection — mock evidence only

The separate process watchdog, one-application entry and independent
model/geometry/physics reader are connected to the existing comparison runner.
No simulator, real model load, training job or native shot was started.

The full fabricated bridge/runner/verifier fixture covers **18,520 logical
requests**, **539 batches**, **128 simulated update calls** and **144 fake
checkpoint files**. Each copy retains the proposed **8,960-request TRAIN budget**
and **64 updates**. The shared final positions are unchanged: **70 broad plus
25 nearby**, used once by each of the parent and two final copies. Fabricated
captures, durations and memory are not model-performance measurements.

Tests exercise process identity reuse, phase and absolute deadlines, storage,
TERM/KILL escalation through fake pidfds, incomplete exits, source/trace/model
changes, likelihoods, original reader routing and report bindings. The full
synthetic comparison cannot be published as a native pass. Review additionally
fixed the startup ownership handshake, close-error cleanup, signed-receipt
encoding and immediate phase/reservation publication.

The independent model audit checks stored weights, Adam state/counters, RNG
identities and shot probabilities; it does not replay CUDA optimization bit for
bit. Native simulation and physical behavior remain unmeasured for this new
comparison. Final source/input packaging and actual-host CPU preflight are still
needed before a separately authorized real launch.

**142 focused tests passed with zero skips**, including 22 production-entry,
31 process-watchdog, 16 full-verifier and 14 native-bridge cases. All 17 pinned
original helpers matched. Exact checks are recorded in `summary.json`. Repository regressions also passed: 34 physical mocks, five simulation mocks,
1,654 discovered Isaac cases (121 optional-dependency skips), 59 JavaScript
cases and nine numerical scripts. Two localhost-binding cases initially blocked
by the sandbox passed when their two modules were rerun with loopback permission
(four cases). Final focused tests cover the last integration changes. Raw test
logs and synthetic artifacts remain outside Git.
See [implementation and boundaries](../../sircus/isaac/LOCATION_COMPARISON_PRODUCTION.md).
