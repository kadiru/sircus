# Candidate learning driver: software verification only

The bounded update engine, persistent TRAIN driver, new checkpoint schema and
independent structural/saved-state audit are implemented. **Real seed-22
training remains off.** No new native shot or capture result belongs here.

| Check | Result |
|---|---:|
| Focused CPU cases executed / failures / errors / skips | 103 / 0 / 0 / 0 |
| Fabricated-policy imitation schedule, one end-to-end case | 240 steps / 30,680 presentations |
| Mock TRAIN requests / releases in that case | 1,400 / 1,400 |
| Mock PPO updates / steps in that case | 10 / 40 |
| Final synthetic save/load and structural saved-run audit | Passed |
| Real seed-22 frozen-selector choice parity | 35 / 35 |
| Real-parent optimizer steps / backward calls | 0 / 0 |
| New native requests / task releases / held-out shot files read | 0 / 0 / 0 |

[Verification receipt](verification.json) binds all 21 continuation module/test
files, the complete 350-file development snapshot archive, input and log hashes,
original experiment/split/audit/checkpoint bytes, the frozen reader report and
test scopes. [Actual-parent readiness](readiness.json) records the separate
CPU forward-only check: 1,814 qualified candidates, 1,534 exact demonstrations,
485 clean at 21/35 contexts, unchanged weights/RNG and no gradients.
Raw inputs, synthetic logs and all checkpoint bytes stay outside Git.

The focused CPU process took 98.207 s; tests themselves reported 97.369 s.
The full verification, including separate parent compatibility, took 100.285 s.
These measure software tests, not native or production training speed. Standard
project mock/numerical checks also passed: the Isaac portion ran 858 cases with
120 environment-dependent skips. All relevant tensor cases executed in the
focused CPU suite above. Existing environments were used without installation;
CUDA was hidden. No viewer or native job was restarted.

The fixed limits remain 1,400 releases or 1,750 requests, whichever comes first.
Ledger v2 also ends active TRAIN batches at 140-request optimizer boundaries,
preventing smaller rejected batches near the release cap from straddling an
update. Context order, caps, physics, scoring and guards are unchanged. The new
compact export has identical supports and demonstrations; only source bindings
change. Historical receipts and failed checks remain preserved separately.

Tests cover caps, final partial buffers, rejection coverage, detached old
targets, actor-only imitation, fresh optimizer states, dedicated RNG, stale or
nonfinite data, checkpoint tampering, atomic new-only writes, partial outcomes
and interrupted optimizer operations. A failed operation is incomplete and
cannot resume or buy replacement requests. The first full CPU snapshot caught
a NumPy-versus-Python float mismatch in a fabricated test input; its correction
changed the fixture, not a production guard. Two sandboxed project HTTP tests
needed temporary localhost-socket permission before the full suite passed.

The independent audit certifies structural accounting, schedules, exact
candidate identity, terminal coverage, detached advantages, original reward
constants/bounds and final saved-state bindings. Verification additionally
checks seeded demonstration order, RNG consumption and objective-term sums.
It does **not** replay intermediate Adam/gradient arithmetic or certify native
physics, actual sampled-choice probabilities at each intermediate weight,
IK/path/settling/release guards, full trajectory rewards or capture labels.

See the [implementation contract and remaining gates](../../sircus/isaac/CANDIDATE_RUNNER.md).
Outcome-blind fresh preparation, actual native integration/supervision, full
native scoring/guard audit and final-only assessment remain prerequisites.
The [previous audited native comparison](../isaac-candidate-native-20261109/README.md)
remains 13/35 versus 16/35 captures; no learning improvement is claimed here.
