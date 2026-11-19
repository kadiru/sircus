# Candidate native audit / launcher software verification

Mock and synthetic numerical verification only. No simulator, real policy
training, new task/diagnostic shots, host service changes or frozen benchmark
edits were performed.

[Implementation and exact limits](../../sircus/isaac/CANDIDATE_NATIVE_GATES.md)
describe the independent native TRAIN evidence auditor and mock-only bounded
launcher. The launcher deliberately has no production backend. A native TRAIN
audit does not replace checkpoint/learning integrity or the fresh assessment audit.

[`verification.json`](verification.json) records focused case counts and hashes
of the new code/tests. Focused cases cover scoring/reward tampering, saved guard
readbacks and full joint velocities, request/packet/ledger joining, ownership,
clocks/storage/source changes and one-start diagnostic limits. All test evidence
is fabricated; none of its counts are model performance.

All **77 focused cases passed, zero skipped**: 35 launcher, 13 evidence/ledger
joins, 15 guards and 14 score/reward cases. The local run used Python 3.12.14,
NumPy 2.5.4 and SciPy 1.18.1. These new cases need no Torch, simulator or GPU.

Raw/generated traces, local test logs, host configuration, observations and model
bytes remain outside Git. Existing baseline evidence is preserved.
