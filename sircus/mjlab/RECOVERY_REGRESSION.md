# Recovery capture-loss diagnosis and one height intervention

This is a frozen **63-request development experiment**, not new training or a
new held-out evaluation. The five inspected losses from the completed recovery
comparison are explicitly development data now. All original reports, score
definitions and model weights remain unchanged. No active policy is replaced.

## Evidence and hypothesis

Parent `coverage-recovery-v1` completed 970 requests and passed 283,251 native
audit checks. Recovery BC captured 37/70 versus parent BC 34/70, gaining eight
and losing five. Exact parent report SHA-256:
`6a255223820656059fe16a5de2944ba2593f7ac3603b287f4955979cd4677504`.
Independent parent proof:
`e2ba653792b37d302425ba4af6cb22460c3d5b426274b3c1b3cbce270a857a9b`.
Parent fingerprint:
`f54c341eba5258b427e4e1eb05a1b2e93bf76518d9f94a2fe8cf6bad744f6082`.

CPU-only saved-evidence diagnosis SHA-256:
`511e9f82004af0d603628e25adf8fab71284faa8f11de9307b4c8a611d049ab7`.
Its separate 3,103-check proof SHA-256:
`911101291f8119611f05b071980d408c658984074557976ef153ddf8acbd97a6`.

Three losses became setup declines: one had zero admitted dropper IK branches
at 257.834 mm instead of the parent's 282.162 mm board height; another had zero
board IK branches at 478.303 versus 460.706 mm; the third failed the table–Arm14
link 5 clearance gate at 241.129 versus 280.424 mm. Finite IK decline does not
prove an unreachable pose. The two released losses had no forbidden contact
and no whole-ball entry; their drop gaps grew 30.567 and 37.745 mm. Height is a
focused admission hypothesis and need not fix those two flights.

[The literal rule](recovery-regression-height-correction.json) restores only
action index 2 (`board_height_m`) to the corresponding captured parent-BC
value. Board X/Y, tilt and drop gap remain exact recovery-BC values. **With gap
fixed, derived release-center height changes by the same board-height delta.**
There is no compensation to keep absolute release Z fixed.

| Former TEST context | Recovery board height (mm) | Restored parent height (mm) |
| --- | ---: | ---: |
| recovery-test-r0-c5-n1 | 424.899238 | 428.821926 |
| recovery-test-r0-c6-n1 | 443.245177 | 433.084086 |
| recovery-test-r1-c1-n1 | 257.833776 | 282.162101 |
| recovery-test-r2-c4-n0 | 478.302747 | 460.705580 |
| recovery-test-r3-c1-n0 | 241.129048 | 280.424275 |

The worker derives full-precision coordinates directly from authenticated
parent rows. Rounded values above are descriptive and never commands.

## Complete pre-outcome plan

1. Three exact original 17-source direct captures, three table-assisted captures
   and three expected misses: nine formal pre-controls.
2. Five literal contexts, each with three exact parent-BC requests, three exact
   recovery-BC requests and three corrected recovery-BC requests: 45 requests.
3. The same nine formal controls after all diagnostic requests.

All 63 requests and both source-row hashes are frozen before any new native
outcome. No additional coordinate search, retries, outcome-dependent settings,
or winner selection is allowed. The five parent captures are diagnostic
positive controls: instability is reported, never a reason to remove a case or
abort. Only the two formal direct/table/miss control blocks are hard gates.

Original admitted branches and joint proofs are copied exactly. Original
setup declines are **cached authenticated refusals**, counted in the requested
denominator; they are not three new IK searches and never release placeholder
joints. Each changed action gets a new strict connection proof under unchanged
admission guards, reused unchanged for its nominal repeats. Any corrected
decline remains a failed requested outcome. No failed cell is replaced.

Diagnostics use the same 19-source prior and `native/` evidence namespace.
Formal controls retain the original 17-source prior and
`native-controls/`, hydrated from the parent's `control-parent/native/` proofs.
Both share one unchanged lazily allocated native64/v3 environment; no proof
from one prior can stand in for a proof from the other. Table-assisted captures
retain their v3 credit. Scoring, geometry and validation thresholds do not change.

## Runtime limits and audit

Only RTX 4090 UUID `GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4` is permitted.
The worker obtains its own cooperative lease, refuses foreign 4090 work, has a
maximum 900-second queue and 3,600-second acquired slot, and preserves at least
8 GiB free storage. The supervisor has one original 6,300-second absolute cap
including a 1,800-second independent CPU audit reserve. It may signal only its
own child. No extension or restart resumes partial evidence under the same name.

Worker, request packets, parent artifacts, diagnosis, exact rule and full source
snapshot are bound by hashes. The independent CPU auditor reconstructs all 63
requests without the producer plan builder, verifies unchanged coordinates and
joint branches, separately checks both geometry banks, and rescores every
released dense trace with the independent v3 oracle. Actual ball release,
held joints, native timing/options and contact geometry are checked. It retains
every setup decline, ordinary miss and invalid-physics attempt separately.

The pipeline passes only after all planned slots and formal controls complete
and the independent audit passes. Partial or failed evidence remains visible
and unverified. No retry follows an undesired scientific result.

The 4070 fountain, Isaac, physical hardware, Viam configuration and running
services are outside this experiment. There is no noise trial, 128-substep
qualification, learned arm transit, full dynamics claim or hardware validation.

Entry points: `python -m sircus.mjlab.recovery_regression_pipeline` runs the
finite worker then independent CPU audit; `recovery_regression_verify` audits
saved evidence without initializing CUDA. Source synchronization is not launch
or deployment; the root operator records the frozen commit and checks live
GPU/process state before the single authorized experiment.
