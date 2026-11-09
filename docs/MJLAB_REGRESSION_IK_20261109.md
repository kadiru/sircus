# Two remaining IK regressions — audited feasible-pose results

Date: November 9, 2026. Simulation only, fixed 64 substeps.

Both inspected contexts now have nearby feasible recipes with **3/3 captures**.
Restoring the successful parent's board X/Y on each original rejected recovery
action works for both cases. This discovers local alternatives under unchanged
strict admission; it does not fix the IK algorithm or establish that previously
rejected poses were impossible.

[Complete outcomes and exact compact data](../baselines/mjlab-regression-ik-20261109/README.md)
are the handoff source. [Frozen protocol](../sircus/mjlab/REGRESSION_IK.md) and
[literal rule](../sircus/mjlab/regression-ik-correction.json) define the finite
proposal grid before outcomes. Draft [PR #111](https://github.com/kadiru/sircus/pull/111)
stacks on #110; frozen producer/UI commit is
`2ec2c0baae8c9413d121d5c0b71c841b8108986a`.

## What ran and what passed

- **66 fixed requests**: nine formal controls, 48 diagnostic requests, nine
  formal controls. Diagnostics are two cases × eight variants × three requests.
  No winner selection or extra trials were triggered by a successful outcome.
- **33/48 diagnostic captures**, with 33 native releases, 15 setup refusals,
  zero flight misses and zero diagnostic robot contacts. All captures passed
  measured board bounce, whole-ball entry and one continuous second retained.
  All were direct board → bucket; this classification does not exclude rim
  assistance. Table-assisted capture remains allowed by unchanged v3 scoring.
- Exact parent recipes retained **6/6**. Exact recovery actions had six cached
  refusal slots and no releases. Of 12 new configurations, nine captured every
  nominal repeat (**27/36 new-action requests**); three declined setup.
- Both X/Y corrections and all 25/50/75% blends captured 3/3 per case.
  Parent height + gap worked only for `r1-c1-n1`. Parent tilt alone declined for
  both cases. Every admission failure stays in the denominator.
- Both formal control blocks passed: three direct captures, three table-assisted
  captures and three expected misses each. In all 66 slots there were 51
  releases, 45 captures, six expected misses, 15 setup refusals and no invalid
  physics. The six robot-contact shots were the intentional negative controls.
- Worker wall time: **203.8 seconds**, not isolated GPU compute time.
  Independent audit: **25,329 checks**, with native and full checks verified.

## Concrete corrections

| Context | Original failure | X/Y change from recovery | New board X/Y mm | Unchanged recovery height / tilt / gap | New valid branches, dropper / board | Result |
| --- | --- | --- | --- | --- | --- | --- |
| `recovery-test-r1-c1-n1` | Dropper 0, board 9 finite solutions | −5.892 / +23.525 mm | 635.421 / −392.848 | 257.834 mm / 32.196° / 236.932 mm | 10 / 8 | 3/3 |
| `recovery-test-r2-c4-n0` | Dropper 8, board 0 finite solutions | +13.693 / +2.605 mm | 579.343 / −59.661 | 478.303 mm / 25.631° / 182.232 mm | 7 / 7 | 3/3 |

Bucket X/Y stays at **(378.083, −438.148) mm** for the first context and
**(898.306, −244.167) mm** for the second. X/Y changes also alter target-facing
yaw and the derived release pose; this is not an isolated translation with all
physical orientation/joints fixed. Full precision belongs to the committed
summary, not the rounded table.

An original rejected recovery has no actual settled pose. The new admitted
pose comparisons therefore explicitly use each **matched successful parent**,
not zero-joint placeholders from a refusal. For example, the first X/Y variant
settled 24.328 mm below its parent board center with a 0.57349° normal change;
both arms' solved joint vectors changed. The UI and summary expose these
realizations separately from requested parameter changes.

## Evidence and reproduction

| Item | SHA-256 / identity |
| --- | --- |
| New contract | `f80dc7fa8f0b535bf5d03040076e9bc0199f4781b2be3dd3ce7c8333f913c8d2` |
| Completed report | `8199b13d944f507e78664a8c78734e92ab56e3db14025d7a93eefd4730a6ecd2` |
| Independent audit | `b5b65c8292147458cc79650638c42d17c63a71b302b228bff57b998234068d60` |
| Original diagnosis | `511e9f82004af0d603628e25adf8fab71284faa8f11de9307b4c8a611d049ab7` |
| Original policy report | `6a255223820656059fe16a5de2944ba2593f7ac3603b287f4955979cd4677504` |

The private runtime evidence is under the existing Ubuntu study namespace
`output/regression-ik-v1`, with its adjacent audit. Local inspection artifacts
are `.runtime/ik/{report,audit,summary,snapshot,publication-receipt}.json`.
Raw native traces, proofs, queues, logs and runtime configuration are excluded
from Git. The compact baseline summary is copied byte-for-byte from the
audit-bound publisher result. Reproduction uses the modules and authenticated
inputs in the protocol; it does not overwrite or silently resume old runs.

Every new action had one strict proof reused across its three nominal requests;
the six original refusal slots reused cached evidence. Three refused requests
are not three independent IK searches. Admitted repeats are fresh native
releases. Admission tolerances, seeds, endpoint/clearance checks, geometry
namespaces, physics, scoring and original branch records were preserved.

## Interface and isolation

The additive `#regressionIK` panel contains all 16 action settings, branch counts,
refusal reasons, matched outcomes, controls/audit and actual saved first-repeat
ball/posture traces. Refusals show no invented motion. Its **99,300-byte**
snapshot preserved all **57 existing report keys and report bytes**, historical
HTML/CSS outside its own addition, and the previous movie. Visual checks verified
both X/Y results, the complete table, disabled flight playback for a refusal,
and actual sample scrubbing for the second X/Y correction.

No service restart or selector activation was needed. Only the Mac SSH tunnel
was restored to reach the existing viewer. Final read-only isolation checks
confirmed the 4090 worker exited, the viewer and 4070 workload kept their PIDs
and process start times, and historical reports retained their hashes. Isaac
was untouched. Source CI passed; publisher and UI unit
checks plus independent actual-result audit and browser checks passed.

## What this changes next

All five inspected lost captures now have local successful corrections when
combined with the earlier height and drop-gap studies. Preserve these recipes
as development evidence. The original policy score remains **37/70**; these
inspected positions cannot be advertised as a new held-out evaluation. No new
training, model weights, interactive policy or automatic fallback was installed.

A separate follow-up can incorporate validated local teachers or an explicit
feasibility fallback, then freeze that procedure and evaluate it on genuinely
new bucket positions. The current result alone does not establish noise
robustness, timestep robustness, all-table coverage, executed arm transit or
physical-robot readiness. No further experiment is automatically launched.
