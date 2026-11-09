# Recovery capture losses and matched height correction

This follow-up inspects the five captures lost by recovery imitation relative
to parent imitation. The original comparison remains **37/70 versus 34/70**,
with eight gains and five losses. All inspected positions are now development
data; this follow-up is not another held-out policy evaluation.

The [compact diagnosis](../baselines/mjlab-recovery-regression-20261109/README.md)
and [frozen protocol](../sircus/mjlab/RECOVERY_REGRESSION.md) record exact settings
and evidence identities. The independent CPU diagnosis passed 3,103 checks,
including rescoring saved native traces. Historical evidence remains unchanged.

## Completed correction result

All 63 slots finished: 48 native releases and 15 refused requests. The native
worker took 221.1 seconds; the pipeline including audit took 227.6 seconds.
The independent audit passed **22,266 checks**, with no physics-invalid attempt.
All 18 controls behaved as required. These are elapsed wall times, not isolated
GPU compute times.

| Variant on five selected losses | Captures / requests | Releases | Declines |
| --- | ---: | ---: | ---: |
| Exact parent recipes | 15/15 | 15 | 0 |
| Exact recovery recipes | 0/15 | 6 | 9 |
| Restore parent board height | 3/15 | 9 | 6 |

Only `recovery-test-r3-c1-n0` recovered, with three clean direct captures.
Raising its board-height parameter **241.129 → 280.424 mm** resolved the former
table/link5 setup decline. Board X/Y, tilt and gap remained the recovery values:
**613.087/−14.842 mm, 26.792°, 173.760 mm**. Bucket X/Y was
**339.213/−30.949 mm**; requested ball-release Z became **491.685 mm**.

The two ordinary flight regressions still missed all three repeats, without
robot contact. The other two setup cases still failed finite IK admission:
`r1-c1-n1` had zero dropper/eight board branches; `r2-c4-n0` had six dropper/zero
board branches. Neither reached endpoint-pair or path testing; no clearance
proof exists there (unknown, not zero clearance).
All diagnostic flights had zero robot/forbidden contact; the six intentional
negative controls had robot contact and remained misses as designed.

Nine recovery refusals reuse their original proofs; six corrected refusals
reuse two new strict proofs. They are not repeated independent solver searches.
The five complete parent recipes retained 15/15 captures, so there is no reason
to discard them. This posthoc fix does not revise 37/70 to 38/70 or establish
a general height-restoration policy. Next: a separately frozen drop-gap test
for the two contact-free flight losses, and full parent recipes as controls for
the two unresolved IK cases, before more learning.

Final report SHA-256:
`a6f2d560e9bbd0e721f4b3aaaab651a11de967a6bdb221906208876865582931`.
Independent audit SHA-256:
`81d45d57ea47c0ad558102a0cd8377fbd08caff0ce1952ae8b0ae1d57641f50a`.
Contract fingerprint:
`74d500fea9c941f4101ad47176b8daaa3cabaabba12e45bd6514e34fd42e1778`.

## What the saved failures show

- All 17 released recovery misses failed whole-ball entry. Seven had forbidden
  contact. The first post-board contact was bucket in six, table in six and
  robot in five. A table bounce remains allowed; a contact is an observation,
  not by itself an explanation of the miss.
- Of 16 setup declines, five lacked a dropper IK branch, five a board branch
  and one both. Two failed strict endpoint checks, two fixed-base clearance and
  one action bounds. These finite searches do not prove unreachability.
- Three of the five lost captures became setup declines. Their board heights
  shifted by approximately 18–39 mm. The two flight losses had no forbidden
  contact but struck the bucket 24–34 mm lower than their parent captures.
  Their larger drop gaps are a separate hypothesis, not a tested correction.

## The single intervention

Restore only recovery action coordinate 2, board height, to the corresponding
parent value for all five losses. Keep bucket, board X/Y, tilt and drop gap
unchanged. **Absolute release height shifts by the same amount as board height**
because the gap is fixed. This does not isolate a physical board-only movement.

The 63-slot plan compares parent exact, recovery exact and corrected settings,
three times each at five positions, with 18 original controls before/after.
Original refusal requests reuse their authenticated decisions; they are not
new solver attempts or native flights. Corrected actions receive fresh strict
admission proofs. No rejected request is replaced and no guard is weakened.

Diagnostics use the same 19-source prior; formal controls retain their exact
17-source branches in a separate namespace. Fixed64 physics, version3 scoring,
geometry and timing remain unchanged. A capture requires a board bounce,
whole-ball entry and one continuous second inside; table/rim assistance counts,
while robot/floor/unattributed contact and invalid physics do not.

## Reproducibility and boundaries

- Frozen native producer: `a95e3033730e6b76d5452178c176101e1aa2b09d`.
- Separate video-publication revision: `699e411bb7e44ae3772a37e3fe840ac9a869b61d`.
- Draft PR: [#109](https://github.com/kadiru/sircus/pull/109), stacked on #106.
- Ubuntu base: `/home/kadir/sircus-mjlab`; frozen source
  `studies/recovery-regression-v1/`; evidence `output/recovery-regression-v1/`.
- Adjacent pipeline state/log, worker budget and independent audit use the same
  `recovery-regression-v1` name. The launch receipt is under `launch-receipts/`.
- Actual-parent CPU preflight reconstructed all 63 requests and hydrated both
  proof namespaces, passing 156 checks without creating a native environment.
- Maximum acquired 4090 time is 3,600 seconds, queue 900 seconds, original total
  pipeline window 6,300 seconds, with 1,800 seconds reserved for CPU audit and
  an 8 GiB free-storage floor. No outcome-dependent budget extension or retry.

Validation: the complete local `scripts/test.sh` passed (optional backend tests
without local dependencies remain skipped), plus focused publisher/video tests.
Ubuntu verified pinned dependencies and actual parent artifacts before launch;
the final native audit checked all released trajectories and both admission
namespaces. GitHub's mock/simulation workflow passed the video-publication code
revision. Repository/staged file checks exclude runtime artifacts and credentials.

The existing 18781 viewer remains read-only. Its additive `#recoveryRegression`
panel uses a bounded JSON snapshot in a uniquely marked CSS comment because
the existing static allowlist has no separate data endpoint. Existing HTML,
CSS and display-report keys are preserved; no service restart or solver switch
is needed. A separately verified movie uses actual saved ball centers and
held arm joints, rendered on CPU with no new physics or invented transit.

The compact terminal snapshot is 228,766 bytes and includes first-repeat records
for all 15 case/variant pairs; unflown cases have no fabricated trajectory.
The bounded publisher exited after two updates with zero errors. All 57 existing
display-report keys and the report bytes were preserved. Chrome verified the
passed audit, all outcomes, saved-sample playback and disabled playback for a
setup decline. The viewer retained its original PID/start time; the 4090 native
worker and supervisor exited, and the 4070 fountain process retained its identity.

The final movie is **60.13 seconds / 1,804 decoded frames**, with both paired
historical flight regressions and one representative forbidden-contact miss.
It uses normal speed and 0.25× close-ups, actual recorded ball samples and held
joint poses; it does not depict the new height correction or learned arm transit.
Corrected first-repeat trajectories are separately available in the sample
selector. [Video provenance](../baselines/mjlab-recovery-regression-20261109/video-summary.json)
binds clips, cameras, scene, frame plan, MP4 and static publication bytes.

The successful renderer was CPU Mesa llvmpipe with both CUDA devices hidden
and the Mesa EGL vendor explicitly selected. Two failed backend probes remain
recorded: missing OSMesa, then default EGL selecting NVIDIA and being rejected
by the mandatory software guard before frames. No native simulation was rerun.
All 1,804 final frames decoded; Chrome loaded the 60.13-second movie, advanced
playback and paused without a media error. Final HTML outside this panel and
CSS outside its snapshot marker are byte-identical to their originals.

Native media and receipts are under
`verification/recovery-regression-v1-video-v3/` and the separate presentation
source/receipts under `verification/recovery-regression-v1-ui/`. On the Mac,
ignored copies are in the main checkout's
`.runtime/artifacts/recovery-regression-20261109/`. Raw media is not in Git.

No new PPO, policy promotion, grasp/transit/perception learning, noise study,
128-substep claim or hardware validation. The 4070 workload and Isaac setup
are outside this task. Raw observations, full proofs, generated configs, logs
and media remain outside Git; compact evidence and code are shared.
