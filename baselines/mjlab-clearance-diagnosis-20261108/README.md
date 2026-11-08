# Finger-clearance diagnosis and one targeted correction

The tighter conservative distance bound recovered two useful shot selections
without changing the old flight horizon or 2 mm required clearance. This is a
targeted development replay, not a new training run or a generalization test.

| Method | Captures / requested | Clean | Clearance abstentions | Native misses | Robot-contact trials / releases |
|---|---:|---:|---:|---:|---:|
| Original clearance gate | 0/18 | 0 | 12 | 6 | 6/6 |
| Refined distance bound | 6/18 | 3 | 9 | 3 | 3/9 |

Six previously observed layouts × three identical repeats per method. There
were no ordinary setup rejections or invalid-physics observations in this
selected experiment. Abstentions remain in the denominator and have unknown
contact outcomes. Twelve controls passed separately: six expected clean
captures and six expected misses. Total: **48 requested slots, 27 native
releases, 21 abstentions**. The native audit passed **65,467 checks**; the CPU
diagnosis passed **180 checks**. No active selector was changed.

## Concrete findings

- **07:** the old face-plane bound was −3.632 mm; the refined conservative
  bound is +4.541 mm, above the unchanged +2 mm requirement. The corrected gate
  admitted the same previously withheld shot and obtained **3/3 clean captures**,
  with no observed post-rebound robot contacts.
- **44:** refining the bounds admitted the earlier ranked candidate 2 instead
  of candidate 9. The old selection missed **3/3** with robot contacts; the
  new selection captured **3/3**, with no such contacts. These three captures
  were assisted/marginal, not clean.
- **01:** both methods replayed the same shot and missed **3/3** with robot
  contacts. The original recorded contact occurred at 334.896 ms after rebound,
  beyond every predicted horizon (321.488–326.923 ms). The time horizon was
  intentionally unchanged in this single-factor experiment.
- **00, 04, 39:** both gates withheld all three requests. They were not native
  misses. In particular, the refined predictions for 04 and 39 remain below
  the margin; the distance correction does not resolve their model errors or
  close geometry.

The earlier active consistent-flight selector had captured cases 04 and 39.
This test compares two versions of the **clearance gate**, not a new paired
comparison against that active selector. Do not interpret 6/18 as an overall
policy success rate, or the targeted gain as justification for promotion.

### Successful corrected shot settings

All lengths below are millimetres in the existing simulator table frame.
Tilt is measured from horizontal; release gap retains its existing contract
definition. These coordinates are synthetic and are not hardware calibration.

| Case | Bucket X / Y | Board X / Y / height | Tilt | Drop gap | Result |
|---|---|---|---:|---:|---|
| 07 | 439.606 / 151.701 | 592.385 / 101.498 / 269.285 | 27.267° | 99.131 | 3/3 clean |
| 44 | 430.810 / −102.857 | 632.944 / −274.450 / 274.148 | 27.534° | 96.365 | 3/3 assisted/marginal |

The complete 36 comparison requests, exact values and unknown observations
are retained in [the sanitized scalar summary](results.json).

## Evidence and reproducibility

- [Protocol, one-factor rule and commands](../../sircus/mjlab/CLEARANCE_DIAGNOSIS.md).
- Frozen producer commit: `885483630af588a7cec7cfbc2cd3b092e42d6763`.
- Contract bytes: `79623f0903665ee303d63391ddf50232f4c6927408ab255173a6bcaf0473cf0e`.
- Requested plan: `4a5746f16209e4fe4b09056c4f4f03f5025e801418f356914bfa3948ffee82dc`.
- Selection ledger: `83c030b78ca57a289723e4724100535952c4835df5d80d923dc5c3fe65612104`.
- Native report: `80e9bdc4a656d329a54da5a9fb3e4c14be350be959528c0d8b03580b1d073843`.
- Evidence manifest: `354afcd2f6f2e8442cf7ff1d06e246b12d49e0e0be256bbfb558ac23083b3e19`.
- Full native audit: `7f21dcd9af89d19ee4c7b82caf1a2680009ca42b5b37992d426dc3540e846e35`.
- An additional CPU source-provenance audit matched all 1,093 tracked snapshot
  files to the committed archive, all 34 imported native-pinned dependencies,
  and all 155 immutable native-source files. Zero mismatches. Its digest is
  stored in the scalar summary; it did not alter source, evidence or outcomes.

The existing interface has an additive read-only `#clearanceDiagnosis` panel
with separate proof gates for diagnosis and replay. It retains earlier panels
and requires no service restart. Raw traces, host paths, logs, calibration,
source archives and the full proof ledgers stay outside Git.

## Remaining work

Next, isolate the horizon failure in case 01. A CPU-only extension toward the
table-height boundary rejected its prediction, but this is not yet a native
selector test. Other late robot contacts followed table or bucket interactions;
extending a ballistic curve does not model those deflections. Resolve these
development issues before freezing a fresh held-out selector comparison or
starting more training. Preserve the old failures and report abstention cost.

Only the 4090 ran the finite replay; it returned idle on completion. The 4070
workload and Isaac setup were not changed. Fixed 64 substeps, privileged
geometry and held endpoints do not establish driven-motion, noise, 128-substep,
perception, physical-transfer or all-table reliability.
