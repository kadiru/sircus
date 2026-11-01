# Dense fixed-plane catalogue — synthetic Isaac/PhysX results

Run: `shot-catalogue-20261101T022504Z`. Fixed400 mm bucket-center distance. Complete search and confirmation; independent100-trial follow-up also completed.

495 coarse settings +84 local refinements =579 searched settings:39 clean captures,89 assisted/marginal captures,354 released misses and97 rejected endpoints. There were819 attempts including240 confirmation trials;722 released. None of the released trials failed the measured settling gate.

| Board center H mm | Gap mm | Tilt ° | Ball bottom Z mm | Ball center Z mm | Flange Z mm | Nominal clean / assisted / miss | Perturbed clean / assisted / miss |
|---|---|---|---|---|---|---|---|
| 325 | 230 | 12 | 555.0 | 592.5 | 687.5 | 20 / 0 / 0 | 1 / 0 / 19 |
| 450 | 120 | 21 | 570.0 | 607.5 | 702.5 | 20 / 0 / 0 | 18 / 0 / 2 |
| 400 | 130 | 21 | 530.0 | 567.5 | 662.5 | 20 / 0 / 0 | 20 / 0 / 0 |
| 350 | 190 | 15 | 540.0 | 577.5 | 672.5 | 20 / 0 / 0 | 5 / 0 / 15 |
| 475 | 120 | 27 | 595.0 | 632.5 | 727.5 | 20 / 0 / 0 | 0 / 11 / 9 |
| 375 | 130 | 24 | 505.0 | 542.5 | 637.5 | 20 / 0 / 0 | 10 / 3 / 7 |

Each column group contains20 trials. All240 confirmation trials were valid endpoints. Perturbations independently vary H ±2 mm, gap ±2 mm and tilt ±0.5°. The six settings were frozen before confirmation. All120 nominal trials were clean captures, but only68/120 perturbed trials captured (54 clean,14 assisted;52 missed). Identical repeats greatly overstate robustness here.

**Initial strongest candidate:** H400 mm, gap130 mm, tilt21°. It passed20/20 nominal and20/20 perturbed trials, all clean. It was selected after inspecting this confirmation set, so the separate100-trial follow-up uses fresh seeds and no retuning. The H325/gap230/12° setting is particularly fragile (1/20 perturbed captures) despite20/20 identical successes; do not recommend it as robust.

Quality2 means clean capture with ≥1 mm sampled mouth clearance and no detected pre-entry/rim-band contact or later partial exit; quality1 means contact-assisted/marginal capture; quality0 means miss. Both capture tiers retain the historical board-bounce, whole-ball entry and continuous1second retention requirement. Rejections are separate. No old results or rewards were relabeled.

All heights are above tabletop Z=0. Board/release XY=(670,−100) mm; bucket XY=(670,−500) mm. The ball radius37.5 mm, flange offset95 mm, rigid board mount and materials are provisional. Bucket175 mm reported height interpretation is unresolved; model uses175 mm interior depth+8 mm base. Endpoint initialization is not grasp/path learning or collision-safe execution.

## Evidence

- `results.json`: compact search/confirmation rows, immutable contract/source hashes, validated aggregate checks, source-summary SHA256 and original run-artifact hashes.
- `searched-settings.csv`: all579 settings, exact release-height meanings, score and rejection information.
- Raw819 trials, contact probes, frozen selection and original snapshots stay on Ubuntu outside Git. All five run-artifact hashes verified against the completed files.
- Released shots: maximum settled position error0.11490 mm; normal error0.01584°; joint speed0.00007481 rad/s. Configured gates were unchanged.
- Contact probes: floor190.98 N, rim119.75 N, outside0 N; both positive and negative bucket-filter checks passed.
- Executed snapshot `studies/catalogue-v2` is immutable. Final source includes a formatting-only change to the run writer and adds the report/UI/fresh-confirmation runner; executed content hashes identify exactly what ran.

Clean refers to bucket entry; the scorer permits multiple board contacts before entry. No learned policy was trained. The grid capture fraction is not policy performance. Zero failures in a finite simulation test is not a guarantee, nor physical validation. Next useful work is a broader perturbation/contact audit around the strongest setting before expanding bucket distances.

[Method and reproduction](../../sircus/isaac/SHOT_CATALOGUE.md).

## Fresh independent confirmation

The frozen strongest setting (H400 mm, gap130 mm, tilt21°, distance400 mm) scored
**100/100 clean captures** on100 fresh seeded perturbations, with0 misses and0
rejected poses. No retuning occurred. Including its initial20 nominal and20
perturbed trials, this setting produced140/140 clean captures across the tested
conditions. This remains finite, narrow simulation evidence, not a guarantee.

`independent-confirmation.json` preserves all100 compact rows, source-summary
hash, frozen-selection rule, source/artifact hashes and verification metrics.
Both raw run-artifact hashes were verified on Ubuntu. All100 parameters matched
the disjoint seed schedule; settling and one-second capture checks passed.
Minimum sampled mouth clearance: 34.18 mm.

The whole task evaluated919 parameter attempts (819 initial +100 fresh), with
822 releases and97 endpoint rejections. Both finite jobs exited; the read-only
preview remains available. Physical hardware, Viam and audio were untouched.

Validation:67 Isaac unit tests,21 JS tests across existing study/diagnostic/UI
suites, full project mock/numerical regression, Ruff and repository file checks.
