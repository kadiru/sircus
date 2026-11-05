# Geometry-guided continuous PPO — November 5, 2026

The geometry map reduced endpoint reach/IK failures, but this particular local
proposal strategy reduced capture performance. The new model is a diagnostic
checkpoint, not a promoted table-wide solution. The original broad pilot and
historical benchmarks remain separate.

## Matched native results

Both methods used 1,600 attempted training shots and 16 PPO updates, with the
same 25 training/five validation/five test coordinates and eight nominal frozen
repeats per site. Geometry support changes the physical action domain and
starting prior; this is not an isolated comparison of PPO itself.

| Training metric | Broad pilot | Guided pilot |
|---|---:|---:|
| Attempted shots | 1,600 | 1,600 |
| Released shots | 386 | 721 |
| Rejected requests | 1,214 (75.9%) | 879 (54.9%) |
| Endpoint reach/IK rejection, before native correction | 1,106 | 54 |
| Native endpoint not settled | 49 | 606 |
| Settling-correction IK rejection | 33 | 73 |
| No geometry support | 0 | 128 |
| Board/table/bucket geometry proxy rejection | 26 | 18 |
| Qualified captures | 62 | 23 |
| Clean captures | 25 | 0 |
| Single-board-bounce captures | 62 | 22 |

The after-training policies were frozen before evaluation:

| Subset | Broad final | Guided initial | Guided final |
|---|---:|---:|---:|
| Training sites | 32/200 (16%) | 7/200 (3.5%) | 8/200 (4%) |
| Validation sites | 8/40 (20%) | 0/40 | 0/40 |
| Test sites | 8/40 (20%) | 8/40 (20%) | 8/40 (20%) |
| All 35 sites | 48/280 (17.1%) | 15/280 (5.4%) | 16/280 (5.7%) |

Only `r2c3` (train, 7→8/8) and `r3c3` (test, 8→8/8) captured under the new mean
actions. The one extra nominal repeat is not evidence of a meaningful spatial
learning gain. The old test winner was `r4c4`; an unchanged test percentage does
not mean the same layout succeeded. Frozen clean captures fell from eight in
the broad final policy to zero. No training/evaluation capture had requested
bounce radius ≥600 mm; 369 admitted training releases in that range missed.

These coordinates are inherited from earlier studies. They are held out from
this optimizer's reward updates, not globally unseen project layouts. The
geometry map also assists the decoder at held-out coordinates. Eight identical
nominal repeats measure numerical repeatability, not independent scene/noise
robustness. This is one training seed; no population generalization claim.

## What geometry guidance did

The prior geometry-only map contained 155 IK/FK/proxy-qualified candidates at
33/35 sites. A deterministic geometry/joint-travel rule chose one anchor per
supported site. The policy then sampled continuous residuals inside a local
five-dimensional box: bearing, radius, board height, drop gap and tilt. Every
requested shot still received a fresh solve, independent FK/path proxies and
the unchanged measured release gates. Failed requests remained in the
optimizer and denominator; no projection or successful-shot lookup was used.

Preparation retained all 1,040 local geometry probes; 693 passed overall.
Selected boxes passed 452/528 probes (85.6%), with five explicitly marked
below-target fallbacks. Two contexts remain unsupported, not proven impossible.
Additional preparation cost was 21.58 s wall / 116.31 s user CPU with eight
workers. Maximum RSS 223,128 KiB is not an aggregate worker-memory peak and
excludes the original map cost. Finite probes do not certify every box point,
mesh collision, self-collision or native settling.

The fresh 2×64 actor/critic used the original seed, PPO settings and normalized
Gaussian standard deviations. The new decoder nevertheless changes physical
exploration scale and its geometry-derived initial shot. See the
[protocol](../../sircus/isaac/FEASIBILITY_GUIDED_PILOT.md).

## Why the comparison matters

A separately verified **posthoc** diagnostic found that all six original final
successful parameter settings lay outside the new boxes: none was retained.
All six were outside the radius and height bounds; five each were outside the
bearing, gap and tilt bounds. This evaluates the support change after it was
frozen. It was not fed into anchor selection, rewards or training.

This does not prove the new boxes cannot capture: both guided scoring sites
have successful alternative settings. It does show that one geometry-selected
local box can discard demonstrated solutions. Reduced reach/IK rejection alone
is not an aiming improvement. The next bounded change should preserve more
geometry-qualified regions and diagnose settling before another longer run.

All 721 admitted training shots rebounded from the board. Of 698 admitted
misses, first external contact was table 658 or robot 40. In 129 cases bucket
contact occurred first, so these cannot all be called undershoots. All observed
robot contacts followed board contact. Twenty-seven retained balls were
excluded for intervening table contact; final evaluation had 15 such route
failures (eight table, seven robot). Raw retention never substitutes for the
unchanged board-bounce/whole-ball-entry/one-continuous-second capture rule.

## Recording and evidence

The native run took 57.26 minutes including coverage, recording and evaluation.
Twenty prescheduled batches produced 17,989 actual 1080p/30fps frames. The
exact saved camera, silver-gray stage and 35 distinct bucket coordinates are
preserved. No ball-following camera or generated replacement imagery was used.
The full movie is 9:59.64; its action edit is 9:42.11. The edit removes 526 setup-dwell frames and retains all 2,400 release frames without speeding them up. The foreground has 16 training and two coverage releases, with zero qualified captures; the film is not a success montage. The [film receipt](film.json) records digests and cuts.

Large visible arm motion is scripted goal travel after canonical episode resets
before filming. Grasping, ball loading and joint paths are not learned; there
are no scored-endpoint joint-state resets. Across the 33 supported contexts,
initial→final mean changes were at most 0.46° bearing, 3.26 mm radius, 1.35 mm
board height, 4.03 mm gap and 0.49° tilt. Large setup/IK branch movement must not
be described as learned bucket orbits. Unsuccessful/withheld trials stay visible.

Independent audits checked fresh initialization, 66 checkpoints, all 1,600
optimizer samples, 1,195 native releases and 2,822,400 observations. Maximum
CPU/CUDA likelihood discrepancy was 1.91e−6. All 35 evaluation homes matched
within 0.000199 rad; maximum board displacement through scoring was 0.865 mm.
Actual admitted board/release pose errors and every camera release readback
were checked; this does not establish physical calibration or complete
collision-safe robot paths.

## Portable prediction bundle

The compact [decoder.json](decoder.json) has its **own** trusted fingerprint:
`a28e915cbbbd90d4a32bc3ec5ead07a303faad879d1a835e99d433306d1aefeb`.
Its original full-support fingerprint and SHA remain separate. The unchanged
guided checkpoint loader still checks the original identity. Do not substitute
a stripped summary under the original fingerprint or rewrite model hashes.

The CPU export proved all 2,800 policy contexts equivalent (2,160 active,
640 inactive); four geometry-coverage batches remain excluded. Raw and portable
initial/final mean actions are bit-identical on CPU for all 35 contexts. Ordinary
portable inference rejects unknown or unsupported coordinates. It performs no
motion commands or per-context capture search.

From the repository root, in the documented CPU Torch environment:

```python
import json
from pathlib import Path
import torch
from continuous_policy import build_observations, sample
from feasibility_guided_decoder import load_policy, decode_actions

base = Path('baselines/isaac-feasibility-guided-pilot-20261105')
manifest = json.loads((base / 'manifest.json').read_text())
expected = manifest['decoder_fingerprint']  # trusted committed manifest
policy, saved, decoder = load_policy(base / 'policy-final.pt', base / 'decoder.json',
    expected_decoder_fingerprint=expected, device='cpu')
contexts = [next(c for c in decoder['cells'] if c['id'] == 'r2c3')]
inputs = torch.as_tensor(build_observations(contexts), dtype=torch.float32)
prediction = sample(policy, inputs, deterministic=True)
values = decode_actions(prediction['actions_normalized'], decoder, contexts,
    expected_fingerprint=expected)
print(values)  # bearing rad, radius m, board H m, gap m, tilt rad
```

Set `PYTHONPATH=sircus/isaac` and `CUDA_VISIBLE_DEVICES=''`. This bundle supports
these known simulation contexts; new bucket coordinates require new geometry
support, qualification and a compatible study contract.

- [comparison.json](comparison.json), [summary.json](summary.json): independently matched totals and scope.
- [support-definitions.json](support-definitions.json), [support-exclusion.json](support-exclusion.json): compact geometry definitions and posthoc classification.
- [checkpoint-audit.json](checkpoint-audit.json), [native-audit.json](native-audit.json): policy/native receipts and raw-file digests.
- [decoder-export-audit.json](decoder-export-audit.json), [portable-roundtrip.json](portable-roundtrip.json): faithful extraction and both model roundtrips.
- [failure-summary.json](failure-summary.json): aggregate surface, pose-error, distance and per-site results.
- [requested-settings.csv](requested-settings.csv): all 35 requested final configurations and nominal results; direct pose/readback columns remain in the private full CSV.
- [manifest.json](manifest.json): all compact file digests; full telemetry/RGB, raw probe histories, runtime configurations, logs and media stay outside Git.

Producer `2dabb21`, auditors `6fe65ff`, portable exporter `ae6a1db`, report/video
exporter and posthoc diagnostic `3f14801` are pinned separately. Mac full
regression passed (434 Isaac tests with 57 dependency skips, 34 physical mocks,
five simulation tests, 37 JS checks and historical two-arm checks). All 141
continuous/guided tests passed on actual Ubuntu CPU Torch without skips.

Board 200×300×5 mm landscape geometry is preserved. Bucket inside diameters
177/150 mm and outside 200/158 mm are unchanged. Reported 175 mm height still
has unresolved overall-height versus interior-depth meaning; the existing
interior 175 mm plus 8 mm bottom remains provisional. Material/contact settings
are fixed simulation assumptions. No physical, Viam, audio or prior-service
configuration was changed; the 4070 workload was preserved.
