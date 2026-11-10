# Fixed-grid recipe recovery — November 9, 2026

**Five of 11 failures recovered; 29 of 35 fixed positions now have confirmed
recipes.** Each new recipe captured all three fresh nominal repeats. All 24
unchanged parent recipes captured again. The existing frozen policy remains
**24/35**; this study performed search, not policy training or promotion.

The portable [summary](summary.json) preserves exact actions, bucket coordinates,
record identities, confirmation groups, parent recipes and audit hashes.
[Full interpretation and settings](../../docs/MJLAB_FIXED35_RECOVERY_20261109.md)
and [frozen protocol](../../sircus/mjlab/FIXED35_RECOVERY.md) explain the boundaries.
The interface section is `#fixed35Recovery`.

| Bucket X/Y (mm) | Board centre Z (mm) | Tilt (degrees) | Drop gap (mm) | Recorded release centre Z (mm) | Fresh captures |
|---|---:|---:|---:|---:|---:|
| 100, −650 | 306.984 | 26.087883 | 398.769 | 743.253 | 3/3 |
| 1240, −425 | 459.395 | 26.036104 | 252.213 | 749.108 | 3/3 |
| 670, −200 | 509.582 | 10.096249 | 195.046 | 742.128 | 3/3 |
| 1240, −200 | 472.215 | 27.237403 | 162.596 | 672.311 | 3/3 |
| 860, 250 | 460.004 | 25.993019 | 252.826 | 750.331 | 3/3 |

Displayed values are rounded. Use the full recipe, including board X/Y and
stored joint branch, from the summary; height/tilt/gap alone is not a complete
command. Gap is not absolute release height. The 1240/−425 recipe is the existing
refit alternative newly confirmed here; it is not claimed as a novel discovery.
The two recipes with bucket X=1240 used table assistance.

## Remaining locations

Four positions exhausted all 80 frozen candidates without capture:
(1050,−425), (860,25), (1050,25), (1240,250) mm. The two Y=25 misses had early
robot contact in every admitted flight. The far-right Y=250 cell admitted only
four flights out of 80 requests, predominantly lacking a valid board-arm branch.
The 1050/−425 cell had no whole-ball entries; its robot contacts occurred after
bucket/table contact, so do not establish the initial cause.

Two positions, (100,25) and (1240,25) mm, remain blocked by the unchanged
conservative fixed-base clearance test and were not searched. This is not a
proof of actual mesh overlap. Finite IK/search failures do not prove a location
impossible.

## Accounting and limits

- 441 requests: 384 search, 15 confirmations, 24 parent replays, 18 controls.
- 203 native releases; 238 setup declines; 56 captures including repeats and
  controls; 147 released noncaptures. This aggregate is not a policy score.
- Three noncaptures were invalid physics at bucket (670,−200): roughly 8-degree
  tilt and 235.046 mm gap caused `through_wall_after_entry`. They retained for
  less than 0.1 second and never counted as capture. The confirmed replacement
  uses 10.096249 degrees and 195.046 mm gap.
- All pre/post control gates and 141,041 independent checks passed. Worker
  4,105.75 s; pipeline 4,135.29 s. Search completed its finite caps, not a timeout.
- Native64 and v3 scoring: board bounce, whole-ball entry and one continuous
  second retained; table/rim assistance allowed, robot/floor/unattributed contact
  excluded. No bucket perturbations, noise or 128-substep qualification.
- Held admitted arm endpoints only, not transit/grasp/physical validation.
  No training or generalization claim. Raw traces/logs remain outside Git.
- Final isolation passed 31/31 checks. Owned 4090 work exited; 4070 and Isaac
  remained unchanged. UI publication preserved historical evidence and services.

## Provenance

- Producer: `7bcba3980a1baf24d6ac2505e35189b95daead64`.
- Contract: `a7b58fc50ce6d7b28b0d2a2ab045f4031bc77d3c70145c01a670ec134a003bf5`.
- Report SHA-256: `2d62f6b43b54210b83c7323b5e46f0ccdf0817f36b67fbe43e1aa7618fb099d9`.
- Audit SHA-256: `460e0e5496b54e9426fb170d64b9c0687813eb28e3b7310d3e8f212eda211569`.
- [Draft PR #120](https://github.com/kadiru/sircus/pull/120), based on #119.
