# Fixed 35-position imitation: completed comparison

Current imitation captured **24/35 (68.6%)**; refitted imitation captured
**23/35 (65.7%)**. The candidate gained one position, lost two, shared 22 captures
and shared ten failures with the current policy. It did not improve this sample;
keep the current checkpoint and all successful recipes. No policy was promoted.

The [compact audited summary](../baselines/mjlab-fixed35-imitation-20261109/summary.json)
and [protocol](../sircus/mjlab/FIXED35_IMITATION.md) preserve the exact coordinates,
commands, outcome counts and checkpoint identities. The live `#fixed35Imitation`
panel shows one cell per bucket location and each model's corresponding result.

## What was evaluated

There are **35 locations**, arranged as seven X values × five Y values:
X = 100, 290, 480, 670, 860, 1050, 1240 mm;
Y = −650, −425, −200, 25, 250 mm.
Each model received one request at each location: **70 model attempts**.
Another 18 pre/post controls brought the total to **88 requested slots**.
Control captures are not part of either 35-position policy score.

Both policies used the same fixed bucket-facing gripper rotation, full-pose IK
and clearance checks. The simulation stayed at 64 substeps with unchanged
version 3 scoring: board bounce, whole-ball entry, continuous one-second
retention; table and rim assistance allowed. Robot, floor and unattributed
contacts still disallow capture. This is not an orientation A/B experiment.

Twenty-three evaluation positions exactly overlap teacher coordinates. All 35
belong to the historical development grid. The result measures fixed-grid
coverage/regression, not unseen-position generalization. No placement jitter,
noise or 128-substep run was added.

| Outcome | Current imitation | Refitted imitation |
| --- | ---: | ---: |
| Captures / requests | 24 / 35 | 23 / 35 |
| Admitted native releases | 32 | 29 |
| Setup declines | 3 | 6 |
| Released noncaptures | 8 | 6 |
| Invalid physics, included above | 0 | 1 |
| Direct captures | 22 | 20 |
| Table-assisted captures | 2 | 3 |

The invalid attempt occurred at `(1050, −200) mm`,
`fixed35-dev-grid-02-05`; it remains an unsuccessful request. The independent
audit confirms faithful accounting and source/native evidence, not that every
rollout was numerically valid. There are five ordinary released misses plus
this invalid attempt in the refit's six released noncaptures.

The gained position is `(1240, −425) mm`. The lost positions are
`(1050, −200) mm` (invalid physics) and `(1050, 250) mm` (a forbidden-contact
noncapture). These exact paired actions remain visible in the summary and UI.

## Fitting and reproducibility

The candidate retained 97 historical teacher contexts and added two confirmed
bucket-facing corrections. Each of the 99 layouts contributed one equal-weight
example; repeated drops and ten successful alternatives at the farther layout
were not duplicated as extra training layouts. Historical teacher records keep
their original orientation provenance and are not relabeled as verified
bucket-facing captures.

Training ran 800 full-batch actor-only Adam steps at learning rate 0.001. The
critic and exploration parameters stayed unchanged. The model outputs five
continuous shot adjustments around the existing geometry anchors. Both weights
and all 70 predicted commands froze before evaluation; there was no evaluation
flight search, retry, fitting or outcome-based checkpoint selection. This was
imitation learning, not another PPO run.

| Artifact | SHA-256 |
| --- | --- |
| Current checkpoint | `688acc7981522c3281dff47edd19edaf46f084f113455e50f14d7e5cabc3b9e9` |
| Refitted checkpoint | `d17765096854446bc666870478358fd2e6337936cbe3fceb3d9a6cb71e95ad8b` |
| Native report | `91640ad07e7b8e396c58d937dcf8a5652a07c3eb5f7ae2ea0bce27f40425a389` |
| Independent audit | `374eb3ff78b302168622c091aca645283f48d1bee33355a9ac4eec3794e116d6` |
| Contract fingerprint | `9a5de51b320a36d3e498301c8f3b2c228c11ed999032bc7b1d9ef8777f213f6f` |

All **18 controls** and **40,692 independent audit checks** passed. Worker time
was 1,100.49 seconds; full pipeline time was 1,114.97 seconds. These are elapsed
wall times, not isolated GPU compute measurements. Generated checkpoints and
complete native traces remain outside Git, with compact provenance committed.

Frozen producer: `3fa71e2f1bc658aa62d8ed56f7bbfaff67093968`;
[draft PR #119](https://github.com/kadiru/sircus/pull/119), stacked on #116.
Source CI run 42744234269 passed. Ubuntu validation passed 48 tests; the Node
interoperability case skipped there passed on Mac. Ten UI tests passed. The
completed panel and its 35-cell matrix were visually verified.

All 27 final isolation checks passed. The finite 4090 worker exited; the existing
4070 workload and viewer retained their PIDs/start identities. Historical
reports and unrelated UI content were preserved. No service restart, hardware
command or Isaac configuration change occurred.

Next: inspect the two lost captures and additional setup declines while keeping
the current model. Bucket perturbations remain deferred. This held-endpoint
simulation study does not validate learned arm transit, grasping or perception.
