# Expanded shot selector — November 24, 2026

**Completed and independently audited:** both the current and expanded planners
captured **44 of 50 fresh bucket placements (88%)** on their first drop. The
expanded planner recovered two placements and lost two others, so this test
shows **no aggregate capture improvement**. The live planner was not changed.
The machine-readable [summary](summary.json) records every tested placement,
method outcome, timing, control result and evidence hash.

## What changed

A shot setting specifies bucket location, board position/orientation, drop gap
and arm configuration. The expanded library consolidates recorded settings,
removes repeated copies from training weight, and keeps successful examples
that contacted the table or robot after the first board rebound.

The learned selector is a supervised network: **44 geometry/source-hint inputs
→ 64 → 32 → 2 outputs** for setup feasibility and capture given setup. It orders
candidate shots; exact arm/path checks still precede release. This is not PPO,
image perception, point-cloud learning or a controller that predicts joint
motions over time.

| Evidence / split | Distinct rows or settings | Bucket positions |
|---|---:|---:|
| Included native records, before deduplication | 28,876 attempts | — |
| Distinct native settings | 26,789 | — |
| TRAIN, including setup-only evidence | 30,607 rows | 1,195 |
| Validation | 2,469 rows | 292 |
| Successful TRAIN settings available to propose shots | 12,341 | 1,034 |

The library contains **15,001 recorded captures**, representing **13,415 distinct
successful settings across TRAIN and validation**. Its TRAIN candidate settings
comprise **7,248 direct, 4,994 table-assisted, 74 robot-assisted and 25 mixed**
contact categories. These are recorded outcomes, not fresh-position accuracy.
Contact presence does not establish that the contact caused a useful rebound.

All **33 usable fixed-grid positions are in TRAIN**. Other historical positions
are assigned deterministically to TRAIN/validation, keeping an entire coordinate
in one split. Historical TEST observations are explicitly reused as development
data for this new experiment; their original published results remain intact.

Each bucket receives equal total loss weight per output head; each distinct
setting within it receives equal weight. Repeats determine empirical capture
fractions, not extra weight. This included corpus has **zero conflicting repeat
outcomes**, so its actual capture targets are 0 or 1. Two TRAIN positions provide
only setup evidence; the capture head has 1,193 contributing positions.

Adam uses learning rate **0.003**, weight decay **0.0001** and seed
**2026112411**. Validation loss at epochs **100 / 250 / 500** was
**0.62759 / 0.56848 / 0.52330**; epoch **500** was frozen. Fitting ran on CPU.

## Controlled comparison

- **Current:** the live hybrid planner's original model, source library and
  flight ranking. Seven input files match its audited `fast-planning-v2` archive
  byte-for-byte.
- **Expanded:** expanded TRAIN-only sources, refitted selector and a flight
  ranking rule that permits contacts after the board rebound. This compares
  **complete systems**, not the isolated effect of model training.
- **50 new positions**, at least **5 mm** from all **5,016 excluded historical
  coordinates**, including this dataset and the previous 50-position test.
- Both planners receive **30 seconds of decision time**, independent reusable
  CPU workers and the same exact arm/path checks. Full request time additionally
  includes worker initialization, shutdown and evidence handling.
- One selected native drop per planner per position at **64 substeps**. A missing
  checked setup remains a failure in the 50-position denominator. Weights, menus,
  choices and proofs freeze before test drops.
- Before and after TEST: three direct captures, three table-assisted captures and
  three expected misses. Version 4 scoring requires the first confirmed rebound
  on the board, whole-ball bucket entry and one uninterrupted second inside;
  later contacts with other surfaces are allowed.

| Result on 50 fresh placements | Current | Expanded |
|---|---:|---:|
| First-drop captures | 44 (88%) | 44 (88%) |
| Direct / table-assisted captures | 34 / 10 | 34 / 10 |
| Setup rejections / invalid physics | 0 / 0 | 0 / 0 |
| Released misses | 6 | 6 |
| Mean decision time | 11.06 s | 10.62 s |
| Mean full request time | 11.29 s | 10.93 s |
| Median full request time | 6.78 s | 7.47 s |

Both planners captured the same **42** placements; each captured **two** the
other missed, and both missed **four**. The expanded mean request was 0.36 seconds
shorter, but its median was 0.69 seconds longer. This single 50-placement test
does not establish a consistent speed or capture advantage.

Both nine-shot control blocks passed: each produced three direct captures,
three table-assisted captures and zero captures among three negative controls.
The **118 native releases** comprise 100 test drops plus 18 controls. The 4090
lease lasted **1,157.91 seconds (19.30 minutes)**. The independent audit passed
**139,349 checks**, including frozen provenance, position separation, native
initial conditions and rescoring of recorded ball/contact arrays.

The immediate follow-up is to inspect the two gained and two lost placements
before changing the interactive default; the successful historical settings
and both frozen planners remain available.

## Reproduction and provenance

Frozen computational producer:
**`fafdc4447ee0a8942cdff61193ba30b845873f17`**, draft
[PR #165](https://github.com/kadiru/sircus/pull/165).
The pinned Ubuntu environment is `/home/kadir/sircus-mjlab/venv/bin/python`.
No frozen computational files may change between stages.

| Artifact | SHA-256 |
|---|---|
| Contract | `ae8da6f745639d4c98113449e8e1626e579c9976262e2368debfb183feffd2c3` |
| Fifty layouts | `20fefbfe9e093a2fc24a091646a2c69d3a847df681749a6e4f401041c4d6cee7` |
| Expanded checkpoint | `c34a574a6b6e39416fcd581184ec5b40dce31f5a93a17c6a956269aea24b8980` |
| Current checkpoint | `3ea83197149204bb92782f9efd9a7ee985d6c7bbbfb0823bd068642eebdd1a1e` |
| Library manifest | `70fe0d03c591a94e3e21b2da5a802a259960dae591601a76c48a1c736beaed91` |
| Library summary | `3d4163e77336259be94d89eed5302d3bace6934517a4bc857dbe7483a5fb2779` |
| Fit report | `6cabf391544917abebebb4b3f83e08e2f2d4bfea6c43af69c25c7c54c6f51a31` |
| Final report | `d72966f031f91ad1f231b0b8c9bed3e64f37032c1f0d20b020c22bd041baa0d2` |
| Evidence manifest | `3117645239b4b56ba2a03ea81ccc616435dfea50b6014db864952630aa10a8ad` |
| Independent audit | `af5f6d4efdba7829bc3fda48a67c28e655a0306ca613abe5ce55659e1afe585a` |

Ubuntu evidence locations:

- Runtime: `/mnt/robot-training/sircus-mjlab/studies/expanded-selector-v1`
- Dataset: `/mnt/robot-training/sircus-mjlab/output/expanded-shot-library-v1`
- Study: `/mnt/robot-training/sircus-mjlab/output/expanded-selector-v1`
- Final audit: sibling `expanded-selector-v1-audit.json` (passed).
- Existing launcher: `/mnt/robot-training/sircus-mjlab/run-expanded-study.py`.
  Its log and status are siblings of the study, outside immutable evidence.

The launcher runs `fit → prepare → run → finalize → audit` using
`python -u -m sircus.mjlab.expanded_selector_study` with these input arguments:

```text
--neural-study /mnt/robot-training/sircus-mjlab/neural-selector-v1
--neural-audit /home/kadir/sircus-mjlab/output/neural-selector-v1-audit.json
--current-planner-study /home/kadir/sircus-mjlab/output/fast-planning-v2
--current-planner-audit /home/kadir/sircus-mjlab/output/fast-planning-v2-audit.json
--library /mnt/robot-training/sircus-mjlab/output/expanded-shot-library-v1
--previous-comparison /mnt/robot-training/sircus-mjlab/output/board-first-selector-v1
--parent-study /home/kadir/sircus-mjlab/output/fixed35-sixhour-v2
--parent-audit /home/kadir/sircus-mjlab/output/fixed35-sixhour-v2-audit.json
--reference /home/kadir/sircus-mjlab/output/corrected-grid-v1
--assets /home/kadir/sircus-isaac/assets
--history-root /home/kadir/sircus-mjlab/output
--history-root /mnt/robot-training/sircus-mjlab/output
--history-root /mnt/robot-training/sircus-mjlab/neural-selector-v1
--gpu-lock /home/kadir/sircus-isaac/scene.lock
--gpu-wait-s 7200
```

A reproduction requires a **new output directory**; never rerun the launcher
against existing evidence. CPU stages set `CUDA_VISIBLE_DEVICES=''`; only `run`
exposes the RTX 4090 UUID. The native stage acquires the shared lease, checks for
foreign workloads and caps its lease at one GPU-hour. The final audit uses a new
`--audit-output` outside the study. See the
[complete protocol](../../sircus/mjlab/EXPANDED_SELECTOR_STUDY.md) and
[library builder](../../sircus/mjlab/expanded_shot_library.py).

## Limits

The bounded source pass covers 60 archived studies. Unsupported records,
missing original command provenance, other scene/contact protocols and non64
substeps are excluded, without deleting their evidence. Inputs are privileged
simulator geometry. The ballistic ranker models direct/one-table flight, not
arbitrary arm-bounce dynamics. Native flight holds the arms at checked endpoints;
it does not validate driven arm trajectories, perception or real hardware.
The 4070 workload, Isaac setup and live planner remain unchanged.

Storage cleanup preserved every unique historical dataset: 76 byte-identical
native-array copies were consolidated using verified hard links, reclaiming
**2,871,066,624 allocated bytes**. The task's duplicate development library was
removed only after all six files matched the retained final library, removing
another **126,003,130 logical bytes**. Source/runtime hashes remain unchanged;
this is duplicate removal, not deletion of old training evidence.
