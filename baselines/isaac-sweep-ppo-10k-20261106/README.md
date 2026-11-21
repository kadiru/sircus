# Sweep versus PPO: completed three-seed benchmark

All **six native production runs and six original independent audits passed**.
Each run spent exactly 10,000 paid TRAIN requests, including all rejects:
60,000 total. Another 420 frozen assessment requests per run are accounted for
separately. The unchanged finite coordinator completed on November 8, 2026,
at 10:30 UTC. This final addendum preserves every historical summary and model
byte; it does not relabel the interim results below.

| Final unseen TEST result, each /70 | Seed 20261121 | Seed 20261122 | Seed 20261123 |
|---|---:|---:|---:|
| Sweep captures (clean) | 23 (15) | 22 (12) | 25 (18) |
| PPO captures (clean) | 28 (10) | 28 (7) | 28 (14) |
| Sweep rejected requests | 19 | 29 | 20 |
| PPO rejected requests | 18 | 17 | 42 |

| Rate or cost | Sweep mean ± sample SD | PPO mean ± sample SD |
|---|---:|---:|
| Final unseen TEST capture | 33.33% ± 2.18 pp | 40.00% ± 0.00 pp |
| Final unseen TEST clean capture | 21.43% ± 4.29 pp | 14.76% ± 5.02 pp |
| Final unseen TEST rejection | 32.38% ± 7.87 pp | 36.67% ± 20.22 pp |
| Final validation capture | 32.38% ± 7.19 pp | 32.38% ± 1.65 pp |
| Final validation clean capture | 20.95% ± 4.36 pp | 12.38% ± 1.65 pp |
| Exploratory TRAIN capture | 7.22% ± 0.79 pp | 15.09% ± 0.44 pp |
| Paid TRAIN rejection | 83.99% ± 1.21 pp | 65.57% ± 2.18 pp |
| Full TRAIN method wall | 4.618 ± 0.359 h | 6.389 ± 0.458 h |

PPO gained **5, 6 and 3** TEST captures in the paired seeds; sweep gained
**5, 5 and 4** clean captures. The paired PPO-minus-sweep TEST capture difference
was +6.67 percentage points on average, with sample SD 2.18 pp. The clean-capture
difference was −6.67 pp, with sample SD 0.82 pp. PPO used 38.3% more mean TRAIN
wall time. These are capture/clean-entry/cost tradeoffs, not a single method
dominating every outcome or a model-promotion decision.

All methods and seeds use the **same 35 validation and 70 unseen TEST positions**.
They are three repeated training seeds, not 210 independently sampled TEST
positions. Means, ranges and seed sample SD are descriptive, not confidence
intervals. PPO's equal aggregate 28/70 counts do not establish equal per-position
behavior or zero uncertainty; its third seed rejected 42/70 TEST requests.
Capture includes rim-assisted success; every clean rate above uses **all requested
positions** as its denominator, not only released or captured shots.

Exploratory TRAIN counts reflect changing proposals and repeated positions.
Separate final frozen direct predictions on the 35 known TRAIN positions captured
20 / 20 / 21 for sweep and 13 / 13 / 14 for PPO. Those are training diagnostics,
not held-out results. The 420 assessment requests per run comprise 175 frozen
TRAIN diagnostics, 175 validation requests and 70 final TEST requests. No final
model was chosen from validation, and no held-out physics search, retry,
adaptation or free replacement was used.

| Frozen validation captures /35 | 0 TRAIN | 1,000 | 2,500 | 5,000 | 10,000 |
|---|---:|---:|---:|---:|---:|
| Sweep 20261121 | 3 | 6 | 8 | 11 | 9 |
| Sweep 20261122 | 3 | 8 | 7 | 9 | 11 |
| Sweep 20261123 | 3 | 9 | 10 | 12 | 14 |
| PPO 20261121 | 3 | 5 | 7 | 10 | 11 |
| PPO 20261122 | 3 | 4 | 5 | 8 | 12 |
| PPO 20261123 | 3 | 5 | 9 | 8 | 11 |

## Actual compute and memory

| Run | Full TRAIN wall h | Headless episode h | Rendered episode min | Peak process RSS MiB | Sampled GPU MiB headless / render |
|---|---:|---:|---:|---:|---:|
| Sweep 20261121 | 4.286 | 4.385 | 11.31 | 7,072.2 | 6,188.1 / 6,169.3 |
| Sweep 20261122 | 4.998 | 5.143 | 11.13 | 6,989.5 | 6,221.4 / 6,201.3 |
| Sweep 20261123 | 4.571 | 4.735 | 8.14 | 6,962.8 | 6,189.4 / 6,169.3 |
| PPO 20261121 | 5.985 | 6.132 | 9.19 | 7,408.9 | 6,216.1 / 6,199.3 |
| PPO 20261122 | 6.295 | 6.408 | 8.98 | 7,448.7 | 6,217.4 / 6,199.3 |
| PPO 20261123 | 6.886 | 6.974 | 12.90 | 7,631.6 | 6,217.4 / 6,199.3 |

TRAIN wall includes proposal/inference, before-batch checkpointing, native
execution, serialization and updates, excluding evaluation. Episode-only
headless/render intervals **include evaluation** and exclude outside learner,
proposal and checkpoint calls; they are not additive to TRAIN wall. Exact
phase-separated CPU seconds, wall seconds, setup/scoring intervals and measured
algorithm overheads are in `final-comparison.json` and the interface. Memory is
process-lifetime RSS high-water and sampled whole-device GPU occupancy, not
policy-only memory, aggregate worker usage or isolated phase peaks. Geometry
preparation wall time remains unknown. Subsequent audits/video publication are
separate work; no energy or monetary cost was measured.

## Final artifacts, interface and limits

- `final-comparison.json`: completed six-run aggregate curves and frozen results,
  paired descriptive statistics, scoped resources and all audit/checkpoint/video
  bindings. Frozen TRAIN diagnostics remain separate from validation and TEST.
- `ppo-seed20261123.pt`: exact final actor/critic and training-state checkpoint,
  136,545 bytes; SHA256
  `f4fa02cfff64f5bf3f07fe5b280375fe5bb7a75c0c3efbe46d2c3f2b28481f0f`.
- `manifest.json`: additive final entries. Every earlier file and checksum is
  preserved, including the three historical partial-result summaries and all
  five previously packaged final models.

The final PPO audit SHA256 is
`74daeb252612041503a91ad74c1f6227d24bbc458b7408523714e3b919ff5d2b`.
Its prescribed final frozen proof SHA256 is
`ae966e48f890607dd6b23eb7d1625d51d5c899a0317d3d67400f376151f47a41`.
All six exact models match their original audits and final frozen assessments.
CPU-only guarded loading verifies the final PPO's 73 updates and 10,000 attempted
shots, finite policy weights/inference and fingerprint rejection without
initializing CUDA or spending any new physics trials.

The read-only [interface](http://127.0.0.1:18797/#frozen) exposes complete learning
curves, all six audit identities, frozen assessments, compute and 24 prescheduled
native TRAIN movie chapters. Videos keep the exact selected camera, scene and
normal timing; they are execution evidence, not held-out scores. All-rejected
setup-only chapters remain included and labeled. Scripted joint travel, loading
and initialized release are not learned motor control. After the earlier Mac
reader and mirror exited, the user explicitly approved relaunching only the
frozen GET-only reader. The complete local report needs no mirror restart; no
native job or other service was restarted.

The original archive/experiment, source, physics, scoring, materials, rewards,
geometry, camera, splits, run order and release gates stayed unchanged. Privileged
geometry inputs, provisional bucket/material/mount assumptions, only three seeds
and the shared finite position set limit the conclusion. No perception-driven
control, grasp learning, physical validation, new ball properties or Cosmos
development follows this benchmark. No model is promoted. Raw observations,
media, full audits, host calibration and runtime files remain outside Git.

## Historical first paired seed

This is an **interim checkpoint**, not the finished three-seed benchmark. Both
seed-20261121 runs completed their prescribed 10,000 TRAIN requests and passed
the unchanged original independent auditor. The four remaining runs keep the
original experiment; these results do not select or tune their parameters.

| Final metric | Sweep | Continuous PPO |
|---|---:|---:|
| Paid TRAIN requests, including rejects | 10,000 | 10,000 |
| TRAIN captures | 803 | 1,535 |
| TRAIN clean captures | 240 | 439 |
| TRAIN rejected setups | 8,306 | 6,549 |
| Optimizer updates | 0 | 73 |
| Frozen validation captures / 35 | 9 | 11 |
| Frozen TEST captures / 70 | 23 (32.9%) | 28 (40.0%) |
| Frozen TEST clean captures / 70 | 15 (21.4%) | 10 (14.3%) |
| Frozen TEST rejected setups | 19 | 18 |
| Full TRAIN method wall, seconds | 15,428.5 | 21,545.1 |
| Lifetime peak process RSS, MiB | 7,072.2 | 7,408.9 |
| Sampled whole-device occupied GPU memory, MiB | 6,188.1 | 6,216.1 |

PPO captured at five more TEST positions in this pair, but sweep had five more
clean captures. This does **not** establish a three-seed winner or statistical
significance. Exploratory TRAIN rates are separate from frozen prediction.
Neither method ran a physics search, retried, or adapted at a held-out position.
The final 10,000-request artifacts were selected in advance; the sweep's earlier
validation peak does not replace its final model.

| TRAIN requests before assessment | Sweep VAL / 35 | PPO VAL / 35 |
|---:|---:|---:|
| 0 | 3 | 3 |
| 1,000 | 6 | 5 |
| 2,500 | 8 | 7 |
| 5,000 | 11 | 10 |
| 10,000 | 9 | 11 |

The unchanged capture definition requires a board bounce, whole-ball entry and
one continuous second inside, without an external-contact route. Rim-assisted
captures count; clean captures are reported separately. The policy chooses five
cooperative shot parameters using privileged geometry. Joint travel, ball
loading and release initialization remain scripted. This does not validate
perception, grasping, learned motor control, new ball properties or real arms.
The reported 175 mm bucket height and contact/material assumptions remain
provisional.

## Files and verification

- `first-pair.json`: complete aggregate learning curves, frozen assessments,
  phase-separated resource measurements, source identities and all eight
  prescheduled native video receipts. No raw camera/state observations.
- `sweep-seed20261121.json`: exact final guided-sweep library, 31,831 bytes.
- `ppo-seed20261121.pt`: exact final actor/critic and training-state checkpoint,
  136,545 bytes. Inference loads with `weights_only=True` and does not restore
  the optimizer or RNG.
- `manifest.json`: checksums of the packaged model/result files.

All model bytes match both the final frozen assessment and the passed original
audit. Sweep audit SHA256:
`e126486453c9b50886dcf2c910d266cdc2cb44019934b9f43475d296a7ced1cc`.
PPO audit SHA256:
`fee6a8e53a5aa982ac1bfd8bfc369813f5b7747737e896b7d4dcd1d36c09149d`.
The full audit/observation/media files stay outside Git. The loopback reader on
port 18797 exposes current curves, audit status and videos; the raw source is
the frozen `sweep-ppo-10k-20261106-v2` study on Ubuntu.

The exact packaged bytes were also loaded on Ubuntu CPU through the original
guarded loaders. Both loads passed, all actor weights were finite and on CPU,
CUDA remained uninitialized, and incompatible geometry/protocol fingerprints
were rejected. This packaging check spent zero new physics trials.

The original audit rescored 1,985 sweep and 3,693 PPO releases across TRAIN and
evaluation, checked all 294 checkpoint artifacts per method, and preserved the
full 8/6-DOF release gates. PPO CPU likelihood error was at most 6.35e-6.

Run this load-only check from the repository root in a CPU environment with the
documented NumPy/SciPy/Torch dependencies. It starts no simulator or robot:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=sircus/isaac python - <<'PY'
from pathlib import Path
import json
from shot_benchmark_methods import GuidedSweep
from continuous_policy import load_checkpoint

root = Path('baselines/isaac-sweep-ppo-10k-20261106')
summary = json.loads((root / 'first-pair.json').read_text())
protocol = summary['protocol']
sweep = GuidedSweep.load(root / 'sweep-seed20261121.json',
    protocol_fingerprint=protocol['fingerprint'])
policy, metadata = load_checkpoint(root / 'ppo-seed20261121.pt',
    study_fingerprint=protocol['fingerprint'],
    geometry_fingerprint=summary['source']['geometry_fingerprint'],
    domain_fingerprint=protocol['domain']['fingerprint'], device='cpu',
    expected_updates=73, expected_attempted_shots=10000)
assert sweep.observed == sweep.issued == 10000 and not sweep.pending
assert metadata['seed'] == 20261121
print('Both final artifacts passed their original compatibility guards.')
PY
```

The training-wall measurement includes proposal, checkpoint, native execution
and updates, excluding evaluation. The headless/render intervals in JSON
include evaluation and exclude outside learner overhead; do not add them to
TRAIN wall or treat them as a second total. RSS is a process-lifetime high-water
measurement; GPU occupancy includes the simulator/rendering context and is not
policy-only memory. Recording and subsequent independent audits are identified
separately. See the [frozen protocol](../../sircus/isaac/SWEEP_PPO_10K.md).

## Second PPO seed addendum — November 7, 2026

The second PPO run, seed **20261122**, completed its prescribed 10,000 TRAIN
requests and passed the unchanged original independent auditor. This addendum
preserves the first-pair JSON and both earlier model files byte-for-byte. Three
of six production runs are now complete and audited; the remaining runs retain
their original budgets, source, split, reward and evaluation protocol. This is
still an interim benchmark, with no method winner or model promotion.

| Final metric | PPO seed 20261121 | PPO seed 20261122 |
|---|---:|---:|
| Paid TRAIN requests, including rejects | 10,000 | 10,000 |
| TRAIN captures | 1,535 | 1,534 |
| TRAIN clean captures | 439 | 485 |
| TRAIN rejected setups | 6,549 | 6,344 |
| Optimizer updates | 73 | 73 |
| Frozen validation captures / 35 | 11 | 12 |
| Frozen validation clean captures / 35 | 4 | 5 |
| Frozen TEST captures / 70 | 28 (40.0%) | 28 (40.0%) |
| Frozen TEST clean captures / 70 | 10 (14.3%) | 7 (10.0%) |
| Frozen TEST rejected setups | 18 | 17 |
| Full TRAIN method wall, seconds | 21,545.1 | 22,663.2 |
| Lifetime peak process RSS, MiB | 7,408.9 | 7,448.7 |
| Sampled whole-device occupied GPU memory, MiB | 6,216.1 | 6,217.4 |

Both PPO seeds captured at 28 of the same 70 prescribed TEST coordinates, with
different clean-capture counts. Equal aggregate counts do not establish equal
per-position behavior. These repeated seeds are not 140 independently sampled
TEST positions. The second paired sweep is still outstanding at this checkpoint;
the first sweep's 23/70 TEST result remains an interim reference.

| TRAIN requests before assessment | PPO seed 20261122 VAL / 35 |
|---:|---:|
| 0 | 3 |
| 1,000 | 4 |
| 2,500 | 5 |
| 5,000 | 8 |
| 10,000 | 12 |

New files:

- `second-ppo.json`: all 289 aggregate learning-curve points, five frozen
  assessments, phase-separated resources, original source/scoring identities,
  the independent audit bindings and four prescheduled native video receipts.
  It supplements `first-pair.json`; it does not replace or relabel it.
- `ppo-seed20261122.pt`: exact final actor/critic and training-state checkpoint,
  136,545 bytes. SHA256:
  `57d7658e9ed2f72ad09cf4e8523ab497ccbe0708364f1b8ec4a4c5c1aef1fe9f`.
- `manifest.json`: additive checksums for those two files; all existing model
  and result checksums remain unchanged.

The canonical second-PPO audit SHA256 is
`cec9cff0a6de59b07dd35de3342ee00bc4106d4801edc0237c570e006aedc1c8`.
It verified 10,000 paid TRAIN requests, 420 separate assessment requests,
73 updates, 3,889 released trajectories and 414 camera readbacks. Maximum CPU
likelihood error was 5.90e-6. The packaged checkpoint matches both the audit and
the final frozen assessment. Full raw audits, observations and video files stay
outside Git; the existing read-only interface exposes the video chapters.

CPU-only loading through the original guarded loader passed with seed 20261122,
73 updates and 10,000 attempted shots. All policy weights and a test inference
were finite on CPU; incompatible study, geometry and action-domain fingerprints
were rejected. CUDA remained uninitialized, and this check spent zero physics
trials. The first-pair load command above can also load the new checkpoint by
using `second-ppo.json` and `ppo-seed20261122.pt`, then checking seed 20261122.

All capture, geometry, privileged-input, scripted-motion and resource-scope
limitations above continue to apply. New-ball adaptation, perception and
physical deployment remain outside this frozen benchmark.

## Later sweep seeds addendum — November 7, 2026

Sweep seeds **20261122 and 20261123** completed exactly 10,000 paid TRAIN
requests and 420 separate evaluation requests each, and passed the unchanged
original independent auditor. Five of six runs are now complete and audited.
Final PPO seed 20261123 is still training on the 4090; this is an interim
package, with no complete three-seed method comparison or model promotion.
The historical first-pair and second-PPO summaries and models are unchanged.

| Final metric | Sweep seed 20261122 | Sweep seed 20261123 |
|---|---:|---:|
| TRAIN captures / clean | 717 / 201 | 646 / 222 |
| TRAIN rejected setups /10,000 | 8,355 | 8,535 |
| Frozen validation captures /35 | 11 | 14 |
| Frozen validation clean captures /35 | 7 | 9 |
| Frozen TEST captures /70 | 22 (31.4%) | 25 (35.7%) |
| Frozen TEST clean captures /70 | 12 (17.1%) | 18 (25.7%) |
| Frozen TEST rejected setups | 29 | 20 |
| Full TRAIN method wall, seconds | 17,994.1 | 16,456.5 |
| Lifetime peak process RSS, MiB | 6,989.5 | 6,962.8 |
| Sampled whole-device GPU occupancy, MiB | 6,221.4 | 6,189.4 |

The first sweep remains 23/70 with 15 clean captures. All three runs share the
same 70 TEST coordinates; they are not 210 independently sampled positions.
The prescribed final artifacts were assessed once per site, without shot
search, retries or validation-based selection. Remaining-budget accounting in
the JSON refers to the final PPO run's original 10,000-request budget, including
its concurrent partial progress. The live reader shows actual observed requests.

New files:

- `later-sweeps.json`: 289 aggregate learning-curve points and five frozen
  assessments per added run, scoped compute measurements, original audit
  bindings and all eight prescheduled video receipts. It binds both earlier
  summary hashes and contains no per-layout observations or media.
- `sweep-seed20261122.json`: exact final sweep state, 31,821 bytes; SHA256
  `475729036456835e12a2bf1bd703a666507ef353b5d9d1f435b8c4c6608adf2c`.
- `sweep-seed20261123.json`: exact final sweep state, 31,716 bytes; SHA256
  `8bed1e9549e31d2ad9af02c774f2763a261de28f1a258f432109129cd0fd0415`.
- `manifest.json`: three additive checksum entries; all earlier file mappings
  remain unchanged.

Original passed audit SHA256 values:

- Seed 20261122:
  `9dc2f11c781c9c9eac260096695ab957e6f20b3001e4e97e5023e51a080d0cf0`.
- Seed 20261123:
  `57138d0f28c13a42fff0b10cfad611168ba534b1a0b37fb0ac1ce05d786b1212`.

The exact downloaded model bytes match the final frozen assessments and the
original audits. CPU-only checks load both through `GuidedSweep.load`, verify
10,000 issued/observed requests and no pending proposals, and query TRAIN and
unseen coordinates without mutating their frozen state. Protocol, action-domain
and TRAIN-membership guards remain enforced. Geometry compatibility is bound
externally through the package's original source/audit fingerprints; the sweep
loader has no separate geometry-fingerprint argument. These checks spend no
physics trials and leave the simulator and current PPO process untouched.

All scheduled footage is retained, including seed 20261122's all-rejected 5k
chapter. The final third-sweep TRAIN chapter includes four captures elsewhere
in the farm; the foreground request failed board-arm IK. Video execution is
separate from frozen held-out performance. Original privileged-input,
scripted-motion, material, geometry and resource-scope limitations apply.
