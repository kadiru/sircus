# Sweep versus PPO: first paired seed

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
