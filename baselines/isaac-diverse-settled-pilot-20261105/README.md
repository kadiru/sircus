# Diverse-region PPO and measured settling

Completed simulation-only matched pilot: 1,600 attempted training shots, 16 PPO
updates, 25 training/five validation/five test coordinates. Every rejected active
attempt remains in the optimizer and denominator. Both older studies stay frozen.

| Frozen final subset | Broad | One local region | Diverse + settling |
|---|---:|---:|---:|
| train | 32/200 (16.0%) | 8/200 (4.0%) | 24/200 (12.0%) |
| validation | 8/40 (20.0%) | 0/40 (0.0%) | 0/40 (0.0%) |
| test | 8/40 (20.0%) | 8/40 (20.0%) | 8/40 (20.0%) |

This is one seed on inherited layouts, with geometry assistance at held-out
coordinates. Nominal repeats measure numerical repeatability; they are not new
layouts or independent noise trials. Proposal support, hybrid policy and setup
controller change together. No isolated PPO, arbitrary-table, physical-readiness
or collision-safe-path claim follows from these numbers.

Success requires a board bounce, whole-ball bucket entry and one continuous
second inside without intervening external route contact. Rim-assisted captures
count; clean captures remain separate. Historical definitions are unchanged.

The signed `support.json` is preserved byte-for-byte with its original fingerprint.
It contains offline geometry anchors, IK warm-start joints and path-proxy records,
not measured native trajectories. Local regions have qualified anchors and
unqualified interiors. Each supported site also has an explicitly unqualified
global exploration domain. Neither a box nor its envelope certifies feasibility.

The direct final-producer setup check admitted 48
before and 80 after on 100 fixed prior training
requests, with 32 repairs and 0
regressions. It made zero task-shot releases and measures admission, not capture
or held-out policy performance. Its producer identity remains separate from all
earlier prototype/queued runs.

The actual fixed-camera film is 1658.44s full and
1362.94s after setup-dwell trimming. All
2400 recorded release frames are retained. Large
visible arm travel is scripted; grasp/loading/joint paths are not learned. Film
digests/cuts are in `film.json`; media stays outside Git.

For CPU prediction with the repository's documented Torch environment, set
`PYTHONPATH=sircus/isaac` and `CUDA_VISIBLE_DEVICES=''`:

```python
import json
from pathlib import Path
import torch
from diverse_policy import load_checkpoint, build_observations, sample
from diverse_support import physical_actions

base = Path('baselines/isaac-diverse-settled-pilot-20261105')
m = json.loads((base/'manifest.json').read_text())
support = json.loads((base/'support.json').read_text())
policy, saved = load_checkpoint(base/'policy-final.pt', support=support,
    study_fingerprint=m['study_fingerprint'], geometry_fingerprint=m['geometry_fingerprint'],
    domain_fingerprint=m['domain_fingerprint'], support_fingerprint=m['support_fingerprint'], device='cpu')
study = json.loads((base/'study.json').read_text())
contexts = [c for c in study['cells'] if c['id'] == 'r2c3']
picked = sample(policy, torch.as_tensor(build_observations(contexts), dtype=torch.float32), deterministic=True)
print(physical_actions(picked['actions_normalized'], support, contexts, picked['region_indices']))
```

Ordinary inference rejects unknown/unresolved geometry contexts. Predicted
parameters are not native-qualified motion commands. Do not bypass the saved
study/domain/support identities or native release/IK/path guards.

`manifest.json` checksums every selected file. `requested-settings.csv` retains
requests, outcomes and derived errors; direct native pose/readback columns stay
private. Native/checkpoint/setup audits, exact source hashes and aggregate
provenance are retained. Raw physics/paired/transit observations, RGB/depth,
generated runtime configuration, credentials, queues, logs and media are excluded.
The 175 mm bucket-height interpretation and fixed material assumptions remain
provisional; see the unchanged signed contract.
