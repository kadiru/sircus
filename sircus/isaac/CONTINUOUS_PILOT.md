# Broad feasibility and continuous PPO pilot

The [completed study and checkpoints](../../baselines/isaac-continuous-pilot-20261105/README.md)
show learning on known training layouts, with no held-out capture improvement.
This is a separate, single-seed, terminal shot-selection task. Grasping, motor
trajectories, reactive interception and perception are not learned.

## What the policy chooses

A shared 2×64 tanh actor/critic receives five privileged geometric features:
bucket X/Y, distance to the workspace reference, and sine/cosine of its bearing.
Its five tanh-Gaussian actions select bearing offset around the bucket (−180 to
180°), radius (280–1200 mm), board-center height (140–600 mm), vertical drop gap
(80–560 mm), and board tilt from horizontal (12–52°). Heading faces the bucket;
release XY is above board center. Gap is ball-bottom release Z minus board-center
Z; ball-center height adds the provisional 37.5 mm radius.

The hand-set initial mean uses board height 350 mm, gap 240 mm and tilt 29°.
Radius starts at bucket-to-reference distance, clipped to 290–1190 mm, toward the
shared workspace reference (670, −100) mm. Initial log standard deviations are
−2.2/−2.0/−1.5/−1.5/−1.2. This declared prior is not a learned result. There is
no successful-shot menu, action projection, resampling of rejected commands or
per-scene ballistics search during inference. Numerical IK solves the mechanical
endpoint problem without changing the requested shot parameters.

## Feasibility and physical execution in simulation

The geometry survey retains 512 independently scrambled Sobol requests at each
of 35 distinct bucket sites: 17,920 requests. Full robust IK v2, independent FK and
joint-limit checks, board/table bounds and provisional joint-chain capsules
admitted 155 candidates at 33 sites. Of these, 153 passed sampled joint-path
proxies from canonical home. These are endpoint candidates, not capture
witnesses. `r1c7` and `r2c7` remain unresolved within this finite sampling budget.
Full-circle action bounds do not imply a reachable full orbit.

Native trials repeat qualification, sample 16 points along the commanded joint
path, and enforce the existing measured release gates: ≤2 mm position error,
≤0.5° board-normal error, ≤0.005 rad/s joint speed, and ≤0.1 mm/0.05° drift over
250 ms. A 1.75-second hold occurs before each measurement window starts. These
checks do not certify complete mesh/self-collision safety; inherited assets
still disable self-collision and the proxy omits some protrusions/contact pairs.

Every episode resets to the same canonical simulated home before filming and
holds for six seconds. Actual measured evaluation homes matched within 1 mrad.
Recording then shows live articulation drives to the chosen shot. There is no
joint-state reset at that scored endpoint. Ball release is initialized at zero
velocity; carrying, grasping and joint paths remain scripted. Earlier bounded-IK
and inconsistent-home smoke runs stay separate from the completed study.

## Training and evaluation

The 35 inherited bucket coordinates are divided before rewards into 25 training,
five validation and five test sites. These sites were explored in older project
work; the held-out designation applies to this fresh optimizer, not globally
unseen positions. Four geometry-coverage batches never enter PPO updates.

The fixed budget is 64 × 25 = 1,600 attempted training shots and 16 PPO updates, four
epochs each. Learning rate 0.0003, clip 0.2, value weight 0.5, squashed entropy
weight 0.005 and gradient norm 1. Invalid requests remain in the denominator and
buffer with the original reward. The unchanged capture evaluator requires a
board bounce, whole-ball entry and one continuous second retained, with no
external table/floor/robot route contact. Rim assistance counts; clean and
single-board-bounce captures are separate metrics.

Frozen initial and final deterministic policies each receive eight repeats at
every site, with no search or optimization feedback:

| Layout subset | Initial captures | Final captures |
|---|---:|---:|
| Training, 25 sites | 8/200 (4%) | 32/200 (16%) |
| Validation, 5 sites | 8/40 (20%) | 8/40 (20%) |
| Test, 5 sites | 8/40 (20%) | 8/40 (20%) |
| All 35 | 24/280 (8.6%) | 48/280 (17.1%) |

The number of capturing sites rose from 3 to 6, entirely through three added
training sites. Twenty-nine final sites remained 0/8. Identical nominal repeats
are not independent new layouts or perturbed robustness tests. Clean frozen
captures stayed 8→8. No training/evaluation capture used radius ≥600 mm; 176
admitted training releases in that range missed. No captured target exceeded
the conservative general dropper-only reach bound. Bucket-anywhere performance
and the intended far-target advantage are not demonstrated yet.

The training mixture yielded 62/1,600 qualified captures and 1,214 admission
failures, mainly IK/reach. All 386 admitted throws rebounded from the board.
Among 324 admitted misses, first external contact was table 297, robot 16, and
unobserved 11. Some had already contacted the bucket, so table contact does not
by itself prove an undershoot. Seventeen balls retained after a table-assisted
route were excluded from qualified training captures.

## Recording and verified results

The run took 61.34 minutes on the 4090, including filming and evaluation. The exact
saved camera and silver-gray stage remain fixed; there is no ball following.
Twenty prescheduled batches—four geometry coverage and sixteen early/middle/
late PPO batches—produced 19,929 native 1080p/30fps frames, 664.3 seconds. Prefix
captions use only earlier completed exploratory training outcomes, including
unfilmed batches; they are not frozen-policy performance estimates.

Full per-batch recordings remain alongside a setup-dwell-trimmed alternative.
No flight frames or attempted chapters are dropped to hide failures. The closest
table made six recorded releases and no captures; some other attempts were
withheld. Both arms show substantial measured movement, but joint winding from
IK branches must not be described as learned trajectories or a full bucket orbit.
Raw RGB, complete 240 Hz ball/arm/board/contact arrays and logs stay outside Git.

Independent CPU audits verified fresh seed initialization, 66 checkpoints,
84 trial batches, 16 update links, 1,600 optimizer samples and all admitted native
outcomes from 2,733,850 observations. Maximum CPU/GPU log-likelihood difference
was 6.68e−6. Native camera readbacks bind 2,393 release frames to telemetry.

## Reproduction and interface

Use the pinned Isaac Sim 5.1 / Isaac Lab 2.3.2 environment and generated UF850
assets in [README](README.md). Native producer commit is `c0e5dfd`; audits are
`1cb55b2`; video/report exporter is `d407651`. Preserve the existing scene lock
and dedicated GPU supervisor; do not start a competing scene or restart services.

The committed `baselines/isaac-expanded-farm-20261102/` is a portable source
bundle: its contract, qualified geometry metadata, camera and probe receipts
pass `presentation_recording.load_source` without historical raw videos.
First run the CPU geometry survey with that contract:

```bash
CUDA_VISIBLE_DEVICES='' python continuous_feasibility.py \
  --source-spec SOURCE_BUNDLE/contract.json --assets ASSETS \
  --output NEW_MAP --samples 512 --workers 8 --ik-protocol full_robust_v2
```

Through the existing bounded 4090 launcher, run:

```bash
python.sh film_cylinder_farm.py --headless --enable_cameras --device cuda:0 \
  --assets /workspace/assets --output /workspace/output/NEW_RUN \
  --continuous-source /workspace/SOURCE_BUNDLE \
  --continuous-camera-json /workspace/SOURCE_BUNDLE/selected-camera.json \
  --continuous-feasibility /workspace/output/NEW_MAP/feasibility.json \
  --continuous-batches 64 --appearance silver
```

Use `--continuous-batches 4` for a separate smoke. Auditors
`continuous_pilot_audit.py` and `continuous_native_audit.py` verify a completed
run. The CPU-only `continuous_pilot_report.py RAW NEW_EXPORT --media --action-edit`
produces per-cell settings, curves, complete clips and the trimmed alternative.
The read-only `continuous_pilot_server.py --output NEW_EXPORT --port 18794`
serves them without simulator or hardware controls.

Bucket inside top/bottom diameters remain 177/150 mm and outside 200/158 mm.
Reported height 175 mm still has unresolved overall-height versus interior-depth
meaning; the existing interior 175 mm plus 8 mm bottom is provisional. No hardware,
Viam, audio, existing service or 4070 deployment accompanies this study.
