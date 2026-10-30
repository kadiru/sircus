# Scene-conditioned learning checkpoint

This study builds on the reviewed PR #2 merge
`6f00ac3faa4e34d508b005891e62d9911ead81cf`. It is a numerical one-shot learning
experiment using the existing reduced-order compliant board and tapered bucket.
It does not commission robot execution or replace the active dashboard/runtime.

The first [published checkpoint and scorecard](../baselines/scene-conditioned-20261030/README.md)
uses 24/6/8 layouts and seed 20261030. Frozen inference captured 8/8 test shots,
the fixed baseline 6/8, and separately labeled per-scene search 8/8. No test
layout was excluded. These eight synthetic tests do not establish reliable
generalization; the checkpoint was not retuned after seeing them.

## Experiment boundary

Each episode supplies a board center and upright bucket position. The policy
chooses roll, pitch, board-local U/V and world-Z release gap. Board center, board
yaw and bucket placement stay fixed during a shot. Gap, finite-board contact
height and ball-center release coordinates are recorded separately using the
existing coordinate contract. Tool offsets and contact/support parameters remain
provisional; the reported 175 mm bucket height is still unresolved.

The initial dataset covers a declared, narrow region around the existing scene.
Train, validation and test contain separate seeded layouts, with disjoint layout
hashes. Scene-generation, action-search and rollout seeds are recorded. Layout
observations contain scene geometry, not a teacher's chosen action or its outcome.

Teacher search produces action labels on training layouts. Unsuccessful labels
and invalid attempts remain identified; choosing the best sampled action does
not prove capture. A small ridge regression model learns scene-to-action weights,
using training-only normalization. Validation can choose regularization. The
checkpoint is frozen before test evaluation.

Inference evaluates the saved model directly; it does not run a per-scene search
or refine an action using test outcomes. Fixed-action and explicitly named search
baselines are evaluated separately on the same held-out layouts and seeds.
Search-baseline performance must never be presented as learned-policy inference.

## Scoring and feasibility

All methods use the PR #2 versioned scorer: board bounce, valid whole-ball entry
and one continuous simulated second inside. Clean and rim-assisted captures
both count; containment loss resets the timer. Unknown/timeout/model-limit
results are not successes. See [the scoring contract](SCORING_CONTRACT.md).

Both-arm endpoint IK is checked separately from numerical contact simulation.
An endpoint-feasible action has **not** passed collision-aware native path
planning. Evaluation denominators include invalid/unreachable/model-limit
proposals. This checkpoint has no native or physical success rate.

## Existing Mac UI

The explicit study panel accepts JSON checkpoint and scene/episode files. It
requests a numerical prediction/rollout through `/api/study/preview` and reuses
the existing trajectory viewer. A separate subprocess avoids collisions between
legacy and two-arm Python module names. The request cannot select an executable,
load a pickle, modify scene files, invoke Viam or command a physical component.

Numerical preview is distinct from the normal Viam controls. Native state cannot
silently replace its replay, and returning to the Viam scene is explicit. Exported
preview evidence is marked synthetic, with checkpoint identity, coordinate and
scoring versions and unverified native paths. Loading an incompatible checkpoint
fails rather than falling back to another model or simulator.

**The existing running service is not deployed by this task.** The new UI/API
becomes available only after a separately authorized deployment. V2 native replay
remains blocked until its matching adapter is implemented; that compatibility
guard is not removed here.

## Reproducibility and remaining work

Use the project's pinned Python environment. This command creates a new output
directory and never connects to a robot or Viam runtime:

```sh
python sircus/two-arm/train_scene_study.py \
  --train-scenes 24 --validation-scenes 6 --test-scenes 8 \
  --teacher-budget 12 --oracle-budget 4 --seed 20261030 \
  --output .runtime/studies/checkpoint-001
```

`episodes.json` freezes all layouts before teacher search. `training-candidates.jsonl`
and `training-labels.json` preserve teacher attempts; `validation.json` records
regularization selection. `checkpoint.json` is written and reloaded before test
inference; its bytes and the layout manifest are checked unchanged afterward.
`test.json` keeps policy, fixed-action and search-baseline results separate.
`summary.json` reports every test layout, failures, model limits, capture classes
and opening-crossing error statistics. Layout/teacher/rollout RNG streams have
separate identities; rollout seeds use the physics engine's uint32 range.

For the UI, select `checkpoint.json` and `preview-scene.json` in the study panel.
The latter identifies one predetermined test layout and seed, without choosing
a favorable test result. The preview evaluates frozen weights once, checks
endpoints, then runs numerical contact physics. No inference optimization occurs.
Offline `preview.json` records the same numerical result; replay is visual only.

To inspect this branch independently of the running Mac dashboard, use an unused
port and explicitly disable hardware:

```sh
python sircus/simulation/server.py --offline --physical-mode disabled --port 18769
```

Open `http://127.0.0.1:18769/`, then load the two files from
`baselines/scene-conditioned-20261030/`. **Predict & preview** evaluates the saved
model and numerical physics once; **Replay prediction** only plays those frames.
The study badge, hidden arm meshes and disabled Viam scene controls identify the
mode. This starts no Viam server, reuses no cloud identity and requires no robot
credentials. Use the matching source checkout and pinned requirements; source
changes can intentionally invalidate the checkpoint.

The narrow distribution uses board X 570–625 mm, Y −25–25 mm, Z 330–375 mm and
bucket offsets from the board of X 40–110 mm and Y 200–275 mm. Only positions
vary between layouts. Each shot samples explicit release uncertainty of ±2 mm
in world X/Y and ±3 mm in gap, identically seeded across evaluated methods.
Contact parameters, board yaw, bucket dimensions and arm/tool geometry are fixed
and remain provisional. Scene IDs differ across splits, and geometry-hash
overlap is rejected. These are distinct sampled layouts from the same narrow
distribution, not held-out parameter regions or an out-of-distribution test.

Five action bounds are roll −20–20°, pitch 10–50°, U −70–70 mm, V −100–100 mm
and gap 60–380 mm. U/V are independent policy outputs. The fixed-action baseline
is declared before the study and comes from the historical nominal scene; it is
not tuned on the test set. The search baseline optimizes each test layout with
its stated budget and is reported as search, not learned inference.

Publish a small checkpoint, serialized layouts and summary alongside source, with
checksums and source/dependency manifests. Full training/evaluation rows stay in
ignored runtime storage. A published checkpoint identifies its exact simulator,
scorer, action/observation schema, distribution and model weights; changed assets
or incompatible source must cause a compatibility error.

Mock and headless tests establish implementation boundaries, not accurate real
contact dynamics. Broader distributions, repeated studies, independent Ubuntu
reproduction, measured calibration, native replay and eventual operator-approved
hardware trials remain distinct follow-up work. Audio integration remains on its
separate draft; no audio PR is merged as part of this study.
