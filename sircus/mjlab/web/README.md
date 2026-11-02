# mjlab shot interface

A separate localhost-only simulation interface. It does not import Viam clients,
control physical hardware, start training, or change any existing dashboard.
The user can drag the bucket in the table plane, optionally lock board-center
XYZ or release-ball-center XYZ, request a prediction, and replay the resulting
simulated ball trajectory. Locks constrain positions; orientation remains free.

The manual solve button and backend both remain locked until the supplied report
has `status: completed` and a nonempty `frozen_evaluation` object. Build and mock
checks may run earlier. This gate respects the study-before-manual-testing order;
it does not certify that the learned policy is successful everywhere.

## Launch after the study

The completed two-model setup uses backend-only `SIRCUS_MJLAB_STUDY` (continuous),
`SIRCUS_MJLAB_BANK_STUDY` (categorical bank), `SIRCUS_MJLAB_ASSETS` and
`SIRCUS_MJLAB_GPU_LOCK`; see [the path template](../interface.env.example).
After both CPU auditors pass, use `prepare_interface` to produce a separate
display report and supply that to `--study-report`. Original reports stay intact.
Free requests use the frozen bank selector; positional-lock requests use the
continuous policy. Explicit search stays separate. Neither model guarantees
capture at arbitrary bucket or lock coordinates.

Run from a source checkout with its Python dependencies available:

```bash
python -m sircus.mjlab.interface \
  --study-report /absolute/local/study/report.json \
  --scene-json /absolute/local/study/scene.json \
  --solver sircus.mjlab.learning_interface:solve \
  --host 127.0.0.1 --port 18781
```

The numerical adapter defines its own explicit study/checkpoint configuration.
The server binds only localhost; use an SSH local forward when viewing on another
computer. There is no cloud part, credential, or automatic service installation.
Three.js and OrbitControls are served from the repository's existing vendored
assets under `sircus/simulation/web/vendor/`, with existing license/provenance.

The UI distinguishes a **frozen learned policy**, **per-scene search/refinement**,
and **saved setting lookup**. Search is requested explicitly; it is never silently
used to repair and inflate reported policy inference. The server rejects a search
or lookup result returned for a policy request. A missing or failed result remains
an unresolved request. A qualified simulated capture still requires the study's
board-bounce, whole-ball entry and continuous retention criteria.

## Adapter contract

The injected solver receives one dictionary:

```json
{
  "request_id": "unique-id",
  "mode": "policy",
  "bucket_xy": [0.65, -0.4],
  "constraints": {
    "board_position": null,
    "dropper_position": null
  }
}
```

Coordinates are **metres** in the shared world frame. `board_position`, when
supplied, is `[x, y, z]` of the impact-board center. `dropper_position` is the
**released ball's center**, not a flange or TCP. Both can be supplied; incompatible
constraints must return unresolved. Board dimensions remain 300 × 200 × 5 mm. Native board-body dimensions are
200 × 300 × 5 mm; the quaternion already includes the landscape mount’s Rz90.
The renderer applies that native quaternion once and never swaps X/Y again.
The bucket uses the measured inner/outer diameters and straight taper; the reported
175 mm height's interpretation remains provisional.

The callback returns a JSON-serializable object, for example:

```json
{
  "request_id": "unique-id",
  "method": "frozen_policy",
  "success": true,
  "message": "Measured qualified simulated capture",
  "configuration": {
    "board_center": [0.65, 0.0, 0.4],
    "board_normal": [0, -0.3, 0.953939],
    "board_quaternion_wxyz": [0.988418, 0.151758, 0, 0],
    "release_center": [0.65, 0, 0.6875],
    "q10": [],
    "q14": [],
    "link_positions10": [],
    "link_positions14": []
  },
  "metrics": {"board_bounces": 1, "retention_s": 1.0, "quality": 2},
  "trajectory": [{"time": 0, "ball": [0.65, 0, 0.6875]}]
}
```

This example specifies the schema, not a verified successful setting. Native
world-space link positions improve the pose drawing. The displayed links use
simplified surfaces, not collision meshes. Without native positions the interface
labels the arms as an endpoint schematic. The ball path is replayed from solver
samples; interpolation is for display. Whole-arm collision-free transit, grasping,
and point-cloud perception are not implemented by this viewer.

### HTTP

- `GET /api/study`: the report plus `interface_ready`, excluding raw `test_rows`
  trajectories; full evidence remains unchanged on disk.
- `GET /api/scene`: shared geometry, bucket bounds and initial positions. An
  optional scene JSON supplies top-level overrides; `simulation_only` stays true.
- `POST /api/solve`: validates a JSON request; returns 202 and `job_id`. Missing
  study evaluation returns 423; missing adapter returns 503; another active solve
  returns 409. Unknown controls and nonfinite coordinates are rejected.
- `GET /api/solution` (also `/api/status`): status `idle`, `running`, `completed`,
  or `error`; successful completion contains the adapter `result`.

The solver runs in a single background job. The browser freezes inputs while it
runs and checks the response request ID. The interface rejects cross-origin POSTs
and serves only explicitly allowed static files.

### Optional report fields

The study summary accepts `train_episodes`, `updates`, `seed`; test summary accepts
`frozen_evaluation.captures`, `.trials`, `.capture_rate`, `.layouts`. Coverage uses
`coverage.cells` (or `coverage_grid`) with rows containing `id`, `bucket_xy`,
`split`, `captures`, `trials`. Unresolved positions remain gray; test positions have
an outlined marker. Show scene fingerprint with `scene_fingerprint` or `contract_fingerprint`.

Actual study comparisons use `frozen_evaluation.comparison` entries `initial`,
`warm`, `selected`, `board_locked`, and `dropper_locked`, each with `attempts`,
`captures`, `invalid`, and `mean_reward`. The table displays all available
checkpoints, including failures and invalid endpoints; selected-policy results
are not replaced by search results. Learning curves use `learning_curve` rows
`update`, `captures`, `attempts`, `mean_reward`, and optional
`validation.mean_reward`. Training and validation curves are distinct from test
results.

Coverage-cell `confirmation: {attempts, captures}` is displayed separately from
the exploratory capture count. Gold-ringed cells passed every recorded small
perturbation repeat; this finite check does not establish unlimited robustness.
The displayed study region is X 100–1240 mm, Y −650–250 mm. Other tabletop
positions are explicitly unvalidated. `scope` text is displayed as supplied and
keeps privileged geometry and initialized/held endpoints visible.

## Mock checks

```bash
python -m unittest sircus.mjlab.tests.test_interface -v
node --test sircus/mjlab/tests/test_interface_ui.mjs
```

These checks use mocks and an ephemeral HTTP port. They do not call the numerical
adapter, connect hardware, start the production viewer, or start a training job.


### Optional relative-bank evaluation

An optional `bank_evaluation` keeps a subsequent categorical selector study
separate from the original continuous-v2 evidence. Its `status` must be completed
before manual solves unlock. It may include `train_episodes`, `updates`,
`test_layouts`, `source_scope`, and `comparison` entries `continuous_v2`,
`bank_uniform`, `bank_selected` with the same statistical fields as above.
These three methods share a **new split**; the original continuous table remains
unchanged. The uniform-bank baseline makes the hand-designed geometry prior
visible, rather than attributing all success to learned weights.

When the validated bank adapter is configured, free-position policy requests use
the frozen relative-bank selector. Locked-position requests retain the continuous
v2 policy. Explicit search remains a separate requested mode. The UI explains
this route; the actual adapter still must return truthful method and source
provenance. A `bank_policy` result is labeled as a frozen relative-bank policy
with a geometry prior.


The optional bank coverage map consumes `bank_evaluation.coverage` rows
`{bucket_xy_m, captures, trials, eligible_candidates}`. It distinguishes captured
(teal), eligible-but-missed (coral), and no-eligible/unknown feasibility (gray).
It is separate from the original train-grid exploration map. Raw
`bank_evaluation.rows` are omitted from the public report; summarized coverage
and comparison statistics are preserved, and the original evidence stays on disk.
After a solve, unlocked pose fields prefill from the actual configuration to
0.1 mm, allowing the operator to lock the displayed board/release position.


A duplicate **Find posture** button stays visible on the 3D scene. It shares the
same readiness/pending state and request handler as the full control-panel button,
so solving does not require scrolling the scene out of view. Displayed training
counts are **attempts**, including invalid endpoint requests. Metric labels round
angles to 0.1° and retention to 0.01 s, with readable entry-quality names.

### Optional expanded-bank development study

The read-only **Expand the shot bank** panel at `#expandedStudy` consumes
`expansion_evaluation: {search, training, status, active_policy}`. It does not
change inference routing or existing study-readiness gates. The prior bank stays
active until a separate, audited adapter/deployment task is authorized.

Create a separate derived display report from the existing audited viewer report:

```bash
python -m sircus.mjlab.prepare_expanded_report \
  --base-display /absolute/local/display-before-expansion.json \
  --expansion-study /absolute/local/expanded-search \
  --training-study /absolute/local/expanded-training \
  --output /absolute/local/display-with-expansion.json \
  --watch-seconds 1800
```

The optional watcher refreshes every five seconds for at most 1,800 seconds,
exiting early when both reports complete. Missing in-progress reports remain
explicitly pending. It neither starts training nor changes a service; connecting
the derived report to an existing viewer is a separate operation. Original reports
and the base display are read-only. Raw shot rows, trajectories, geometry chunks
and candidate arrays from the new studies are omitted from the compact payload.

The twelve old test failures are now **development** coordinates, and their
search captures and independently perturbed confirmations are displayed separately.
A new frozen comparison uses the same untouched test positions for old uniform,
old learned, expanded uniform and expanded learned selection. When action bounds
were widened, an additional old-bank/wider-bounds baseline isolates that change.
Overall capture retains every request; conditional capture counts only eligible
endpoints. No per-query outcome optimization occurs during this evaluation.
These endpoint simulations do not establish grasping, collision-free transit,
perception, physical robustness or complete table coverage.

```bash
python -m unittest sircus.mjlab.tests.test_prepare_expanded_report -v
node --test sircus/mjlab/tests/test_expanded_ui.mjs
```

### Optional IK stability and remaining-target study

The read-only `#ikStability` section consumes `stable_ik_evaluation`. It reports
the original and query-stable solver on the same controls, order/batch mask
changes, saved-branch preservation, and the ten pinned development coordinates.
Direct route captures and assisted entries stay separate. A newly qualified
setting requires the declared first and final confirmation thresholds; the UI
does not convert aggregate search captures into confirmed reliability.
Per-target screening/final summaries are derived read-only from `shots.jsonl`, including failed final groups when there are zero promotions. A final-tested source retains its own screening denominator; otherwise the highest-capture completed screening group is shown. Invalid endpoints stay in the counts, and incomplete groups remain pending.

Append the compact progress to a **copy of the current derived viewer report**
so continuous, bank and expanded-study results are retained:

```bash
python -m sircus.mjlab.prepare_stable_report \
  --base-display /absolute/local/display-before-ik.json \
  --study /absolute/local/stable-ik-study \
  --output /absolute/local/display-with-ik.json \
  --watch-seconds 1800
```

This bounded, CPU-only watcher reads reports every five seconds. It exits early
only after the study completes and `audit-stable.json` reports a passed audit.
The output cannot be the base display or any path inside the study directory.
No robot, GPU job, training run, server restart or model deployment is performed.
The original inference handlers and active policy remain unchanged.

```bash
python -m unittest sircus.mjlab.tests.test_prepare_stable_report -v
node --test sircus/mjlab/tests/test_stable_ui.mjs
```
