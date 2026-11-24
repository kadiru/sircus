# Expanded selector comparison

This experiment tests a complete proposed planner against the current live
`StalledReferencePlanner` (`warm_early` / hybrid). It is not a pure neural-network
ablation: the proposed system changes the source library, learned ordering, and
post-board contact ranking. The live planner is not changed by this experiment.

## Data and learning

1. Read authenticated historical native64 trajectories re-scored with the
   board-first version 4 contract. A capture requires the first confirmed
   rebound to be on the board and the whole ball to remain inside the bucket for
   one continuous second. Later robot, table and other contacts are allowed.
2. Deduplicate settings by bucket, requested board/release geometry, both solved
   arm configurations, release point and physics/scene identity. Retain every
   underlying outcome and its original evidence. A setting with three captures
   in four valid repeats contributes one target of **0.75**, not four examples.
   Invalid physics does not become a capture-negative label.
3. Keep all settings from a bucket coordinate in one split. The 33 usable fixed
   grid positions belong to TRAIN; deterministically hold out roughly 20% of
   other historical positions for validation. Historical TEST evidence is now
   explicitly development data; its published results are not rewritten.
4. Fit a supervised network with **44 inputs → 64 → 32 → 2 outputs**. Inputs are
   proposed geometry, direct/table *intent*, and authenticated original pre-IK
   joint hints. Outputs estimate setup feasibility and capture conditional on
   setup; their product orders candidates. There is no image, point-cloud input,
   PPO update, or prediction of time-varying joint actions here.
5. Use Adam at learning rate 0.003 and weight decay 0.0001. Compare epochs 100,
   250 and 500 using position-balanced validation binary cross-entropy; select
   the lowest validation loss. Every bucket has equal weight within each head,
   and each distinct setting within a bucket has equal weight. Repeats only
   determine empirical target fractions. The CPU fit does not allocate CUDA.

The canonical proposal for repeated settings is selected deterministically,
independent of capture outcome. Actual solved historical joints remain in the
setting identity and evidence; they are not secretly substituted into model
input hints. Successful candidate records use that same original feature
contract. A setting that captured once is a demonstrated example, not a claim
of repeatability.

## Frozen planning and native evaluation

- **Current:** original audited model, original source library and flight ranker;
  the existing hybrid arm checks, worker reuse, early return and five-second
  reference-path fallback.
- **Expanded:** new TRAIN-only successful settings and refitted selector. Up to
  16 nearby source settings are balanced across observed contact categories and
  nearby source positions. Full, zero and half XY translations provide up to 48
  proposals; the original menu fills remaining slots up to 56. Height, tilt, gap,
  heading and twist retain their source values. The neural network orders this
  finite menu. Original pre-IK hints remain hints and every accepted pose/path
  must pass fresh exact checks. Its ballistic ranker allows later arm contacts,
  but only models direct and one-table flight; arbitrary arm rebounds remain
  unmodeled.
- Both lanes use independent reusable workers, the same **30-second decision
  limit**, at most 16 initial candidate checks plus the existing bounded branch
  checks, and unchanged 5 mm path margin / 0.5° path sampling. No native drop is
  used to select a candidate. Initialization and shutdown are excluded from
  decision time and included separately in full planning request time.
- Freeze **50 fresh bucket positions**, at least 5 mm from every known historical
  position, all training/validation/library sources and the preceding 50-position
  comparison. Freeze weights, input manifests, setup proofs and selected commands
  before releasing any test ball.
- Execute **one final drop per lane per position at 64 substeps**. A position with
  no checked setup within the budget stays in the 50-position denominator.
  Three known direct captures, three table captures and three expected misses
  run before and after the test as matched controls.
- Native execution requires the shared GPU lease and an otherwise idle 4090.
  The 4070 and Isaac runtimes are not modified. Both arms are held at their checked
  endpoints during flight; this does not validate driven arm trajectories.

## Reproduction and artifact boundaries

`expanded_shot_library.py` creates a new output directory with:
`train-dataset.jsonl`, `validation-dataset.jsonl`, `candidate-sources.json`,
`all-positions.json`, `library-summary.json`, and `library-manifest.json`.
`verify_library()` checks source hashes, split isolation, fractional labels and
candidate provenance.

Run `python -m sircus.mjlab.expanded_selector_study --help` in the pinned Ubuntu
mjlab environment. Required study inputs are the original audited neural study,
the audited current planner input study (checked byte-for-byte against the
original neural model/library), the expanded library, the preceding board-first comparison, audited control
parent, runtime reference, assets, historical roots, fresh output and shared GPU
lock. Stages are `fit`, `prepare`, `run`, `finalize`, then `audit`. Fit/preparation
require `CUDA_VISIBLE_DEVICES=''`; native execution selects only the authorized
4090 UUID through the existing GPU lease protocol.

All computational source hashes are frozen with fitting. Use a new runtime and
output directory; do not edit the frozen source between stages. Preparation has an exclusive start receipt and refuses partial reruns. Native-start
receipts are created only after the GPU lease is acquired. A queue timeout with
no receipt is retryable, but partial native execution is never silently rerun.

`expanded_selector_audit.py` checks immutable inputs and weight fingerprints,
TRAIN/validation and TEST separation, deterministic menus/order, deadline and
fallback receipts, exact arm proofs, each saved native column, independent
board-first scoring, controls and the final comparison. Publish only after this
read-only audit passes. Raw traces, logs, generated configurations and models
remain outside Git; commit code and sanitized aggregates only.
