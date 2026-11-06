# Fresh-layout sweep versus PPO, three seeds

This opt-in experiment is separate from the frozen 1,600-attempt pilots. It uses
the same 35-table / 70-arm Isaac Sim farm, fixed selected camera, silver-gray
stage, driven setup controller, full articulation release gates and qualified
one-second capture definition. Only the experiment, terminal miss feedback and
learning/search methods change. No physical hardware or Viam is involved.

## Paid budgets and generalization

Each method receives exactly 10,000 TRAIN requests for each of three seeds
(20261121/22/23): **60,000 total requested training shots**. IK, path, clearance
and settling rejects consume a request; no retry or replacement is free. The
289 native training calls per run split partial 35-cell batches at exact
1,000/2,500/5,000/10,000 milestones. All methods use the same request schedule.

The new split has 35 TRAIN, 35 validation and 70 TEST positions, one TRAIN, one
validation and two TEST points in every 7×5 tabletop tile. A frozen seed samples
geometry without conditioning on IK or ball outcomes. Every point is at least
25 mm from every other split point and from the discovered published historical
bucket-coordinate pool. Exclusion provenance and all actual coordinates are
saved in the immutable experiment input. This excludes known published layouts;
it is not a claim about every unpublished experiment.

There are still exactly 35 physical simulation workcells. TRAIN uses its 35
distinct positions; evaluation installs another 35-position cohort only at the
canonical episode reset. TEST uses two cohorts. No bucket moves during a shot.

Frozen TRAIN and validation predictions are measured at 0/1,000/2,500/5,000/
10,000 samples, once per site. TEST is measured only for the final 10,000-sample
artifact, once at each of 70 new positions. Validation never selects a checkpoint
or changes the search. No held-out ball-flight query, retry, adaptation or shot
optimization occurs during prediction. Fresh endpoint IK/path/release checks
remain necessary and their cost/rejections are reported. Three seed results are
descriptive repeats, not a proof of arbitrary-table robustness.

## Methods and reward

Both methods have the same original five-dimensional continuous physical domain:
bucket-relative bearing ±180°, radius 280–1,200 mm, board height 140–600 mm,
ball-bottom drop gap 80–560 mm and board tilt 12–52° from horizontal. A common
geometry-only initial recipe places the bounce toward the shared reference,
with height 350 mm, gap 240 mm and tilt 29°. No old successful-shot labels,
trained weights, exact-context map support or qualified shot menus initialize
either method.

- Guided sweep pays for a shared prior, signed-axis proposals, local scrambled
  Sobol samples, prescheduled global exploration and progressively finer TRAIN
  best-action neighborhoods. At unseen coordinates it transfers the nearest
  TRAIN site's best latent residual relative to the shared prior. Prediction
  is frozen and performs no outcome search.
- PPO uses the existing state-only 2×64 continuous actor/critic, five geometry
  features and five tanh-Gaussian actions. Fresh seeds, Adam 0.0003, four epochs,
  clip 0.2, value coefficient 0.5, entropy coefficient 0.005 and gradient norm 1.
  Four native training batches form an update; partial buffers flush at an exact
  assessment boundary. Neither grasping nor joint trajectories are learned.

`shot_benchmark_reward.py` introduces informative misses **strictly before**
the first table/floor/robot/unattributed external-contact sample. Miss feedback
uses the unchanged scorer's closest target distance and continuous retention.
Later table-assisted proximity or retention cannot improve this feedback.
Rejected requests receive −1, missing a qualified pre-external board bounce
receives −0.8, and other misses range −0.7 to 0. Qualified captures receive 1.5
and clean captures another 0.1. The original reward/scorer files remain unchanged;
these new returns are not pooled with the older pilot returns.

## Evidence, resources and recording

Native 240 Hz ball/contact/arm/board arrays are compressed without loss. Full
setup histories use gzip JSON with compressed and uncompressed checksums. Before
each paid batch the learner artifact and RNG/counters are saved. Independent
CPU audits rescore native trajectories, prefix feedback, release/settling gates,
likelihoods, TRAIN membership and frozen prediction identities.

TRAIN wall time includes before-batch checkpoints, proposal/inference, native
home/setup/scoring, trial serialization and learner updates. It excludes initial/
milestone checkpoint saves, evaluation, progress publication, independent auditing
and video publication. Headless and filmed native episode intervals are reported
separately and include evaluation episodes. Checkpoint overhead overlaps TRAIN
time; do not add these counters together. Memory
scope distinguishes process lifetime RSS and sampled device occupancy from
isolated phase or aggregate worker peaks. Geometry and evaluation work do not
disappear into the 10,000-shot budget.

Prescheduled first, 1,000-sample, 5,000-sample and final training batches are
recorded at the user's exact saved view. Every selected chapter and outcome is
retained at normal time, with scripted setup identified. JPEG sources use
quality 82 at 1080p to reduce storage; the camera pose/FOV are unchanged. The
CPU video publisher verifies source frames and encoded media before exposing
them. No generated replacement images or ball-following camera are used.

The new GET/HEAD-only comparison reader defaults to localhost:18797. Earlier
readers remain independent. The native runner refuses to continue below 6 GiB
free and never deletes previous evidence. Existing 4070 workloads remain outside
this task. No claim of completed training follows from a prepared plan or PID;
the interface displays actual saved counts and pending/failed runs explicitly.

The bucket's reported 175 mm height interpretation, rigid attachment and fixed
material assumptions remain provisional. New-ball adaptation and perception
training are future experiments, not capabilities established by this study.

## Launch checkpoint — November 6, 2026

Native producer `df8de16709852880239b8234869bdc65c6f7e957` is frozen in a
checksum-verified source archive. Both actual 35-TRAIN/35-smoke-validation
runs exited normally and passed independent CPU audits. These diagnostic shots
are separate from the 60,000 production TRAIN budget. The PPO smoke exercised
one actual optimizer update; the sweep smoke verified paid proposal/state replay.
The actual Ubuntu CPU suite passed 77 checks without skips. Earlier Mac full
regressions and GitHub CI also passed.

The finite six-run queue started on the 4090 at 2026-11-06 19:39 UTC, beginning
with sweep seed 20261121. A separate CPU watcher verifies and encodes prescheduled
trial chapters and publishes the comparison. A Mac mirror transfers only bound
reports/media into the new GET/HEAD-only reader at
`http://127.0.0.1:18797/#training`; videos are under `#videos`. Refresh cadence is
approximately five minutes plus transfer time. The page exposes no control APIs.
The Ubuntu job continues if the Mac sleeps, but the Mac mirror/viewer must be
running for local updates. Earlier readers and the 4070 workload were untouched.

Smoke episode timing suggests roughly 2–3 days for all six runs if representative;
this is an estimate, not a completion guarantee. Native setup failures consume
paid samples and remain visible. A failed native run or audit stops subsequent
runs. Partial runs require explicit recovery rather than a silent restart with
another full sample budget. The queue and publisher have finite eight-day bounds;
each native process has a sixteen-hour bound and preserves the shared GPU lock.

Runtime roots (generated evidence remains outside Git):

- Ubuntu input/raw evidence: `/home/kadir/sircus-isaac/output/sweep-ppo-10k-20261106-v2`.
- Ubuntu published report/media: the sibling path ending in `-published`.
- Mac mirror: `<repository>/.runtime/media/shot-benchmark-20261106-v2-live`.
- Mac receipts/mirror status: `<repository>/.runtime/shot-benchmark-20261106/control`.

Source archive SHA256:
`6dcaa4199bb2df890579fe2fde245f1a48b33deae5869c70399cce4cf8712905`.
Experiment SHA256:
`77afe173a07917dd87ce43f37809a9ac6ef99beebd7e8e35f703c0e5cf1e3a26`.
Passed sweep smoke audit SHA256:
`cb668843aa252de83719b24b36f229a9a7d4ac28314bfadb4c7fca6a9a762152`.
Passed PPO smoke audit SHA256:
`4c0e3833c895a3a9200b97deb25cde793e70f9b0766ace33e839b8d014686f16`.

The full production audits and held-out results remain pending at launch. No
winning method, learned physics understanding or physical readiness is claimed.

## Publication recovery — November 6, 2026

A follow-up health check found native training still running on the 4090, while
the CPU video watcher had stopped at a legitimate all-rejected filmed chapter.
At TRAIN prefix 1,000 the scheduled partial batch had 15 requested actions,
all rejected, plus 20 inactive cells: 2,228 setup frames and no released ball.
The unchanged native auditor explicitly accepts that evidence. It was the
separate movie validator that incorrectly demanded release frames in every film.

The offline encoder now accepts setup-only footage only when the enabled mask,
all active rejection rows, explicit null physics and unchanged release/end clock
prove no ball was released. It labels the chapter accordingly. Any admitted or
unverified action still requires release evidence. Camera/frame/trial hashes and
existing known C-encoder receipts remain checked; earlier movies are retained.
A video-export failure now leaves verified curve/earlier-video publication active
and is logged separately for retry. Integrity failures in the report itself still
stop publication.

This recovery changes only the offline publisher and its tests. Execute its
reviewed files from a separate source directory with the original frozen C
modules on PYTHONPATH. Do not replace the native source archive, experiment,
running simulator/coordinator, policy, scoring contract or pinned auditor.
Observed interim first-seed sweep validation captures were 3/35 at initialization,
6/35 after 1,000 requests and 8/35 after 2,500. No production run had finished at
this check, PPO was queued, and final TEST remained unopened; no sweep/PPO winner
can be concluded. These checks do not tune the ongoing study.
