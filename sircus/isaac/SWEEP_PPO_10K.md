# Fresh-layout sweep versus PPO, three seeds

This opt-in experiment is separate from the frozen 1,600-attempt pilots. It uses
the same 35-table / 70-arm Isaac Sim farm, fixed selected camera, silver-gray
stage, driven setup controller, full articulation release gates and qualified
one-second capture definition. Only the experiment, terminal miss feedback and
learning/search methods change. No physical hardware or Viam is involved.

## Final audited comparison — November 8, 2026

All six production runs completed exactly 10,000 paid TRAIN requests each,
including every rejection, and passed the unchanged original independent
auditor. The finite recovery coordinator completed at 10:30 UTC. The final
models, all 289 aggregate curve points per run, five frozen checkpoints,
phase-scoped compute and 24 prescheduled native video receipts are preserved
in the [final additive package](../../baselines/isaac-sweep-ppo-10k-20261106/README.md).
The historical partial summaries and earlier model bytes remain unchanged.

| Final result, seeds 20261121 / 22 / 23 | Sweep | PPO |
|---|---:|---:|
| Unseen TEST captures, each /70 | 23 / 22 / 25 | 28 / 28 / 28 |
| Unseen TEST clean captures, each /70 | 15 / 12 / 18 | 10 / 7 / 14 |
| Unseen TEST rejected requests, each /70 | 19 / 29 / 20 | 18 / 17 / 42 |
| TEST mean capture ± seed sample SD | 33.3% ± 2.2 pp | 40.0% ± 0.0 pp |
| TEST mean clean capture ± seed sample SD | 21.4% ± 4.3 pp | 14.8% ± 5.0 pp |
| Final validation captures, each /35 | 9 / 11 / 14 | 11 / 12 / 11 |
| Final validation mean capture | 32.4% | 32.4% |
| Exploratory TRAIN mean capture | 7.22% | 15.09% |
| Paid TRAIN mean rejection rate | 83.99% | 65.57% |
| Mean full TRAIN wall, excluding evaluation | 4.618 h | 6.389 h |

PPO gained 5, 6 and 3 TEST captures in the paired seeds, while sweep gained
5, 5 and 4 clean captures. PPO used approximately 38.3% more mean TRAIN wall
time. This establishes a capture/clean-entry/cost tradeoff in this experiment,
not one method dominating all outcomes. All seeds share the same 35 validation
and 70 TEST positions. Seed SD is descriptive; zero SD in PPO's aggregate TEST
capture counts is not zero uncertainty or proof of equal per-position behavior.
The third PPO seed's 42/70 TEST rejections show substantial admission variation.

Exploratory TRAIN successes above include changing proposals and repeated TRAIN
positions. Separate final frozen direct predictions on the 35 known TRAIN
positions captured 20 / 20 / 21 for sweep and 13 / 13 / 14 for PPO; those are
diagnostics, not held-out generalization. Validation curves never selected a
model. Every final result uses the prespecified 10,000-request artifact, with
no held-out physics search, retry, adaptation or free replacement. Each run's
420 separate assessment requests include frozen TRAIN diagnostics, validation
and final TEST; they are outside its 10,000 paid training requests.

Full resource records separate measured TRAIN wall from episode-only headless
and rendered intervals, which include evaluation. RSS remains a process-lifetime
high-water mark and GPU occupancy a sampled device measurement, not policy-only
memory or isolated phase peaks. Geometry preparation wall time was not measured
and remains unknown. Video publication and independent audits are separate work;
no energy or monetary-cost measurement is inferred.

The original archive/experiment, source, physics, scoring, materials, rewards,
geometry, camera, splits, run order and release guards remained fixed. Privileged
geometry and scripted arm travel/loading/initialized release remain limitations;
this is neither perception-driven control nor learned joint motion or physical
validation. All scheduled video outcomes, including all-rejected setup-only
chapters, are retained. No model promotion or ball-property, perception or Cosmos
experiment follows this completed benchmark.

The following dated checkpoints are preserved as historical progress reports.

## Later sweep completion checkpoint — November 7, 2026

All three sweep runs have completed and passed their original independent
audits. The additive [later sweep evidence](../../baselines/isaac-sweep-ppo-10k-20261106/README.md#later-sweep-seeds-addendum--november-7-2026)
packages the exact seed-20261122/23 final models, all 289 aggregate learning-curve
points per run, five prescribed frozen assessments and all eight video receipts.
Earlier first-pair and second-PPO evidence remains byte-identical.

| Final metric | Sweep seed 20261122 | Sweep seed 20261123 |
|---|---:|---:|
| Paid TRAIN requests, including rejects | 10,000 | 10,000 |
| TRAIN captures / clean | 717 / 201 | 646 / 222 |
| TRAIN rejected setups | 8,355 | 8,535 |
| Frozen validation captures at 0 / 1k / 2.5k / 5k / 10k, each /35 | 3 / 8 / 7 / 9 / 11 | 3 / 9 / 10 / 12 / 14 |
| Final unseen TEST captures /70 | 22 (31.4%) | 25 (35.7%) |
| Final TEST clean captures /70 | 12 | 18 |
| Final TEST rejected setups /70 | 29 | 20 |
| Full TRAIN wall, excluding evaluation | 4.998 h | 4.571 h |
| Lifetime peak process RSS | 6,989.5 MiB | 6,962.8 MiB |
| Sampled occupied GPU memory, headless episodes | 6,221.4 MiB | 6,189.4 MiB |

The first sweep remains 23/70, with 15 clean captures. All seeds use the same
70 prescribed TEST coordinates; these are repeated training seeds, not 210
independent positions. The two completed PPO seeds remain 28/70 each, with
10 and 7 clean. Final PPO seed 20261123 is still training on the 4090, so the
paired three-seed comparison is incomplete. Validation never selects a model;
each result uses the prescribed final 10,000-request artifact without held-out
physics search, retries or adaptation.

Five audited runs account for 50,000 completed paid TRAIN requests. The final
PPO run retains its original 10,000-request budget, including its concurrent
partial progress; 10,000 is not a count of requests still unspent. Live counts,
scoped compute measurements and video chapters remain on localhost:18797.
The native source/archive, experiment, physics, rewards, release guards,
materials, geometry, camera and splits are unchanged. No native job was restarted
or paid shot repeated to create this repository package.

The 9,975-TRAIN third-sweep movie is 81.97 seconds and includes four releases,
four board bounces and four captures elsewhere in the farm; the foreground
request was rejected by board-arm IK. Seed 20261122's 5k chapter has five
rejected active requests and no release. Both remain included with their
original labels. These are prescheduled native TRAIN chapters with scripted
travel and initialized release, not held-out scores or learned motor control.

Resource scopes below apply unchanged: episode-only phases include evaluation,
TRAIN wall excludes it, and process/device memory is not policy-only memory.
New-ball properties, perception and Cosmos remain outside this benchmark.

## Second PPO completion checkpoint — November 7, 2026

Both PPO seeds below completed exactly 10,000 paid TRAIN requests, 73 updates
and 420 separate frozen evaluation requests, and passed the unchanged original
independent audit in the producer runtime. Each uses its prescribed final
checkpoint; validation did not select it and TEST used no shot search or retries.

| Frozen evaluation | PPO seed 20261121 | PPO seed 20261122 |
|---|---:|---:|
| Validation captures at 0 / 1k / 2.5k / 5k / 10k TRAIN, each /35 | 3 / 5 / 7 / 10 / 11 | 3 / 4 / 5 / 8 / 12 |
| Final unseen TEST captures /70 | 28 (40%) | 28 (40%) |
| Final TEST clean captures /70 | 10 | 7 |
| Final TEST rejected setups /70 | 18 | 17 |
| Paid TRAIN rejected setups /10,000 | 6,549 | 6,344 |
| Full TRAIN wall, excluding evaluation | 5.985 h | 6.295 h |
| Native elapsed, including evaluation/recording | 6.287 h | 6.560 h |
| Lifetime peak process RSS | 7,408.9 MiB | 7,448.7 MiB |
| Sampled occupied GPU memory, headless episodes | 6,216.1 MiB | 6,217.4 MiB |

The policies share 26 successful TEST positions, with two additional captures
unique to each seed. Both runs use the same 70 TEST coordinates; these are two
training-seed repeats, not 140
independent layouts. Their descriptive TEST capture-rate seed SD is zero; that
does not establish certainty or broad robustness. Clean captures differ. The
first sweep seed captured 23/70, with 15 clean; two sweep seeds and one PPO seed
still remain, so the paired three-seed benchmark is not complete.

Episode-only headless/render intervals include evaluation and exclude outside
learner/proposal/checkpoint calls; they are not additive to full TRAIN wall.
RSS is a process lifetime high-water mark and GPU usage is sampled whole-device
occupancy, not isolated phase peaks or policy memory. Full scoped measurements,
aggregate curves and exact final models are in the
[compact evidence package](../../baselines/isaac-sweep-ppo-10k-20261106/README.md).

Second-PPO original audit SHA256:
`cec9cff0a6de59b07dd35de3342ee00bc4106d4801edc0237c570e006aedc1c8`.
Final checkpoint SHA256:
`57d7658e9ed2f72ad09cf4e8523ab497ccbe0708364f1b8ec4a4c5c1aef1fe9f`.

All four prescheduled second-PPO TRAIN movie chapters are published. The final
9,975-request chapter is 83.37 seconds, with seven actual releases, seven board
bounces and four captures. Its foreground request was rejected by Arm10 IK;
the actual drops occur elsewhere in the farm. Playback is native simulation
with scripted joint travel and initialized release, not learned motor control
or held-out evaluation. The original camera and silver-gray look are unchanged.

At the 16:59 UTC check, the finite queue had advanced to sweep seed 20261122
on the 4090. No native job was restarted, no paid sample repeated and no
experiment/source/scoring/geometry/split/budget changed. Three of six runs are
complete and audited. The existing reader/mirror and the 4070 workload remain
untouched. Current progress continues on the read-only localhost:18797 interface.

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

## First completed sweep and audit-runtime recovery — November 6, 2026

Sweep seed 20261121 completed all 10,000 paid TRAIN requests. It captured
803 shots (240 clean), rejected 8,306 setups and released 1,694 balls. The
prescribed final checkpoint predicts directly at unknown coordinates; no
held-out shot search or validation-based checkpoint selection was used.

| Paid TRAIN requests | Frozen validation captures / 35 | Clean | Rejected |
|---|---:|---:|---:|
| 0 | 3 | 0 | 29 |
| 1,000 | 6 | 4 | 13 |
| 2,500 | 8 | 5 | 13 |
| 5,000 | 11 | 6 | 11 |
| 10,000 | 9 | 6 | 13 |

The final, separately opened TEST captured **23/70 (32.9%)**, with 15 clean
captures and 19 rejected setups. Frozen final TRAIN was 20/35. This is one
sweep seed, not an across-method or three-seed conclusion. The 5,000-request
validation peak does not replace the final artifact. Full TRAIN method wall
was 15,428.5 seconds; evaluation and the later audit are separate costs.

The original coordinator stopped before PPO because its host CPU audit used
NumPy 1.26.4, while the native producer used 1.26.0. An inverse-transform
rounding difference of 2.78e-17 changed a byte-derived proposal ID even though
the decoded physical action was identical. The unchanged original auditor
passed in the exact producer runtime with CUDA disabled: 301 trials, 10,000
TRAIN plus 420 evaluation requests, 1,985 rescored releases and 294 checkpoint
artifacts. No comparison, numerical tolerance, source hash or scoring guard
was weakened. The audit took 314.42 seconds on one CPU thread.

Passed original audit SHA256:
`e126486453c9b50886dcf2c910d266cdc2cb44019934b9f43475d296a7ced1cc`.
Its separate launch receipt SHA256:
`f058aab4a1ce8358907dfd9b68b45e733c13565ae83d7e7f2ee97040a6f3c369`.
Raw evidence, logs, runtime provenance and videos remain outside Git.

`shot_benchmark_recover.py` is an external recovery coordinator, not a native
producer patch. It verifies the original archive and completed evidence,
adopts the passed audit byte-for-byte, preserves the failed coordinator/log,
and permits only the five untouched paths in the original protocol order.
Original shared GPU locking, sixteen-hour native bounds, six-GiB storage floor,
eight-day experiment deadline, samples, physics, rewards and layouts remain
unchanged. Any partial next run or failed native/audit stops recovery; it never
repeats paid trials. Future audits use the original wrapper in the existing
producer container, CPU-only. It performs no container or service restart.

The report now displays the saved coordinator state and a separately bound
independent-audit status for each run. A stopped queue cannot appear healthy
merely because a native result says completed. Recovery status is accepted only
with its original-source/experiment/failed-status/adopted-audit receipt bindings.
These are saved status receipts, not process-liveness checks; an operator
status investigation must still inspect actual processes. The static page and
CPU publication can be updated independently of the GET-only reader and native
simulation. PPO results remain pending until actually observed; the live
interface, not this historical checkpoint, shows the current attempt count.

Recovery was activated from committed external source
`e835f652b5991e270c3cd30a345db7d49fa73b0b` (archive SHA256
`93556582c323c81a66b3d69f6859bfb7bb95cf3bb51292385d7f05bf1c0dac80`).
All 113 focused checks passed on Ubuntu CPU without skips, followed by a
read-only adoption preflight against the actual completed evidence. The native
PPO seed 20261121 process was verified on the 4090, initially performing frozen
evaluation; its held-out performance is not established by startup. The original
4070 process remained present and unchanged. The original failed coordinator
files remain intact, with a new separately bound recovery status/launch receipt.
No sweep TRAIN requests were repeated.

Only the separate finite CPU publisher was replaced to load the new report
code. The existing Mac mirror and GET/HEAD-only reader kept their original
processes; HTML/JS were updated with a backup and checksum manifest, preserving
the frozen reach-audit overlay. All four sweep chapters were present and the
latest 81.33-second native chapter played successfully in the browser. The code
checkpoint passed GitHub's mock/simulation workflow. Draft PR #81 is stacked on
#80; neither was merged. Local deployment receipts are under the ignored
`<repository>/.runtime/shot-benchmark-20261106/control/status-recovery-e835f65/`.
