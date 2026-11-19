# Matched sweep versus residual PPO benchmark

This compares a guided-sweep shot library with a continuous residual RL policy
for the **current fixed ball**. It uses the corrected native MuJoCo/mjlab
64-substep physics, independent scalar capture scorer, strict endpoints and
sampled common-home/local path admission. Historical studies remain unchanged.
The experiment uses privileged scene geometry and prescribed held arm endpoints.

## Completed frozen v1 result

All six workers completed. The [checksummed compact evidence bundle](../../baselines/mjlab-sweep-rl-benchmark-20261103/README.md)
contains the frozen contract/report, independent audit, split/prior definitions,
per-seed and aggregate CSVs, source manifests and all six selected artifacts.
The full audit passed **902,252 checks**, covering **7,440 requests / 3,738 native
releases**; the separately quarantined 50 engineering controls are not included.

| Method | Spent TRAIN requests / native releases | Selected budgets, seeds 20261311/12/13 | Frozen formal test successes / requests |
|---|---:|---|---:|
| Guided sweep | 3,072 / 1,385 | 0 / 0 / 0 | 67/216 (31.0%) |
| Residual PPO | 3,072 / 1,641 | 256 / 1,024 / 0 | 58/216 (26.9%) |

Every method/seed spent 1,024 requests, including rejects, regardless of the
earlier checkpoint selected. Native totals by split are 3,026 TRAIN, 496
validation and 216 test. Each frozen test worker released 36/72 natively; its 36
rejected requests remain failures. All methods used the same current ball and
17-recipe historical warm start. Sweep selection retained its zero-budget prior
in all seeds; these results do not demonstrate improved performance from added
sweep samples. All three selected sweep archives have empty learned libraries
and the same zero-residual prior predictor; their different test counts reflect
native execution variability, not different learned solutions. PPO selected three
different budgets. This pilot does not demonstrate a consistent learning gain,
and does not establish an ordinary capture/adaptation ranking under the
contact-sensitive formal metric.

The first full-audit attempt failed in summary assembly because a native-count
keyword was supplied twice. Only the auditor summary was corrected in a separate
source snapshot, then the full audit was rerun and passed. Executed training and
evaluation sources, evidence bytes, formal labels, selections and original
manifests were unchanged. The new result server binds the corrected passed audit;
it does not erase the failed original audit or restart its earlier result server.

## Shared starting information and action family

Both methods receive the same 17 audited recipes from the completed
[35-position study](../../baselines/mjlab-grid35-expanded-20261103/README.md).
That is shared historical knowledge, not free cold-start training. The latest
two studies used 4,234 native releases / 4,243 requests; earlier contact tuning,
source searches and human design also contributed, and their full cost is not
quantified. The contract records that limitation.

For a query bucket, the source recipe is the nearest original training bucket
in XY, breaking ties by source ID. Its board XYZ, tilt and gap stay the initial
values; the board heading faces the query bucket. Both methods choose continuous
adjustments within the same family:

| Parameter | Adjustment from shared source |
|---|---:|
| Board X / Y | ±40 mm each |
| Board height | ±50 mm |
| Tilt from horizontal | ±8 degrees |
| Ball-bottom gap above board center | ±120 mm |

Original global action bounds remain mandatory. Nothing clips or replaces a
failed physical pose with a successful one. Release XY equals board-center XY;
release-ball-center Z is board Z + gap + radius. This is a restricted continuous
shot task, not unrestricted joint control or proof of all-table coverage.

Nine fixed-normalized geometric inputs encode bucket XY, source bucket XY and
the five source parameters. There is no camera input, material oracle, ball
identification, hidden test reward or taught grasp in this comparison.

## Equal budgets and frozen layout pools

The canonical pilot uses three seeds, each with **1,024 new requested training
decisions per method**. Bounds, IK, home and local-path rejects consume decisions.
No extra candidate resampling is free. Actual native releases and geometry work
are reported separately.

- TRAIN: the same 35 development grid positions.
- Validation: 12 fresh seeded coordinates, used only at milestones
  **0 / 256 / 512 / 1,024**.
- Test: 24 other fresh coordinates, revealed to inference only after selecting
  and freezing an artifact.
- Every evaluation predicts one deterministic base action per layout, then
  requests one nominal and two paired Gaussian placement perturbations. There
  is no best-of-three action selection or per-layout search.
- Perturbations are shared across methods/seeds/milestones: board X/Y σ1 mm,
  height/gap σ2 mm, tilt σ0.35 degrees. Global/path rejects remain failures.

Three seeds × two methods give **6,144 training + 864 validation + 432 test =
7,440 requested trials**, excluding separately logged engineering controls.
Splits, seed schedules, priors, packages, sources and budgets are fingerprinted
before outcomes. Current 35-position results are development data.

## Methods

**Guided sweep:** per training context, request the source action, ten signed
axis offsets, ten seeded joint space-filling proposals, then bounded latent
refinements around that context's best TRAIN result. Every candidate consumes a
training decision. Candidate selection ranks valid capture first, then the
unchanged shaped reward, retaining earlier ties.

At validation/test, its frozen library transfers the normalized residual from
the nearest TRAIN context to the query's fixed shared source anchor. It performs
no native query, search, retry or outcome-based refinement there. Therefore this
initial benchmark compares sweep-built library transfer with policy inference.
An online per-query native search would be an additional, separately charged
search reference; it must not masquerade as a library prediction.

**Residual PPO:** a 9→64→64→5 Gaussian actor and 9→64→64→1 critic. The actor starts
at zero residual, matching the sweep's initial source prediction. Each episode
is one terminal shot. Actual native rewards update the actor and critic with
CPU Adam, learning rate 0.0003, six PPO epochs per 32-request batch, ratio clip 0.2,
value weight 0.5, entropy weight 0.001 and gradient clipping 1.0. There is no GAE or
bootstrapping. Latent standard deviations start at exp(−1.5), with log-std bounds
−3 to −0.7. The tanh residual transform is fixed and its values are not projected
onto a candidate menu.

The existing shaped reward remains distinct from success. Invalid decisions get
−1. Proximity/board/retention shaping can guide learning, but capture still requires
board rebound → whole-ball entry → one continuous second retained. Rim assistance
counts; table, floor, other robot or unattributed contacts disqualify the route.

Both methods select a milestone by validation captures, then mean shaped reward,
then earliest budget. The selected model/archive is frozen before test. Live
learning curves show validation data; final test counts appear separately after
all runs complete. Neither higher reward nor more training proves an advantage.

## Geometry and native verification

Each method/seed owns fresh caches. The adapter first checks the query-specific
source anchor and its full sampled common-home path. Each requested adjustment
then passes fresh strict endpoint checks and a sampled local connection through
that anchor. An unproven source connection is a failed request, with no fallback
teacher search. Geometry-only setup checks incur CPU/time cost but provide no
free ball-flight labels.

Before the benchmark, a separate TRAIN-only smoke checks all 17 historical source
recipes, a deliberately invalid request, and 32 discarded PPO samples. The actor
must change from finite, nonconstant native rewards; behavior likelihoods are
checked. Neither that model nor those outcomes enter benchmark training.
Independent CPU controls auditing precedes the long run.

Every actual release saves native positions, velocities and categorized forces.
The unchanged scalar scorer checks the batched label. The independent auditor
checks the raw→canonical row association, samples/splits, source/proof hashes,
sweep dependencies, CPU optimizer/likelihood replay, checkpoint selection and
frozen inference. Path evidence identities are checked; an audit does not claim
a separate continuous collision proof or independently rerun all geometry.

## Numerical limitation in the frozen v1 metric

Completed validation traces exposed sensitivity to wall contact: six shots that
lost capture at the 256-request sweep checkpoint entered early and finished
inside, but brief nominal wall overlaps reset the continuous-retention clock.
The criterion permits only 1 µm wall overlap, while the floor permits 1 mm.
The nominal 64-facet/taper distance formula is consistent between scalar and
batched scoring; this is not evidence of a transcription mismatch.

[MuJoCo's solver documentation](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters)
explains that soft contact can produce penetration. Such overlap alone does not
establish escape from the bucket. Compiled collision geometry and solver contact
distances still need independent calibration to separate their contributions.

The completed protocol remains frozen. Its labels are reproducible **strict
geometric one-second containment**, but should not yet rank ordinary capture
performance or material-adaptation speed. No tolerance or historical label is
changed to recover these failures. A corrected benchmark needs a separately
frozen control set (resting/contacting balls, retained impacts, rim assistance,
real partial exits/re-entry and outside negatives), then a versioned scorer,
reward and contract with fresh evaluation. Diagnostic trace analysis is separate
from method selection and does not use final-test outcomes.

A [validation-only diagnostic with trace hashes](../../baselines/mjlab-sweep-rl-numerics-20261103/README.md)
reproduces six selected failures without changing their scores or using test data.

## Resource measurements

- Training/setup wall time and current worker CPU time exclude validation/test.
  These are instrumented implementation costs, including evidence I/O, not
  isolated algorithm timings. CPU time for the external NVIDIA telemetry
  subprocess is excluded.
- The recorded `geometry_s` subtotal covers training endpoint/home checks.
  Sampled common-home and local-path checks contribute to total CPU/wall time,
  but are not included in that subtotal. Their `path_s` timings remain in the
  per-batch evidence; this frozen run does not aggregate a separate path subtotal.
- Native GPU-phase elapsed wall time is recorded separately from CPU geometry.
  It is not a measurement of active kernel utilization.
- Total exclusive 4090 worker reservation includes evaluation and lazy native
  initialization. Queue waiting is outside the reservation.
- Peak RSS and GPU memory are lifetime process measurements as of each snapshot,
  including the simulator and evaluation, rather than policy-only memory.
  Memory units are **MiB**. NVIDIA process VRAM is sampled once per second, so its
  recorded maximum is a sampled peak; PyTorch allocated/reserved peaks are separate.
- Native initialization time is recorded separately; it may first occur in the
  zero-budget validation stage.
- Algorithm checkpoints and raw audit-data storage are distinct costs. The saved
  PPO artifact includes actor, critic and optimizer state; its file size is not
  an actor-only deployment footprint.
- `query_s` is the total cohort prediction/preparation loop: source-anchor lookup,
  observation generation, one base prediction per layout, metadata and construction
  of all three perturbed requests. Final testing has 24 base layouts / 72 requests.
  Artifact loading occurs before this timer; evidence-file writing and subsequent
  IK/path/native admission and simulation occur afterward. It is neither isolated
  per-query inference latency nor end-to-end shot latency. The compact CSV's
  `frozen_query_mean_s_per_layout` is this cohort total divided by 24, not a
  separately instrumented actor-call timing.

Completed-worker costs below are three-seed mean ± sample SD, read from
[`aggregate-final.csv`](../../baselines/mjlab-sweep-rl-benchmark-20261103/aggregate-final.csv).
They are descriptive measurements on the shared host, not hardware-independent
algorithm costs. Reservation includes evaluation; setup/TRAIN wall and CPU do not.

| Per-worker measurement | Guided sweep | Residual PPO |
|---|---:|---:|
| Setup + TRAIN wall (s) | 2,097.0 ± 62.2 | 2,099.6 ± 56.9 |
| Setup + TRAIN process CPU (s) | 3,111.6 ± 94.1 | 3,123.6 ± 89.1 |
| Full exclusive GPU reservation wall (s) | 3,134.6 ± 66.8 | 3,129.5 ± 75.0 |
| Lifetime peak RSS (MiB) | 1,628.4 ± 5.3 | 1,738.4 ± 0.6 |
| Lifetime sampled process VRAM peak (MiB) | 762.0 ± 0.0 | 762.0 ± 0.0 |

The three sweep artifacts are 8,504 bytes each; the PPO artifacts are 135,965,
135,965 and 45,825 bytes (seed order above). PPO's zero-budget checkpoint has no
populated optimizer history. These saved-file sizes are not actor-only inference
footprints and exclude raw traces, other milestones and per-update histories.

Workers run sequentially with alternating method order across seeds. Every worker
uses the same batch-32 native environment, fresh process/caches and two CPU threads.
Only the 4090 UUID is exposed. The existing 4070 workload and other services stay
untouched; absolute wall-time results reflect the shared host's conditions.

## Running and viewing

Use a frozen source snapshot and the existing pinned environment. Controls and
benchmark outputs must be new, unique directories on the output filesystem.

1. Run `python -m sircus.mjlab.benchmark_smoke` with `--reference`, `--assets`,
   `--gpu-lock`, `--prior-bundle` and `--output`.
2. Hide GPUs and run `python -m sircus.mjlab.benchmark_verify --controls-only`
   with `--study`, `--reference`, `--assets`, `--repository` and external `--output`.
3. Run `python -m sircus.mjlab.benchmark_study` with the shared reference/assets/lock,
   `--prior-bundle`, `--preflight-report` and new `--output`. `--prepare-only`
   freezes inputs without training; an identical prepared study uses `--resume`.
4. The controller runs six method workers, freezes the final report, then performs
   its independent CPU audit. Audit state stays external, preserving report bytes.
5. `python -m sircus.mjlab.benchmark_server` extends the grid-view launch arguments
   with `--benchmark-dir` and optional external `--benchmark-audit`, default
   loopback port 18788 and `#sweepRlBenchmark`.

The new panel shows capture-rate curves, training time, per-seed resources,
counts, hashes and CSV. Its ordinary solver route stays unchanged. A missing
measurement remains unknown. No physical dashboard, Viam/cloud configuration,
audio, camera, soft ball or deformable gripper is commissioned here.

The completed result view is a **new** server at
[`http://127.0.0.1:18789/#sweepRlBenchmark`](http://127.0.0.1:18789/#sweepRlBenchmark),
bound to the passed corrected audit. The original 18788 server and its failed
original-audit binding were left untouched, along with all earlier services.

The first point appears after setup and initial validation finish. Before then,
the last published worker phase can lag CPU anchor/path preparation even after
the GPU lock is acquired. The interface therefore says it is awaiting the first
completed measurements rather than inferring that the GPU is still unavailable.
