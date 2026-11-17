# Reuse planning workers and return a checked shot sooner

The preceding [30-second comparison](PLANNING_BUDGET_RESULTS.md) found
75/100 captures for the learned selector and 69/100 for the physics planner.
Both had a median decision time of 30 seconds. This experiment measures where
that time goes and whether worker reuse and an earlier stopping rule help.
It does not train another model.

## Three versions of the same learned selector

| Version | Worker | When it returns |
| --- | --- | --- |
| `cold_30` | New CPU process for each request | Existing candidate/batch rule, at most 30 decision seconds |
| `warm_30` | Retains a clean initialized CPU process | Same rule and decision limit |
| `warm_early` | Retains a clean initialized CPU process | First fully checked candidate with predicted margin at least 5 mm, otherwise the same rule and limit |

The early margin is the minimum predicted clearance at the arms and bucket
aperture; a table-assisted route also includes clearance from the table edge.
It is a ballistic prediction, not an observed capture or a guarantee. Every
candidate still needs the same completed IK, pose and path checks. Native
simulation determines whether the selected shot captures.

Workers retain initialized geometry and model resources. Query-specific pose,
path and prediction caches clear between requests. Each method has its own
proof directory, and all 50 bucket coordinates are distinct. A deadline,
exception or failed snapshot verification destroys only that owned worker;
a later request initializes a new one. An incomplete worker never returns
to the reusable pool.

## Frozen test

- 50 new bucket positions, stratified across usable interior, edges and near
  fixed bases. Exclude inventoried history, all 400 neural-study positions,
  all 100 preceding time-budget positions and other new positions by 5 mm.
- Layout seed `2026113002`; processing-order seed `2026113003`.
- Three low-priority CPU lanes, one per method. All methods process the same
  shuffled position order; finish all three requests before the next position.
- Freeze all 150 choices before native testing. One drop at most per method
  per position, at 64 substeps, plus nine controls before and nine after.
- Keep candidate libraries, weights, flight ranker, geometry, arm checks,
  scoring and rebound parameters fixed. No native outcome informs selection.
- Count captures over all 50 positions, including requests with no checked
  shot before the deadline. Preserve every miss and invalid-physics result.

Capture means a board rebound, whole-ball entry and one continuous second
inside the bucket. Rim and table assistance count; robot assistance is
reported separately. Inputs remain simulator geometry. Native drops hold the
settled arm endpoints; this does not validate perception or moving-arm transit.

## Measure the work, including work interrupted by the deadline

Exclusive decision-clock intervals cover candidate generation, learned
ordering, arm setup checks, physics ranking, evidence publication and other
work. Arm setup includes IK and collision/path verification together; this
study does not attribute those internal parts separately. The parent closes
an unfinished interval at the cutoff, so slow interrupted work is not omitted.

Report decision time, startup, stopping overhead, receipt verification and the
measured planning request from worker launch/reuse through verified return.
The last metric excludes the caller's result-file write and HTTP/browser work;
it is not total UI latency. Report mean, median, 90th percentile and maximum, worker
reuse/reinitialization counts, setup-ready rate, first-drop capture rate and
paired gains/losses for each comparison.

## Run and verify

`python -m sircus.mjlab.fast_planning_study --help` lists the required archived
inputs, including the preceding `--budget-study` and `--budget-audit`.
Use a fresh output directory and run these stages in order:

1. `--stage initialize`: verify lineage, freeze inputs, positions, rules and
   computational source hashes.
2. `--stage prepare`: profile the three CPU methods and freeze complete choices.
3. `--stage run`: acquire the cooperative 4090 lease, run matched controls,
   alternate the frozen method batches and run controls again.
4. `--stage finalize`: save the comparison and immutable evidence inventory.
5. `--stage audit --audit-output <fresh external path>`: independently check
   deadlines, worker lifecycle, earliest eligible return, pose/path proofs,
   exact frozen commands, native arrays, one-second retention and accounting.

The additive UI panel is `#fastPlanningStudy`. It withholds final captures
until completion and a passed audit. This experiment does not change the live
planner default or restart its service. Native work is limited to one leased
4090 GPU-hour and waits at most 12 hours for that lease. The 4070 and Isaac
setup remain untouched. Raw observations, generated checkpoints, runtime
settings and logs stay outside Git.

## Storage interruption and complete retry

The first attempt stopped during CPU preparation when the shared data disk
fell below the existing 8 GiB reserve. Its last progress report showed 132
plans; 133 preparation files had been saved. It released no balls.
Its evidence is preserved unchanged. A fresh full comparison uses the system
disk, with the same frozen source, model inputs, seeds, methods and deadline.
Replanning all 150 requests preserves each method's uninterrupted worker
lifecycle; partial measurements from the interrupted run are not pooled into
the final comparison. The retry receipt records the original report and layout
hashes. No other workload or evidence was removed, and the storage reserve
was not lowered.
