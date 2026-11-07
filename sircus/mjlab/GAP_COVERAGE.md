# Two-slot gap-coverage comparison

The [residual diagnosis](FLIGHT_RESIDUAL_DIAGNOSIS.md) found that all sixteen
available candidates had negative aperture margins at each failed native layout.
This experiment tests one bounded change in candidate coverage. **No new results
are published yet; this is not a new model fit.**

## The single change

Compare the unchanged consistent-flight selector with a copy that replaces only
candidate slots08/09. Previously those slots used the imitation board pose with
the imitation drop gap minus/plus 36 mm. They now use that same board pose with
the geometry anchor's gap minus/plus 120 mm, the existing residual-domain endpoints.
For example, an anchor gap of 135.341 mm yields proposed endpoint gaps of
15.341 and 255.341 mm. These are proposals, not proven successful shots.

The other fourteen candidates, sixteen-candidate bank size, frozen ensemble
weights, velocity/aperture ranking, maximum four geometry checks, native64
engine and calibrated complete-route score are unchanged. Nonpositive or
unsupported proposals stay literal; no clamping, hidden replacement, hard
negative-margin veto or post-outcome retry makes them appear feasible. Geometry
admission remains authoritative before any native release.

Wider endpoints can lose useful intermediate settings. Improvement is not
assumed. The experiment does not address all IK failures or add a robot-flight
collision predictor.

## Frozen budget and evaluation

- **50 fresh bucket layouts**, seed `20261801`, X 100–1240 mm and Y −650–250 mm;
  each at least 10 mm from declared ancestral/training/evaluation/development
  layouts and other new layouts. Previous 30 TEST positions are excluded.
- Two selectors × one requested shot per layout = 100 TEST slots. Six positive
  and six negative controls split before/after give **112 total requests**.
  Setup rejects remain in the denominator; native misses and invalid physics
  remain separate.
- Both banks, every prediction, geometry check, selected action and provenance
  are frozen before acquiring the GPU or observing a new native outcome.
  There is no fitting, native flight search, outcome-based substitution or retry.
- Compare complete captures, clean captures, admission, native misses and paired
  gains/losses. A complete capture still requires board bounce, whole-ball entry,
  continuous one-second retention and no disallowed external contact before
  capture. Rim-assisted captures may count; partial exits reset retention.
- A separate CPU audit must bind and validate the completed evidence before the
  UI reports verified results. Single nominal observations do not prove repeatability.

## Runtime boundary

`gap_coverage_pipeline.py` prepares on CPU, then waits for the existing exclusive
4090 lock and any foreign workload. Only the 4090 UUID is exposed to its worker.
The queue expires after 24 hours, acquired work is capped at one hour, and at
least 8 GiB of storage must remain. It never signals a foreign process. The
independent auditor runs without CUDA after the worker exits.

The gap-coverage section inside `#flightResidualDiagnosis` is separate from
verified historical results and the active interactive solver. Source synchronization does not
restart a service, activate a model, change cloud configuration or deploy
hardware. The 4070 and Isaac setup stay untouched; no 128-substep or noise
experiment is part of this comparison.

Entry points: `gap_coverage_candidates.py`, `gap_coverage_study.py`,
`gap_coverage_verify.py`, `gap_coverage_pipeline.py` and
`prepare_gap_coverage_report.py`. The pipeline requires explicit paths for the
authenticated parent, original ensemble and consistent-flight studies/audits,
frozen native source/assets, existing GPU lock, existing display and a new output
namespace. Raw records, checkpoints copied for execution, logs and runtime state
remain outside Git.
