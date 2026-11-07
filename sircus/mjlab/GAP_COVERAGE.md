# Two-slot gap-coverage comparison

The [residual diagnosis](FLIGHT_RESIDUAL_DIAGNOSIS.md) found that all sixteen
available candidates had negative aperture margins at each failed native layout.
This experiment tested one bounded change in candidate coverage. **Completed:
both selectors captured 24/50 (48%). Wider gaps did not improve capture in this
paired test.** This was not a new model fit.

## Completed result

| Selector | Complete captures | Clean captures | Setup rejects | Native misses |
|---|---:|---:|---:|---:|
| Original flight-consistent | 24/50 | 13 | 8 | 18 |
| Wider gap proposals | 24/50 | 11 | 8 | 18 |

There were 23 shared captures, 25 positions without capture by either method,
one gained capture and one lost capture. Selected actions changed at 19/50
positions. Case07 gained capture; case38 lost it. Replacing two proposals can
also change which of the other fourteen proposals wins; this is not a pure
gap-only intervention on every selected shot. One shot per method/layout does
not establish repeatability, equivalence or full-table coverage.

All **112 requested slots / 96 native releases** completed. Both control gates
passed (six positive captures and six negative misses total), with zero invalid
physics. The separate CPU v2 audit passed **1,441,075 checks**. No new native
shots were needed for the audit correction. [Curated scalar results and hashes](../../baselines/mjlab-gap-coverage-20261107/README.md).

The original v1 audit stopped on case19: a 0.0000021046-degree float32
reconstruction difference exceeded a mixed-unit absolute tolerance. V2 keeps
the independent 2e-6 latent/residual bound, compares actions in normalized
units, and checks saved tanh/scaling identities at 1e-12. Original producer
source, actions, scoring, records and failed audit are preserved. Only the
auditor and publication proof changed; this is not rescoring to increase capture.

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


## Execution checkpoint

The frozen runtime subset at `fb86eea5fb19635194656694d3437380beb16339`
passed 65 focused CPU tests in Ubuntu's pinned environment. The full local
repository suite passed with optional dependency skips; the follow-up publisher
guards passed 40 focused checks and GitHub CI passed. The bounded pipeline was
started in a new namespace on November 7, 2026. CPU preparation precedes GPU
acquisition; source publishing and a running supervisor do not establish completed
shots or a performance improvement. Use the live panel for current status.

The completed comparison is now published inside `#flightResidualDiagnosis`.
The prior launch/pipeline file still records the original v1 audit failure;
the separate v2 proof and publication checkpoint establish successful recovery.
The current interactive solver remains unchanged. Keep the original selector
as the baseline; this test does not justify adopting the wider-gap replacement.

## Relationship to Isaac work

The `sircus-isaacsim-isaaclab` chat is running a separate Isaac sweep-versus-PPO
benchmark: two methods, three seeds and 10,000 TRAIN requests per method/seed.
This mjlab job is a 112-request frozen-model candidate-coverage test with no
new training. They share the overall two-arm objective and exclusive 4090
scheduling, but neither run duplicates the other's data or workload. Different
engines and protocols prevent a direct cross-engine winner claim. Future mjlab
work should focus on model/flight diagnostics and matched cross-engine checks,
rather than repeat the Isaac benchmark independently.
