# TRAIN-only feasible candidate bank

This separate CPU-only preparation step implements the candidate-list proposal
without changing seed 20261122, its original 10,000-request experiment, the shared
feasibility gates or any running service. No optimizer or native simulator runs.

## Proposal contract

For each of the original 35 TRAIN bucket positions, the list contains:

- The exact final parent's deterministic action, taken from its audited final
  known-TRAIN prediction packet. No actor inference is needed.
- Sixteen draws from that parent's learned five-dimensional Gaussian, followed
  by the original tanh and physical affine action mapping.
- Sixty-four scrambled Sobol proposals spanning the original five physical
  action bounds. No local box is selected or shrunk.
- Every successful paid TRAIN action for that position, copied exactly from
  original audited evidence. This includes assisted captures under the unchanged
  Isaac scorer, and retains the clean subset too.

The fixed proposal seed is 20261124. The 35 +560 +2,240 +1,534 union contains
4,369 source entries. Exact context/five-vector deduplication preserves every
source alias and endpoint witness; it merges no near-neighbors. In this run,
all 4,369 physical requests were distinct.

The bank is **TRAIN-informed**, because known TRAIN successes are included.
Geometry alone decides admission. Neither validation nor TEST outcomes guide
the samples or filtering; no validation or TEST shot file is opened. Metadata
such as the original audit/progress/proof also contains held-out summaries and
receipt names, but those are not used for candidate selection.

## Qualification and survival

Each position uses one declared start: the actual measured home from its first
paid TRAIN request. All candidate branches use the existing coupled-arm
preflight, original tolerances and 16-sample path proxy. Historical successful
endpoint fits are independently checked with FK, then used as seeds for fresh
unchanged robust IK. They never authorize admission or release on their own.
For an exactly duplicated action, all witness aliases remain available; any
finite alternate seed attempts are explicit. No failed action is projected or
replaced, and no geometry threshold is relaxed.

The independent auditor regenerates the parent/Gaussian/Sobol proposal union,
verifies all historical action/context/witness identities and declared seed
associations, and recomputes every reached geometry, FK, limit, clearance and
path check without optimizing IK. It verifies source-specific and per-position
counts, exclusions and all successful/clean historical denominators.

The result is **1,814 admitted exact candidates at all 35 TRAIN positions**.
All **1,534 historical successful TRAIN shots**, including all **485 clean**,
remain admitted. None is excluded by generation or the feasibility layer.
See [counts and reasons](../../baselines/isaac-feasible-candidates-20261109/README.md).

Passing preparation does not reproduce a ball flight, settled release pose,
rectangle twist or successful capture. Board endpoint IK constrains center and
normal, with in-plane spin free; native settling/corrections remain separate.
The common start can differ from an original episode's measured start. A finite
bank cannot guarantee retaining every undiscovered good shot. Failed numerical
IK is unresolved, not proof that a bucket position is impossible.

## Components and reproduction

- `candidate_train_evidence.py`: standard-library, read-only collection from
  original files; verifies all 289 paid TRAIN packet hashes and 10,000 paid
  request slots. Exports success geometry, recorded starts, fits and provenance,
  excluding trajectories/media. Does not open evaluation packets.
- `feasible_candidates.py`: deterministic proposal union and bounded CPU
  qualification. The successful demonstrations influence inclusion, not gates.
- `feasible_candidates_audit.py`: separate numerical/provenance verification.
- `candidate_bank.py`: binds the final model/audit/geometry, reads the one final
  known-TRAIN prediction packet and produces the bank, audit, scalar summary
  and complete successful-shot retention CSV in a new directory.

The parent's Gaussian scales are exported by CPU-only `load_checkpoint` from
the original producer source, with its existing architecture/fingerprint/hash
guards. The export reads `logstd.clamp(-3,1).exp()`, makes zero actor inference
calls and restores no optimizer. The exact mean actions and latents come from
the original final known-TRAIN prediction packet, not CPU approximations to
the original GPU predictions. All checkpoint bytes remain unchanged.

With the bound extraction and scale export in ignored storage:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES='' python sircus/isaac/candidate_bank.py \
  --train-evidence /path/to/train-successes.json \
  --parent-scale /path/to/parent-scale.json \
  --benchmark-metadata /path/to/original-metadata-mirror \
  --published baselines/isaac-sweep-ppo-10k-20261106/final-comparison.json \
  --assets /path/to/original-assets \
  --output /path/to/new-candidate-bank --workers 4
```

The output directory must be new and outside the frozen benchmark. Full plan,
fits, decisions, traces and runtime inputs stay outside Git. Only scalar counts,
hashes and documentation are committed. The results do not change the actor's
sampling distribution or attach old continuous PPO probabilities to the bank.
Board-goal inputs, training integration, native trials and policy promotion are
outside this preparation task.
