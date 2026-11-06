# Completed diverse-pose pilot: reward diagnosis

This is an aggregate, training-only analysis of the completed native Isaac
study. It is not a new experiment, a reward change or an evaluation-driven
optimizer. The [study protocol and comparison](../../../sircus/isaac/DIVERSE_SETTLED_PILOT.md)
keep the earlier broad/local results frozen.

The independent audit recomputed all 1,600 saved training returns from the
unchanged reward function: zero mismatches, maximum float32 error 5.94e-8.

| Training outcome | Attempts | Reward behavior |
|---|---:|---|
| Rejected requests | 736 | Always -1 |
| Released external-contact misses | 777 | Always -0.5 |
| Other released misses | 30 | 0.15711–0.22504 |
| Qualified captures | 57 | 1.52686–1.73631 |

The first two groups are 94.56% of the attempted training data. External-contact
misses span 2.02–366.72 mm closest-opening distance, but the reward returns early
before using the distance and retention terms. This limits useful feedback for
most near/far misses. It is a supported mechanism diagnosis, not a controlled
causal proof that changing the reward will improve the policy.

Legacy retained-ball flags total 125, including 68 disallowed external routes.
The qualified capture count is **57**, including four clean entries. Captures
still require a board bounce, whole-ball entry and one continuous second inside,
without a detected external route contact. Rim-assisted captures count.

The final frozen policy scores 24/200 on training layouts, 0/40 validation and
8/40 test. The test site's entries become clean, but the held-out capture rate
does not improve. Eight identical nominal repeats per site do not establish
unseen-layout or perturbation robustness. No policy promotion is supported.

Recommended next experiment: version more informative miss feedback while
preserving the hard capture definition, then run another bounded comparison.
Do not change the completed returns or count table-assisted legacy outcomes as
new task successes.

The byte-identical [aggregate receipt](reward-diagnosis.json) contains counts,
return ranges, source bindings and an ordered trial-manifest digest. It excludes
raw observations, joint/body/ball state arrays and evaluation samples.

- Native producer: `40dd24ffdfeec0cd521fdfbf409c31c68c69a2d7`.
- Result SHA256: `433ed3496ada64b0d6edf626e4d796c25285441b014b649d1d1fdccc34f236f9`.
- Checkpoint audit SHA256: `e6df3cac143080cdd99f879fc5eb446ffabfa5078fabf50908035d92925361bd`.
- Reward source SHA256: `e1a0d46f48926105a58d885e7b1c8761f03b8745937d6232aa130f00043038b5`.
- Aggregate receipt SHA256: `b976f918ae6cd7632a39f0e35065ed160549e137f95b2202ec5dd293dc98f88f`.

The source reward and scoring were shared by trainer selection and evaluation.
This diagnosis introduces no physical hardware, Viam, audio or GPU action.
