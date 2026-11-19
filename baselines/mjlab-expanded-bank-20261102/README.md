# Native mjlab shot-bank expansion

Simulation-only follow-up to the [original learning studies](../mjlab-learning-20261102/README.md).
The twelve previously empty-mask test positions became **development targets**.
Their outcomes can guide this search and training; they are not fresh test data.
Original results, definitions, checkpoints and fingerprints remain intact.

## Development search

The first search simulated 6,144 endpoint-qualified throws and 290 perturbed
confirmation attempts. It found actual route captures at 6/12 positions, but none
of its initially selected settings passed the declared 8-of-10 qualification.
A separate high-clearance/diversity continuation tested 71 captured settings with
710 fresh confirmation attempts. One setting passed 9/10. The other settings and
all failed confirmations remain recorded; aggregate captures are not a source's
repeatability score.

A deeper search spent another 9,216 valid native throws on the six targets with
no captured source. It covered native-accepted board heights~240–600 mm,
tilt 8–60° and drop gaps 50–465 mm. It produced 2,142 whole-ball entries and 1,884
historical one-second retentions, but **zero qualified direct routes**. Of the
retained entries,1,877 involved table contact and 9 robot contact, with two
overlaps. These assisted entries correctly remain failures. A finite failed
search does not establish impossibility.

Total bank-development effort:15,360 native search throws and 1,000 confirmation
attempts, including CPU geometry rejects in the latter denominator. The joined
bank preserves all 16 original templates and adds one measured template, giving
272 candidate slots after the 16 rotations. Joining evidence creates no new
throws and counts shared prior effort once.

| Newly qualified setting, development target 19 | Value |
|---|---:|
| Bucket X / Y |115.753 /−527.463 mm|
| Board center X / Y |488.862 /−243.926 mm|
| Board center height |314.875 mm|
| Tilt from horizontal, facing bucket |29.527°|
| Vertical drop gap reference |385.257 mm|
| Released ball-center height |737.633 mm|
| Board-to-bucket horizontal distance |468.618 mm|
| Fresh perturbed confirmation |9/10|

Coordinates use the shared provisional tabletop frame, not hardware calibration.
The gap is the vertical ball-bottom-to-board-center reference; ball radius is
37.5 mm. These are held endpoints and a prescribed release, not arm motion or a
trained grasp. Selecting a9/10 setting from many candidates does not establish
a90% population capture rate.

## Repeatability and fair comparison

A separate 132-attempt diagnostic compared saved joint branches, fresh IK and
small action perturbations. Saved-branch nominal shots captured 27/30, fresh-IK
nominal 20/30, and perturbed fresh-IK 15/60. Fresh nominal failures included eight
unresolved IK cases and one endpoint collision. The solver's extra random start
depends on batch slot, so an already feasible request can receive a different
branch or fail. Accepted shots can also be sensitive to numerical contact and
retention. [Diagnostic details](../../sircus/mjlab/REPEAT_DIAGNOSTIC.md).

Before training, the study was versioned to v2 with stable-candidate-ID caches.
Every old candidate reuses exactly the same mask, joint branches and failure
reason in both banks. Widened envelopes preserve strict in-domain failures and
branches; only newly allowed actions receive another audit. Adding a template
therefore cannot improve old slots merely by changing their IK batch position.
This fixes comparison fairness; it does not repair the general IK initializer.

## Evaluation protocol

The new study declares 175 training layouts:35 original grid points,12 converted
development points and 128 fresh random coordinates. It samples the 125 contexts
with an endpoint-eligible expanded-bank action;50 excluded training contexts
remain documented. Thirty fresh validation layouts select a checkpoint by
capture count, then shaped reward. Another 100 previously unused continuous
coordinates receive three nominal repeats per method. New random layouts are at
least 5 mm from every historical/development, training and validation coordinate.
This tests interpolation within X100–1240 / Y−650–250 mm, not the whole tabletop
or disjoint held-out spatial regions.

The categorical actor warm-starts from the prior selected model and trains with
actual on-policy terminal PPO:32 updates×128 throws, Adam 0.0003, six epochs,
ratio clip 0.2, entropy 0.001 and value weight 0.5. Frozen learned choices use
argmax. Uniform controls share a seeded per-position quantile. Each selected
action stays fixed across nominal repeats; no per-scene rollout optimization
repairs an evaluation failure. Empty masks stay in the full test denominator.

## Frozen results

All 32 updates and 4,096 new PPO throws completed. Validation selected update 32
(26/30 captures). The following methods use the **same new 100 positions×3
nominal repeats**; these numbers do not replace the prior 50-position baseline.

| Frozen selection | Captures /300 | Capture rate | Clean captures | No-endpoint attempts |
|---|---:|---:|---:|---:|
| Old bank, uniform |175|58.3%|72|75|
| Old bank, old learned selector |183|61.0%|93|75|
| Expanded bank, uniform |182|60.7%|76|75|
| Expanded bank, retrained selector |188|62.7%|145|75|

The learned-versus-uniform capture difference is **+2.0 percentage points**;
the descriptive 95% paired layout-bootstrap interval is−3.3 to+7.7 points.
This does **not establish a capture-rate advantage**. A secondary clean-capture
comparison improves by 23.0 points (descriptive interval+13.7 to+32.3). It is a
separate quality finding, not a changed definition of capture or a multiple-seed
RL result. Nominal repeats remain grouped by position in both intervals.

Both banks have endpoint candidates at 75/100 test positions; coverage did not
increase on this split. The retrained selector captures at least once at 65/100
positions and captures all three repeats at 60/100. Its conditional rate is
188/225 (83.6%) among eligible attempts; the full 62.7% remains the headline.

On the **converted 12 development coordinates**, a fresh old-bank audit finds
one eligible/captured position (3/36 shots); the expanded bank finds two (6/36,
all three nominal repeats at both). Ten positions remain empty-mask cases in
that rerun. One old position is recovered by fresh IK even without bank growth,
consistent with the diagnosed batch-dependent initializer; this does not
relabel its historical result. Only one new template earned promotion.

The independent post-training audit passes 428 checks: original-prefix cache
identity, split isolation, exact candidate IDs/branches, native scoring and
endpoint invariants, all test denominators, source/assets, PPO probability and
weight history, and frozen checkpoint choices. It audits recorded evidence and
independent URDF FK; it does not rerun ball dynamics or certify arm transit.

Checkpoint SHA 256:
`768d21bdbe104f64c87eeec96bda81395f9ffea9af14f211baa53c70f7a898bd`.

Next research gate: make IK initialization deterministic per query and preserve
known branches, then target **airborne** range/retention at the remaining far
positions. More PPO updates cannot create actions in an empty candidate mask.
Any expanded action/contact protocol needs a new fingerprint and untouched test
layouts. Do not loosen the direct-route scorer to count table-assisted entries.

The original 1920 Hz native integration,240 Hz scorer, joint/FK/collision checks,
one-second whole-ball retention and board-only route rules are unchanged.
Rim-assisted bucket entry counts; table, floor, robot or unattributed contacts
invalidate the direct route. Bucket diameters 177/150 mm inner and 200/158 mm outer
remain measured. Reported 175 mm height is still unresolved; the model assumes
175 mm interior depth plus 8 mm bottom. Ball and mount/contact properties remain
provisional. There is no learned pickup, transit, perception or physical result.

## Files and runtime boundary

Search/refinement/far/join summaries and independent audits are separate files.
`expanded-bank.json` contains source identifiers and confirmation provenance.
The final checkpoint, training contract, compact summary, per-position aggregate
outcomes and checksum manifest are published beside them. Full native JSONL,
geometry/contact arrays, optimizer checkpoints, generated configs and logs remain
outside Git in the isolated Ubuntu runtime.

Runtime study directories under `~/sircus-mjlab/output/` are
`expanded-search-v2`, `expanded-refine-v2`, `repeat-diagnostic-v1`,
`expanded-far-v1`, `expanded-combined-v1` and `expanded-learning-v2`.
Each has its own immutable source snapshot under `~/sircus-mjlab/studies/`;
the far search uses source `expanded-final-v1`, and the join uses source
`expanded-learning-v2`. Source and artifact hashes bind these exact bytes.

Only the 4090 runs this task's native work, through the existing shared GPU lock.
Isaac source/setup, its services and the 4070 workload are outside this task.
The existing localhost interface gains a read-only `#expandedStudy` panel.
Its solve controls retain the previous policy; the new checkpoint is not silently
deployed. [Protocol and commands](../../sircus/mjlab/EXPANSION.md).
