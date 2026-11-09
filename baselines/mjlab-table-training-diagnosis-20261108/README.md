# Table-assisted PPO failure diagnosis

Posthoc analysis of the completed [50-position comparison](../mjlab-table-assisted-training-20261108/README.md).
The new PPO policy captured **15/50**; this package explains the remaining
**31 released misses and four setup declines**. These formerly held-out positions
are now development evidence. No policy was trained or replaced by this diagnosis.

| Observed first contact after the board rebound | Misses |
| --- | ---: |
| Table | 16 |
| Bucket | 8 |
| Robot | 7 |

Every released miss bounced from the board, but **none achieved whole-ball
entry**. Thirteen had a forbidden contact before the original scoring terminal;
18 did not. Contact order describes the saved trajectory: it does not establish
the cause of failure. A bucket contact alone does not prove a rim strike.

## Concrete trajectories

Times below are seconds after release, from saved native force/state arrays.

| Case | Board height | Tilt | Drop gap | Observed sequence | Outcome |
| --- | ---: | ---: | ---: | --- | --- |
| 00 | 340.6 mm | 17.95° | 251.8 mm | Board 0.2245 → table 0.6313 → bucket 0.6374 | No entry |
| 07 | 379.2 mm | 28.38° | 215.3 mm | Board 0.2056 → bucket 0.4915 → table 0.7251 → robot 1.1192 | No entry |
| 18 | 339.1 mm | 16.20° | 240.2 mm | Board 0.2194 → robot 0.5344 → table 0.6432 | No entry |
| 02, capture comparator | 427.6 mm | 31.78° | 158.3 mm | Board 0.1742 → table 0.5178 → bucket 0.7396 → entry 0.8824 → capture 1.8824 | One second retained |

Case 02's bucket was 703 mm horizontally from the board. Its initial flight crossed
the bucket-rim height about 376 mm short, then the table-assisted route captured.
Therefore an initial-flight undershoot is **not** a valid rejection rule for this
task. Later table and bucket interactions must remain part of the evaluation.
The grouped contact sequence and first-flight cutoff agree for every released
failure. An earlier preliminary 17-table/7-bucket tally was a counting error;
the saved, source-bound diagnosis gives **16/8/7**.

## The four setup declines

- **Cases 06 and 36:** the finite search found respectively 9 and 6 valid dropper
  branches, but **zero valid board-arm branches** for the requested action.
  An anchor pose was admitted in each scene; that does not admit the policy's
  requested residual. Requested board heights were 474.7 and 469.6 mm versus the
  anchor's 459.1 mm. This is not proof that those poses are unreachable.
- **Case 14:** the bucket placement failed the board-arm fixed-base clearance
  check: 11 fixed-base blocking pairs, one inconsistent distance query.
  The closest reported pair was `Arm14/link_base/collision/0` and `bucket/wall00`.
- **Case 46:** 20 fixed-base blocking pairs and one inconsistent distance query;
  closest pair `Arm10/link_base/collision/0` and `bucket/wall21`.
  For both fixed-base cases, the conservative minimum clearance lower bound
  was 0 mm against a required 5 mm. **A zero lower bound is not proof of penetration.**

No admission guard was removed, and no rejected action was silently replaced.

## One targeted correction, completed

Freeze board tilt at **32°**, leaving board XY, height, drop gap and bucket
unchanged. Select the first two sorted shallow, far misses per first-contact
class: original tilt <22°, distance ≥450 mm. This selects **00, 01, 18, 26**;
the bucket-first class has no eligible case. The selection stays fixed.

The native test has **42 requests**: four cases × baseline/corrected × three
repeats, plus 18 direct-capture, table-assisted-capture and miss controls before
and after. Baselines preserve their original admitted joint branches; the changed
tilt requires its own exact-action admission. Fixed 64 physics and v3 scoring stay
unchanged. The hypothesis comes from the successful approximately 32° far-shot
family; this is not causal proof, a fresh holdout score or a general improvement.
All **42 requests completed** in **431.8 worker-seconds**. The independent
native/geometry/scoring audit passed **12,557 checks**; both control gates passed.

| Variant | Capture / requests | Released | Setup declined | Robot-contact shots |
| --- | ---: | ---: | ---: | ---: |
| Original exact branch | 0/12 | 12 | 0 | 6/12 releases |
| Tilt changed to 32° | 0/12 | 0 | 12 | No executed flights |

| Case | Board H | Original tilt → tested tilt | Gap | Corrected valid IK branches: dropper / board |
| --- | ---: | ---: | ---: | ---: |
| 00 | 340.6 mm | 17.95° → 32° | 251.8 mm | 10 / 0 |
| 01 | 354.1 mm | 18.56° → 32° | 239.8 mm | 10 / 0 |
| 18 | 339.1 mm | 16.20° → 32° | 240.2 mm | 8 / 0 |
| 26 | 352.4 mm | 15.83° → 32° | 242.8 mm | 6 / 0 |

Each changed request failed `finite_ik_unresolved` before release. No board-arm
branch was found, so no pair/path or corrected-flight test followed. Three
repeat requests retained each identical admission result; they are not three
independent IK searches. Zero corrected robot contacts therefore demonstrates
no contact improvement. The finite search does not prove the pose unreachable.
All 18 control requests behaved as expected: six direct captures, six captures
with table contact and six misses. No physics-invalid shot occurred.

**Do not promote this correction or train on a claimed improvement.** The next
development step is to find jointly admitted height/tilt combinations, keeping
clearance and exact-action guards intact, then test their flights. This test
does not change the original 15/50 policy result.

## Evidence and scope

- [summary.json](summary.json): counts, geometry, source identities, limitations
  and frozen correction protocol.
- [failure-cases.csv](failure-cases.csv): all 35 failures, exact shot parameters,
  first contact times, setup quantities and record/archive/proof hashes.
- [capture-comparators.csv](capture-comparators.csv): two clearly separate
  captured trajectories, cases 02 and 31; these are examples, not another test set.
- [correction.json](correction.json): all four paired configurations, control
  gates, admission reasons, audit and evidence hashes. Frozen producer `fe7f9b3`.
- [Diagnostic producer](../../sircus/mjlab/table_training_diagnosis.py).

The diagnostic's **143 checks** authenticate sources, accounting, exact setup
proofs and saved trace columns. This producer is not an independent native
resimulation or scorer. Original physics, geometry and v3 scores inherit the
separate completed parent audit's **823,442 checks**, authenticated by exact
report and audit hashes. Raw trajectories, archives and machine paths stay out
of this package. The UI plots downsampled actual traces from external evidence.

This is privileged-geometry, held-endpoint simulation at 64 substeps. It does not
validate moving-arm transit, noise robustness, 128-substep convergence, perception
or physical transfer. Observed contact channels use the unchanged >0.01 N
threshold and stop at each original scoring terminal; later contacts are excluded.

Diagnosis SHA-256:
`4ccc5208b5b48bff74504e6103dca5c86f0ec59b22d47f5d7b8918352d9f4544`.
Diagnostic audit SHA-256:
`1c479d2433e5dd6e72a0a3635e179a9e7bd84c3e6661fc30c3735b6d0598713b`.
Correction report SHA-256:
`471b8c0a10862c460b43a12dea8f8e597cec29bc2116f9545c285697ddd95217`.
Correction audit SHA-256:
`2454c4c88f39ce37cf38fe0ea929cca53922da5e5b18ec00d327ac07cf5f7573`.
