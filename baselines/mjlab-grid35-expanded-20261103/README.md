# Expanded 35-position shot-planner study

These are **searched simulation recipes**, not a trained policy, hardware results
or proof of continuous tabletop coverage. All 35 distinct bucket coordinates
remain visible, including unresolved and clearance-admission failures.
See the [frozen protocol](../../sircus/mjlab/GRID_STUDY.md).

## Final selected bank and diversified follow-up

The final result is **17 selected positions, 12 unresolved and 6 clearance
blockers**. All 35 rows, including missing settings, are in
[per-table-settings.csv](per-table-settings.csv). The interface uses this same
export function. [recipe-bank.json](recipe-bank.json) binds each displayed recipe
and frozen rate to its source configuration. No recipe was automatically added
to the ordinary solver catalogue or deployed as a policy.

Sixteen selected recipes retain the completed parent's exact configurations and
original frozen cohorts. One replacement has a new independent cohort. Individual
recorded rates are **28–30/30 perturbed**, with **3/3 nominal** for each; ten recipes
scored 30/30. These stored cohorts are not a fresh common-bank evaluation, unseen
target evaluation or learned-policy success rate.

The replacement is at table **2·2**, bucket **(290, −425) mm**:

| Setting | Value |
|---|---:|
| Board center XYZ above table, mm | 622.894, −398.580, 268.099 |
| Board tilt from horizontal | 31.519845 degrees |
| Nominal ball-bottom drop gap | 229.560 mm |
| Ball-center release Z | 535.158 mm |
| Horizontal board-center to bucket-center distance | 333.941 mm |
| Qualification | 3/3 nominal; 19/20 perturbed, all clean |
| New frozen evaluation | 3/3 nominal; 28/30 perturbed, all clean |

The parent recipe scored 22/30. The replacement changes board XY/bearing, heading
and joint branches; height, tilt, gap and distance are essentially unchanged.
The independent seeds also differ. This is an observed new-configuration result,
not proof of a causal height/gap improvement or a statistical reliability guarantee.

The follow-up checked **13,312 distinct poses** at the 13 unblocked refinement
targets. Of 6,656 broad/inverse proposals, only **101 (1.52%)** passed endpoint and
clearance admission; **all 101** were tested natively. Another 1,563 native search
shots were local proposals. Broad quotas were conditional on eligibility, not a
claim that 64 broad shots were feasible at every target. Six starting-clearance
failures exited before candidate search. No additional target qualified.

Its 1,664 search requests, 118 qualification requests and 33 frozen requests all
released natively. The independent follow-up audit passed **74,791 checks**,
including scalar rescoring and parent/selection/cohort verification. Together the
two studies requested 4,243 shots and released 4,234; those are work counts across
different roles, not a pooled capture rate. Both finite jobs exited normally.

The remaining 18 positions have different bottlenecks:

| Positions | Remaining limitation in the diversified pass |
|---:|---|
| 8 | No qualifying direct capture in 128 shots each. Some basic bucket entries followed external contacts and were correctly rejected. |
| 3 | Scoring flight, but no tested common-home path admitted: table 2·6 (3/128 search captures), 3·6 (1/128), 5·5 (94/128). |
| 1 | Table 1·5 captured 3/3 nominal but only 16/20 perturbed, below the 18/20 qualification gate. |
| 6 | Starting-pose clearance remains unproven; no native shot search started. |

All 1,536 search shots across the twelve unresolved targets hit the board. This
does not imply a valid board-to-bucket route, successful retention, or an admitted
setup motion. More flight trials alone cannot resolve a missing path proof.

## Completed parent pass

The first pass selected settings at **17/35 positions**, with **12 unresolved**
and **6 blocked by the prescribed starting-pose clearance check**. Its independent
audit passed **66,373 checks**, including scalar rescoring of saved native data.
It did not independently rerun endpoint/path geometry.

After selection was frozen, its recipes scored **51/51 nominal** and **495/510
fresh perturbed captures**. The latter comprise **312 clean + 183 rim-assisted**
captures. Ten recipes scored 30/30; sixteen reached at least 27/30. One scored
22/30 despite 19/20 during qualification; that original result remains unchanged.
These are aggregate counts of different recipes, not a learned-policy test rate.

The parent requested 1,280 search trials, 587 qualification trials and 561 frozen
trials. Nine qualification requests were rejected by endpoint/path admission;
they remain in the denominators. In total it requested 2,428 shots and released
2,419 natively. Search, selection and final evaluation must not be pooled as a
performance estimate.

[Parent settings and final per-table rates](parent-settings.csv) include all 35
coordinates and full-precision values. Board Z is above the table. Release Z is
the **ball center**, not a flange coordinate. Gap is the nominal ball-bottom
height above the board center; it is not the travel distance to first contact
with the tilted board. The 37.5 mm radius is separate from the varied gap.

## Evidence layout

`parent/` preserves exact compact report, contract, frozen selection and audit
bytes. `selected-settings.json` adds the hash-bound simulator joint branches and
full native endpoint configuration. `original-manifest.json` refers to the full
retained external study, including raw trajectories; those arrays and runtime
logs are deliberately absent here. The separate bundle `manifest.json` covers
only files actually shipped in that directory. `curate.py` reproduces this compact
copy and refuses incomplete or unaudited studies.

`followup/` uses the same compact layout, with an additional per-target
`diversity-summary.json` distinguishing planned, checked, eligible and selected
parameter ranges. Neither planned bounds nor rejected poses are described as
tested flight coverage. `manifest.sha256` covers this complete published bundle;
use `sha256sum -c manifest.sha256` on Ubuntu to verify it.

## Interpretation and limits

A successful route requires a board rebound, whole-ball entry and one continuous
second inside the bucket. Rim-assisted captures count; table, floor or robot
assistance disqualifies the route. Perturbed repeats use independent Gaussian
board X/Y noise of 1 mm, board Z/gap noise of 2 mm and tilt noise of 0.35 degrees.
Neither nominal repetition nor this small perturbation distribution guarantees
physical reliability.

The six starting-pose blockers record **unproven clearance**, not negative fixed
base penetration. Four concern the frozen arm posture near the bucket; two
concern base/bucket clearance. Other targets have scoring flights but lack an
admitted common-home route. These distinctions must survive future summaries.

Native MuJoCo/mjlab uses prescribed arm endpoints before every contact substep;
the sampled setup paths do not model driven grasp/transit or impact compliance.
The board is 300 x 200 x 5 mm in its landscape mount. Bucket geometry retains the
177/150 mm inner and 200/158 mm outer taper and **reported height 175 mm** with its
interpretation still unresolved. The provisional model uses 175 mm interior plus
an assumed 8 mm bottom. This study changes neither those assumptions nor any
physical/Viam/audio resource.
