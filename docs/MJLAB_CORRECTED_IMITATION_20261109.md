# Corrected imitation — audited new-position results

Date: November 9, 2026. Simulation only, fixed 64 substeps.

**Refitting did not improve this paired test: corrected imitation captured
38/70 (54.3%), current imitation 40/70 (57.1%).** Four positions improved, six
regressed, 34 captured with both and 26 failed with both. Keep the current
policy; there is no automatic promotion. Two fewer captures in this one small
paired sample do not establish significant general worsening.

[Compact results and all paired positions](../baselines/mjlab-corrected-imitation-20261109/README.md)
are the handoff source. [Frozen protocol](../sircus/mjlab/CORRECTED_IMITATION.md)
defines the comparison. Draft [PR #112](https://github.com/kadiru/sircus/pull/112)
stacks on #111. Frozen producer/UI commit:
`fecbe4d61bd92c912e0b4d80b1fdd7be3a2bcb48`.

## Training and evaluation

- The candidate starts from `coverage-recovery-v1` recovery imitation, retains
  all 92 teachers, and adds five unique corrected recipes (one height, two gap,
  two board-X/Y corrections). Each added recipe had 3/3 prior nominal captures;
  repeated shots are evidence, not duplicated teachers. Source contexts,
  records and hashes are preserved in the compact summary.
- 800 full-batch actor-only Adam steps, learning rate 0.001, on 97 teachers.
  Actor training MSE fell from 0.00535240 to 0.00102469. This lower fitting loss
  did not translate into better capture on this new sample. Critic and
  exploration variance, architecture, features, bounds, geometry prior,
  physics and scoring stayed unchanged; no PPO ran.
- Current weights stayed unchanged. Both checkpoint artifacts and all 140 TEST
  commands were frozen before the first TEST outcome. The comparison measures
  five additional examples plus 800 additional fitting steps together, not an
  isolated causal effect of those examples.
- Seed `2026110941`: 70 fresh bucket positions, two per cell in a 7 × 5 grid,
  at least 5 mm from the frozen history/training inventory and each other.
  Freshness is limited to that inventory. Geometry-only admission may choose
  an anchor; no TEST flight search, retries, replacement positions or fitting.
- Each policy gets one request per position. Corrected: 23 setup declines,
  47 releases, 38 captures and nine flight misses. Current: 19 declines,
  51 releases, 40 captures and 11 flight misses. Invalid physics: zero each.
  Of captures, three corrected and two current used table assistance. Direct
  captures (35 and 38) may include rim contact; they are not necessarily clean
  rim-free entries. All declines remain in the denominator.
- Unchanged v3 capture requires measured board bounce, whole-ball entry and one
  continuous second retained. Table/rim assistance counts. Robot, floor and
  unattributed contacts invalidate the route. Historical scoring is preserved.

## Controls, audit and six regressions

All **158 requests** completed: nine controls, 140 TEST requests, nine controls.
Both control blocks passed with three direct captures, three table-assisted
captures and three expected misses each. Across all requests there were 116
releases, 90 captures, 26 released misses and 42 setup declines, with no invalid
physics. Independent CPU audit passed **55,633 checks**; full and native checks
are verified. Worker wall time: **1,387.39 s**; pipeline: **1,414.47 s**.

The six corrected-policy losses relative to current are the next diagnostic
set. These are inspected development cases now, not a future untouched test.
Coordinates below are bucket X/Y in millimetres, rounded for readability.

| New TEST context suffix | Bucket X/Y (mm) | Corrected outcome |
| --- | --- | --- |
| `r0-c0-n0` | 140.4, −608.4 | Finite IK unresolved |
| `r0-c1-n0` | 343.4, −579.0 | Finite IK unresolved |
| `r3-c0-n1` | 215.0, −70.6 | Finite IK unresolved |
| `r0-c6-n1` | 1107.0, −536.3 | Released miss |
| `r1-c1-n0` | 316.7, −453.1 | Released miss |
| `r1-c2-n0` | 496.4, −414.7 | Released miss |

Full IDs start `corrected-imitation-test-`. Current captured in all six cases.
The four gains recovered three current endpoint-clearance/pose declines and
one current released miss. These classifications identify where to inspect;
they do not yet prove the physical cause of each failure. Finite IK failure
does not prove a position impossible. Preserve exact successful commands and
compare requested/realized poses and flight traces before choosing a change.

## Evidence and interface

| Item | SHA-256 |
| --- | --- |
| Contract fingerprint | `297b367fa2702d688084bc2e74dd30a83c6cf4312ba078938def5f9b223adce4` |
| Completed report | `1fde3f7de5b5c2912a133c858105388a704a80ab0e2cb39e8d79c0c7a189bf14` |
| Independent audit | `ea4992828178d9e62ba41c63579ecc3876cb5bffaacf6b590c828d45876a7f63` |
| Corrected checkpoint | `986408887edb9b443f01120201660cc55943310dace81a102652dfc5006acd5b` |
| Current checkpoint | `688acc7981522c3281dff47edd19edaf46f084f113455e50f14d7e5cabc3b9e9` |

Ubuntu evidence is under `/home/kadir/sircus-mjlab/output/corrected-imitation-v1`,
with adjacent `corrected-imitation-v1-audit.json` and `.pipeline.json`.
The study's `checkpoints/{corrected_bc,current_bc}.pt` retain both frozen models.
Local ignored inspection files are `.runtime/corrected/{report,summary,snapshot,
publication-receipt,final-isolation}.json`, `corrected-imitation-v1-audit.json`,
and `ui-results.png`. Raw traces, models, proofs, logs and runtime configuration
stay outside Git; the compact baseline summary is byte-identical to the audited
publisher output. Original recipes, reports and the older 37/70 score remain
unchanged; 37/70 was measured on a different sample and is not this comparator.

The audit-gated `#correctedImitation` panel displays scores, paired gains/losses,
all 70 positions, exact corrections, controls, teacher lineage and model hashes.
Visual QA passed. Its 75,337-byte snapshot preserves all 57 previous report
keys/bytes and historical HTML/CSS outside the new addition. Only the Mac SSH
tunnel was restored; no server restarted. Final read-only isolation checks
confirmed the 4090 worker exited and the existing viewer/4070 workload retained
process identity. Isaac and historical reports stayed unchanged.

This is nominal privileged-geometry, held-endpoint ball-flight evaluation.
It does not validate moving-arm transit, grasping, perception, physical robots,
noise robustness, timestep convergence or all-table reliability. No interactive
selector activation or further experiment was launched.
