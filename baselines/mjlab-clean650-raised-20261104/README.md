# Fixed650 raised-board follow-up — November 4, 2026

The [CPU reach diagnosis](../mjlab-clean650-20261104/reach-diagnostic.json)
motivated a separately frozen height/tilt/gap search. Board/bucket XY remain
unchanged at650mm separation, with the same calibrated contacts, score, strict
5mm endpoint gate and provisional175mm bucket-height interpretation.

Native producer commit `f223aaa985c92aad8325d2aa207877271df7a54b`;
contract `fe28a1277b7d7e32428dd736346d9336ca274d7a0fe9c0dc11ac61bab6970036`.

| Phase | Requested | Native | Clean captures | Rim captures |
|---|---:|---:|---:|---:|
| Initial controls |12|12|6|0|
| Raised grid plus reference |28|16|0|1|
| Local refinement |81|50|0|1|
| Five matched settings |30|30|0|3|
| Final controls |12|12|6|0|
| Total |163|120|12|5|

**No new successful setting and no clean650 capture.** The two development
captures repeat the same original380mm/28°/386.8mm recipe; all12clean all-phase
captures are positive controls. All109 discovery proposals remain in the
denominator, including43 geometry rejections. The163-request study exhausted
its prespecified budget without hidden retries or changed materials/scoring.

| Verification H mm | Tilt ° | Gap mm | Ball-center H mm | Capture64 | Capture128 |
|---:|---:|---:|---:|---:|---:|
|380|28|386.8|804.3|3/3rim|0/3|
|380|28|389.3|806.8|0/3|0/3|
|395|28.5|324.3|756.8|0/3|0/3|
|395|27.5|326.8|759.3|0/3|0/3|
|405|26.5|364.3|806.8|0/3|0/3|

All verification traces had valid physics and online/independent scoring
agreement. Original paired350ms drift was1.734mm; the other targets1.688–2.467mm.
Small early-flight drift did not imply capture agreement at the rim. Both
before/after positive controls were3/3clean at each timestep; negatives0/3.
The full independent auditv3 passed31,902 checks over exact frozen plans,
geometry, native options, dependencies and dense contact observations.

## Combined development evidence

[Byte-bound comparison](comparison.json) preserves every study's original
fingerprint and export hashes. Distinct targets use9-decimal configured-action
keys only for counting; executed values are unchanged.

| Discovery only | Requests | Native drops | Distinct native settings | Captures | Clean |
|---|---:|---:|---:|---:|---:|
| [Coarse v1](../mjlab-clean650-20261104/README.md) |442|101|98|2|0|
| [Fine v2](../mjlab-clean650-fine-20261104/README.md) |157|52|37|4|0|
| Raised v3 |109|66|63|2|0|
| Combined, settings deduplicated |708|219|190|8|0|

The8 captures all use one original parameter target and are rim-assisted;
they are not8 independently successful configurations. There are649distinct
requested targets across profiles, but only190distinct targets admitted native
flight. Controls and verification add162native releases, yielding870total
requests/381native releases. All36clean all-phase captures are positive controls.

This is finite fixed-layout development search and nominal repeated engineering
verification, not an exhaustive feasibility proof, noise robustness, policy
inference/generalization, grasp/transit, perception or physical validation.
Some unresolved IK poses violate a model-specific necessary workspace bound;
other finite solver failures remain unproven. Search ranking on misses can favor
late post-table proximity, so future search should use pre-external flight
progress and consider distinct refinement sources.

Further PPO stays on hold. The next bounded study should reconsider bounce-plane
XY or endpoint-orientation constraints while preserving650mm distance and a new
explicit scene fingerprint; do not weaken scoring/compatibility guards or
silently relabel these results. No noise, material refit or policy activation.
Existing viewer,4070 workload and Isaac setup were not restarted or reconfigured.

Files `summary.json`, `requested-shots.csv`, `checksums.json` and `comparison.json`
contain configured parameters, outcomes and hashes only. Dense observations,
joint/pose arrays, logs, host paths, credentials and generated runtime state
remain outside Git. The [read-only UI](http://127.0.0.1:18781/#clean650Study)
shows the latest profile, matched controls, verification, saved native flights
and earlier publication summaries.
