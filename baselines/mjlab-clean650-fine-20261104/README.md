# Fixed650 fine neighborhood — November 4, 2026

Separately frozen follow-up to the [coarse study](../mjlab-clean650-20261104/README.md).
Same scene, calibration, strict5mm endpoints, ball, provisional bucket and scorer;
no noise or PPO. Native producer commit `6a3c5bf12ecee618dfa5096cdefe76acbdfccea7`.
Contract `176b1d2662b8805e4e41243885f16b25f97ee9c47e11296ea499af8f8c3a233b`.

| Phase | Requested | Native | Clean captures | Rim captures |
|---|---:|---:|---:|---:|
| Initial controls |12|12|6|0|
| Fine grid plus reference |76|21|0|2|
| Local refinement |81|31|0|2|
| Five matched settings |30|30|0|3|
| Final controls |12|12|6|0|
| Total |211|106|12|7|

**No clean650 setting qualified.** All12 clean captures belong to positive
controls. Discovery's four rim captures are repeated observations of the same
original380mm/28°/386.8mm target, not four distinct successful settings. Adaptive
refinement retained two source records with identical geometry; all81 requests
count, but they are not81 independent parameter configurations.

The nominal verification targets were:

| Board H mm | Tilt ° | Gap mm | Ball-center H mm | Capture64 | Capture128 |
|---:|---:|---:|---:|---:|---:|
|380|28|386.8|804.3|3/3rim|0/3|
|380|28|389.3|806.8|0/3|0/3|
|380|28|390|807.5|0/3|0/3|
|380|28|387.5|805|0/3|0/3|
|385|28.25|384.3|806.8|0/3|0/3|

Both initial/final positive controls were3/3clean and negatives0/3 at each
resolution. Full independent auditv2 passed33,824 checks and binds exact
report/manifest bytes. These are nominal simulated held-endpoint tests, not noise
robustness, held-out inference, grasp/transit, perception or physical validation.

`summary.json`, `requested-shots.csv` and `checksums.json` publish allowlisted
configured actions, outcomes and provenance only. Raw joints, dense positions,
native caches, host paths, logs and credentials remain outside Git. The bucket's
175mm reported-height interpretation remains provisional.4070/Isaac unchanged.

The separate [CPU reach diagnosis](../mjlab-clean650-20261104/reach-diagnostic.json)
motivates a higher-board follow-up; it is not evidence of a successful ball flight.
