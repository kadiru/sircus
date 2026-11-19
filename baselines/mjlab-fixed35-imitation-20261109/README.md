# Fixed 35-position imitation results

**Current imitation: 24/35 (68.6%). Refitted imitation: 23/35 (65.7%).**
Both frozen policies used bucket-facing gripper rotation, 64 substeps and the
same 35 canonical bucket locations. The refit gained one capture, lost two,
shared 22 captures and shared ten failures. Keep the current model; no promotion.

| Model | Captures | Setup declines | Released noncaptures | Invalid physics¹ | Table-assisted captures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current | 24 / 35 | 3 | 8 | 0 | 2 |
| Refit | 23 / 35 | 6 | 6 | 1 | 3 |

¹ The refit's invalid attempt is included among its six released noncaptures and
all 35 requests. It occurred at `(1050, −200) mm`. It is not an ordinary flight
miss, and a passed accounting audit does not make that physics valid.

Exactly **35 locations × two policies = 70 model requests**, plus 18 controls:
88 total slots. All controls and **40,692 independent checks** passed. The
aggregate 59 captures across 88 slots includes both models and positive
controls, so it is not a policy capture rate.

The candidate used 97 retained historical teachers plus two confirmed local
bucket-facing recipes, one example per layout, with 800 actor-only Adam steps.
Both checkpoints and all 70 commands froze before outcomes. No PPO, evaluation
search, retries, noise or placement perturbations. The original checkpoint and
successful recipes remain unchanged.

All 35 positions belong to the historical development grid; 23 exactly overlap
teacher coordinates. This is fixed-grid coverage/regression, not unseen-position
generalization. Both models share the same gripper-facing behavior, so this
comparison does not isolate its benefit over the old orientation.

- [Compact audited summary and all 35 paired settings](summary.json)
- [Study protocol](../../sircus/mjlab/FIXED35_IMITATION.md)
- [Complete handoff and hashes](../../docs/MJLAB_FIXED35_IMITATION_20261109.md)
- [Draft PR #119](https://github.com/kadiru/sircus/pull/119)

Frozen producer `3fa71e2f1bc658aa62d8ed56f7bbfaff67093968`.
Worker 1,100.49 s; pipeline 1,114.97 s. The audited matrix is shown in
`#fixed35Imitation`. The 4090 worker exited; 4070/viewer identities, historical
reports and unrelated UI content passed all 27 final isolation checks. No Isaac
change, service restart or physical action. Raw traces and checkpoints stay
outside Git; this directory contains only the compact reviewed result.
