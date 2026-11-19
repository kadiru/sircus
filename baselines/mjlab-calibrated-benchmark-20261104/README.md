# Calibrated current-ball sweep/PPO benchmark — v2

Evidence kind: **native_current_ball_v2**. Independent audit passed 1,204,473 checks over 7,440 requests.

Both methods share the 17 historical recipes and the same geometry-only pre-prediction source/staging resolver. Every requested training decision counts, including rejections. Geometry search is separately timed; held-out predictions do not perform ball-flight outcome search.

| Method | Paid TRAIN requests | Frozen test captures / requests |
|---|---:|---:|
| sweep | 3072 | 78/216 |
| ppo | 3072 | 97/216 |

CSV aggregates are descriptive three-seed means and sample SDs; paired layouts/repetitions are dependent. Setup/TRAIN time excludes evaluation; reservation and lifetime memory include it. Query timing is cohort prediction/preparation, not an isolated actor-call latency; pre_prediction_plus_policy_s adds geometry resolver wall time. Same-action endpoint/path admission and native rollout remain separately timed, not included in that prediction-only sum. PPO checkpoint bytes include critic/optimizer state.

Calibration/verification/home records are copied exactly and their digests are frozen in the contract. Dense native traces were independently rescored by the passed audit; this subset export does not repeat physics or rehash unbundled raw arrays. Original manifests under original-manifests reference unbundled source evidence. SHA256SUMS covers only this compact bundle.

Historical v1 uses different scoring, setup preprocessing and evaluation cohorts; do not pool its counts or attribute a v1/v2 rate difference solely to learning. These are privileged-geometry held-endpoint simulator results, not physical grasp, driven transit, impact compliance, perception or new-ball adaptation validation. No checkpoint is activated by this bundle.
