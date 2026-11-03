# Matched current-ball sweep/PPO study — frozen v1 results

All six declared workers completed and the independent source/provenance/native-trace audit passed.
These are simulation results for **strict geometric one-second containment**, with prescribed held endpoints.

**Interpretation limit:** The v1 wall predicate allows only 1 micrometre nominal overlap; selected validation traces show material contact-boundary sensitivity. Formal labels do not establish an ordinary bucket-capture or material-adaptation winner. Original scores, selected artifacts and source evidence remain unchanged.

Both methods used the same 17 historical recipes, 1,024 new requested training decisions per seed, and the same 35/12/24 train/validation/test layout pools. Bounds, IK and path rejects consume decisions. Three seeds per method give 6,144 TRAIN + 864 validation + 432 test requests = 7,440. The 50 engineering controls and unquantified older development cost remain separate.

## Validation at equal requested budgets

Entries are three-seed mean ± sample SD; each seed has 36 paired requests at each milestone. The repeated layouts/noise are dependent, so these are descriptive seed statistics, not binomial confidence intervals.

| Method | Requested TRAIN | Formal validation rate | Setup + TRAIN wall (s) |
|---|---:|---:|---:|
| ppo | 0 | 44.4% ± 0.0 pp | 0.7 ± 0.0 |
| ppo | 256 | 44.4% ± 4.8 pp | 974.0 ± 13.1 |
| ppo | 512 | 43.5% ± 4.2 pp | 1356.1 ± 18.2 |
| ppo | 1024 | 42.6% ± 7.0 pp | 2099.6 ± 56.9 |
| sweep | 0 | 45.4% ± 1.6 pp | 0.3 ± 0.0 |
| sweep | 256 | 35.2% ± 3.2 pp | 959.2 ± 14.7 |
| sweep | 512 | 40.7% ± 4.2 pp | 1333.5 ± 21.7 |
| sweep | 1024 | 36.1% ± 5.6 pp | 2097.0 ± 62.2 |

## Separately frozen test cohorts

The artifact was selected using validation, frozen before test, and made one deterministic prediction per test layout. Each prediction had one nominal and two paired perturbations; no test search, retry or adaptation.

| Method | Seed | Selected TRAIN budget | Formal test successes / requests | Native releases / requests |
|---|---:|---:|---:|---:|
| ppo | 20261311 | 256 | 16/72 | 36/72 |
| ppo | 20261312 | 1024 | 19/72 | 36/72 |
| ppo | 20261313 | 0 | 23/72 | 36/72 |
| sweep | 20261311 | 0 | 24/72 | 36/72 |
| sweep | 20261312 | 0 | 22/72 | 36/72 |
| sweep | 20261313 | 0 | 21/72 | 36/72 |

## Resource interpretation

`per-seed-curves.csv` and `per-seed-final.csv` retain all observed resource fields. `aggregate-curves.csv` and `aggregate-final.csv` provide three-seed descriptive statistics. Unknown measurements remain empty/null, with observed-seed counts; they are never reported as zero.

Setup/TRAIN wall and worker CPU time exclude evaluation. Geometry is an endpoint/home subtotal; sampled path CPU work remains in total time. Native GPU-phase wall time is not active kernel time. GPU reservation includes evaluation/initialization; RSS/VRAM are lifetime peaks including evaluation. NVIDIA VRAM is sampled once per second. Memory is MiB; CPU excludes telemetry subprocess cost. Selected artifact bytes exclude full per-update histories, other milestone files and raw audit-data storage. The selected PPO checkpoint retains its original frozen CPU optimizer state.

## Scope and next step

This restricted residual benchmark uses privileged geometry and fixed current-ball materials. It does not measure learned grasping, driven arm trajectories, impact compliance, visual perception, physical success or new-ball adaptation. No model is activated by this bundle.

Before claiming an ordinary capture/adaptation ranking, independently freeze contact controls (rest/tangency, retained impacts, rim-assisted entries, true partial exits/re-entry and outside negatives), calibrate the contact envelope without method outcomes, version scorer/reward/contract, and evaluate afresh. Do not retune the original v1 labels to recover selected failures.

## Evidence

Independent audit: 902,252 passed checks; 7,440 requests and 3,738 native releases.

`report.json`, `contract.json`, `audit.json`, and selected artifacts are byte-for-byte copies. `provenance.json` binds them to the original study manifest and frozen source hashes. `original-manifests/` contains the unchanged source manifests as references, including names/digests of unbundled raw evidence. Those are separate from this compact bundle's `SHA256SUMS`; do not run original full-study manifest checks against the subset. Original `prior.json` and `splits.json` preserve the shared geometric inputs needed for frozen inference. The original auditor checked all native traces; this curator rechecks compact copied bytes and source identity, and does not rerun physics or rehash the excluded raw arrays. `SHA256SUMS` covers this bundle. Raw arrays/observations/logs, runtime host paths and credentials are excluded.
