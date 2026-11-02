# All 35 tables participating — November 2, 2026

Actual Isaac Sim 5.1 / Isaac Lab 2.3.2 training on RTX 4090. The expanded pose search found a usable local operating region at every one of the original 35 distinct bucket positions. All 35 tables then released a ball in each of ten fresh batches: **350/350 releases, zero rejected releases, five PPO updates**. There were **111 route-qualified captures (31.7%), 59 clean captures and 101 captures with exactly one board-contact episode**. Clean and single-bounce are overlapping subsets. These are training counts, not held-out policy performance.

All bucket coordinates, arm bases, original geometry, reported-height uncertainty and the silver-gray appearance remain unchanged. The actor now chooses five shot parameters: bucket-relative bearing, horizontal board-to-bucket distance, board height, ball-bottom gap and board tilt. Each table has 24 independently native-qualified options; all five parameters vary within each menu. Nearby configurations stay on a consistent IK branch around one measured holding pose. The original 400 mm-only experiment and its four-output policies retain their own definitions.

## Recording

The user's exact selected camera is fixed throughout; no tracking or generated imagery. All ten training batches are included, including unsuccessful background trials. Each cut retains 36 measured setup frames, the release, rebound and immediate foreground outcome, with 0.20–0.229 seconds of visual padding. Full one-second retention scoring uses the complete 240 Hz telemetry, even when a quiet retention tail is cut from the picture.

| File outside Git | Frames | Duration | SHA256 |
|---|---:|---:|---|
| `action-training.mp4` | 737 | 24.57 s | `38e712086ca7e0e3d19926b751900e3f69a0dbcaeb221b000985fadd7bc278db` |
| `action-training-untrimmed.mp4` | 1570 | 52.33 s | `65d30486fdc761d400169ea348bcc02882dc7a329db9db7d53b4098d7b27147f` |

Both files are native 1920×1080, 30 fps. The complete captured sequence contains every recorded frame, not every second of the 774.65-second training loop. IK, settling, optimizer and capture waits were not filmed; setup samples compress scripted joint travel, while ball flight plays at physical speed. The close foreground site captured in all ten batches; this was the previously selected camera site, not a site selected afterward for successful footage. Nine sites repeat one projected configuration across the ten sampled actions. All 35 release and contribute to PPO, but every arm does not change posture in every batch.

## Search and transition diagnosis

The first CPU screen covered 28 sites, then deeper searches covered all 35. A wider native menu ultimately qualified all sites, but a driven diagnostic still released at only 31. Independent FK checks on all 350 rows and exact cache-target checks ruled out joint-index/frame/cache-key errors. Large joint changes remained at rejected transitions. Without transient contact or torque measurements, a collision or actuator-limit cause was not established. [Diagnostic](transition-audit.json).

The final search refines around one actual holding configuration per site, limits each candidate to 0.35 rad per joint from that anchor, then requalifies it natively. This is an anchor bound, not pairwise swept collision/path certification. Expanded training also waits for all qualified sites, within the existing time budget; legacy filming behavior is preserved. All ten final bounded drive waits ended at 34/35 joint-ready cells. The unchanged subsequent holding/settling checks then independently permitted all 35 releases each time. No tolerance or release guard was relaxed.

A narrower rehearsal is preserved separately as `output/expanded-training-20261102-v3`: four options per site, 350 releases, 109 captures. The final richer run below uses 24 options per site. This is not a controlled policy comparison: the menus differ. No earlier frames or outcomes were reused.

## Evidence and experimental checkpoint

- [Per-cell counts and menu ranges](cell-summary.json), [CSV](cell-summary.csv): exact bucket positions, all 35 release counts and actual sampled-configuration counts. Activating every site does not mean captures were found at every site.
- [Native-qualified menu](expanded-menu-qualified.json), [contract](contract.json), [menu lineage](menu-provenance.json): 840 measured endpoint qualifications and the five-dimensional task. Offline proposal selection uses geometry/IK, not release rewards or capture outcomes.
- [Training report](result.json), [film verification](film.json), [CPU checkpoint audit](checkpoint-audit.json): all 350 sampled latent likelihoods recomputed from saved policies, five logged update links, maximum log-probability error `9.5367431640625e-7`. This does not independently reconstruct the optimizer or establish generalization.
- [Initial policy](policy-00.pt), [experimental final policy](policy-final.pt): load only with `torch.load(..., weights_only=True, map_location="cpu")`. Five-output checkpoint; incompatible historical four-output policies remain rejected. Do not promote it on a training film.
- [Original camera](selected-camera.json), [separate camera binding](camera-binding.json): original bytes and original scene fingerprint preserved, with the new experiment bound explicitly.
- [Native contact/isolation probes](probes.json), [rim rays](rim-rays.json), [verification](verification.json), [file manifest](manifest.json).

New task fingerprint: `63362ef37a93881f769b39d1ee7456335f1cc7bf5db04ffa7e548ce9697455bf`. Original camera-source fingerprint: `98ebe47b48a01b9028747ab03bc913a033d90611e44f03b912e58f41c87ff79a`. All 92 frozen Python source hashes match this task's source. The original 35 cell records match the pinned source contract exactly. Media transfer digests, native video decode, full-rate foreground scoring and all ten cuts passed independent review; the native contact sheet was visually inspected on Mac.

Local verification: 217 Isaac tests, eight expected Torch skips; Black passed all 16 task Python files and diff checks passed. All six new expanded adapter tests and six legacy checkpoint-audit tests passed under Ubuntu's actual CPU Torch environment. The bounded GPU job and exporter exited successfully. Final read-only check: 4090 idle (17 MiB); existing 4070 workload active and untouched. Existing services and the camera scout were preserved. Approximately 4.9 GiB remained on Ubuntu storage; plan storage before a substantially larger experiment.

Run: `output/expanded-training-20261102-v4`; export: `output/expanded-training-20261102-v4-export`; frozen source: `studies/expanded-farm-20261102-v6`. Raw RGB, telemetry, intermediate policies, diagnostic runs and source archives remain outside Git. [Method and reproduction commands](../../sircus/isaac/EXPANDED_FARM_POSES.md).

This remains privileged endpoint-shot PPO with scripted transit and initialized releases. Perception, grasping, learned joint paths, disjoint-layout evaluation, global workspace coverage and physical validation are not established. Historical baselines, bucket measurements and scoring remain intact. No hardware, Viam, audio or service deployment occurred.
