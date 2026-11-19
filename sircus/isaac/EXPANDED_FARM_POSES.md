# Expanded poses for the 35-table farm

This opt-in experiment preserves all 35 original bucket coordinates, both arm bases, the silver-gray stage and the user's exact camera. It expands the original 400 mm board-to-bucket circle into a variable-distance search. Historical four-action policies and results remain separate.

## Completed training pass

[Results and verification evidence](../../baselines/isaac-expanded-farm-20261102/README.md) record the completed v4 run: 350/350 releases, ten at each table, with zero invalid trials and five PPO updates. There were 111 qualified captures, including 59 clean and 101 single-board-bounce captures. These are on-policy training outcomes, not held-out evaluation or evidence of generalization. The CPU checkpoint/likelihood audit passed against the immutable v6 source.

Native qualification retained 24 configurations at every site, 840 total, with variation in all five action dimensions. The final menu uses nearby configurations on each anchor's IK branch. Nine sites still repeated one projected configuration in all ten sampled policy decisions, so the larger menu should not be confused with demonstrated exploration of every option. The exact camera, original distinct bucket coordinates, arm bases, geometry uncertainty and capture scoring are unchanged.

## Geometry and policy

The new actor chooses five quantities: bucket-relative board bearing, horizontal board-to-bucket distance, board-center height, vertical ball-bottom gap and board tilt. Heading toward the bucket remains prescribed. Inputs remain five privileged geometric features; neither the filming camera nor point clouds drive this policy.

| Choice | Bounds |
|---|---|
| Bearing offset | −180° to +180° |
| Horizontal distance | 280–1200 mm |
| Board height | 140–540 mm above the tabletop |
| Ball-bottom gap | 80–560 mm above board center |
| Board tilt | 12–52° from horizontal |

The endpoint search samples a common arm-workspace region, X=380–940 mm and Y=−500–300 mm, and bucket-relative ring proposals. A frictionless range approximation ranks some proposals; seeded broad alternatives remain available. No ball-release rewards or capture outcomes enter pose qualification. Conservative board/table and board/bucket clearance checks precede inverse kinematics. Both arms must satisfy the original joint limits and `1e-4` weighted IK residual; native qualification then uses the unchanged measured pose, normal, speed, drift and dropper-position gates.

Radius is an explicit fifth action. Adding radius-varied prototypes to the old four-dimensional menu would make physically different actions ambiguous. The new task and checkpoint shapes are versioned; old four-output policies cannot be loaded as five-output policies.

## Qualification and transitions

The initial bounded screen found endpoints in 28 cells. Deeper checks found at least two at every site, and the broader menu contained 207 CPU-qualified candidates. Native holds initially qualified at least two poses at 34 sites; refining the remaining site around its measured hold completed 35-site endpoint coverage.

The first driven training diagnostic still released balls at only 31 sites. Independent FK checks on all 350 rows and exact IK-cache checks ruled out joint-index, frame and target-key mapping errors. Large joint changes remained during the rejected transitions; collision or torque failure was not established without transient force/torque instrumentation. Static endpoint qualification does not certify a path.

The subsequent geometry refinement keeps one measured holding configuration at each table and solves small board-height, tilt, XY and gap variations from that configuration's original IK seed. New joint coordinates must stay within 0.35 rad per joint of their anchor; this bounds each pose relative to the anchor, not every pairwise path. The wider search and diagnostic remain preserved. These nearby menus are an initial operating region at each bucket, not exhaustive configuration coverage.

Expanded training waits for qualified cells to meet the existing joint-readiness tolerance within the original bounded wait. That wait can finish with 34/35 cells ready; it does not establish that every transit was ready. Independent final measured settling/release checks subsequently admitted every release in the completed v4 pass. Reaching the wait deadline alone never authorizes a drop. Earlier filming modes retain their foreground/background wait behavior, and transition errors/readiness remain separate from later settling corrections. This is not collision-free path certification.

The run uses ten batches of 35 sampled actions and five PPO updates with the existing reward and full one-second capture evaluator. Invalid attempts stay in training and in the report. Joint travel and ball loading are scripted; no grasping, joint-path learning, held-out evaluation or physical readiness is claimed.

## Camera and evidence identity

`camera-source-contract.json` preserves the exact original native contract (SHA256 `b648bb205e8ffd64018608b3b35b7a620fd8a9713bf26a16dde05000eaf83137`). Its original fingerprint is `98ebe47b48a01b9028747ab03bc913a033d90611e44f03b912e58f41c87ff79a`. This prevents NumPy ULP differences from regenerating different bucket records on Mac and Ubuntu. Static geometry is checked strictly; only tiny numerical differences in regenerated layout arithmetic are tolerated, and the exact frozen records are retained.

The new experiment fingerprint is `63362ef37a93881f769b39d1ee7456335f1cc7bf5db04ffa7e548ce9697455bf`. The original camera JSON remains unchanged, checked against its original source contract. A separate `camera-binding.json` binds it to the new experiment after checking the unchanged scene, 35 bucket identities, display origins and lens. No fingerprint guard is bypassed or overwritten.

## Run in the configured Ubuntu environment

Use the pinned Isaac Sim 5.1 / Isaac Lab 2.3.2 environment and generated assets from [the foundation setup](README.md). Freeze this source directory first. The CPU search needs NumPy/SciPy and the same generated URDFs; the native run uses the existing isolated 4090 container and shared scene lock. Leave other workloads/services running.

Build the new contract with `expanded_farm.make_spec(scene, 10)`, then run geometry-only search with fresh output paths:

```bash
python.sh expanded_pose_menu.py --spec NEW_CONTRACT.json --assets /workspace/assets \
  --legacy-source /workspace/output/cylinder-film-35-v5 \
  --max-checks 24 --accepted 4 --output OFFLINE_MENU.json
```

Use `--previous OFFLINE_MENU.json`, `--cells` for unresolved indices and a larger `--max-checks` to extend a search without overwriting it. `--accepted 6` supplies alternatives for native qualification.

```bash
python.sh film_cylinder_farm.py --headless --enable_cameras --device cuda:0 \
  --appearance silver --training-camera-json ORIGINAL_CAMERA.json \
  --expanded-pose-menu OFFLINE_MENU.json --expanded-options-per-cell 24 \
  --expanded-qualify-only \
  --output NEW_QUALIFICATION_DIRECTORY
```

`expanded_pose_refine.py` can generate alternatives around individually proven native holds, including a partial native menu. For nearby configurations at all sites, use `--local-only --board-first --max-joint-delta .35 --append 24` and list all requested indices after `--cells`. `--append 24` requests 24 new CPU-qualified alternatives around each anchor; the native run retains up to 24 accepted options per cell with `--expanded-options-per-cell 24`. Its output remains offline until fresh native qualification:

```bash
python.sh expanded_pose_refine.py --offline OFFLINE_MENU.json \
  --native QUALIFICATION/expanded-menu-qualified.json --assets /workspace/assets \
  --cells 0 1 2 --local-only --board-first --max-joint-delta .35 --append 24 \
  --output REFINED_MENU.json
```

The example refines three indices; list indices 0 through 34 to refine the whole farm. Run the same filming command with the refined menu and a fresh output directory, retaining `--expanded-options-per-cell 24` and omitting `--expanded-qualify-only`, to qualify and then train. The minimum training gate remains two distinct native-qualified configurations per table; requesting 24 sets the larger qualification target used in the final pass. All original per-trial release checks remain active.

After capture, run `action_training_audit.py` with the simulator's Python on CPU, then `action_training_export.py RUN EXPORT --untrimmed` on the host. The exporter verifies five-dimensional action projection, native-menu proof, camera binding, policy likelihood/update chain and full-rate foreground scoring before encoding. All ten batches are retained. The short edit ends after the immediate outcome; the complete captured sequence retains stationary scoring tails. Neither includes unfilmed setup/optimizer waits. [Editing and timing details](ACTION_TRAINING_FILM.md).

Bucket dimensions and the unresolved reported 175 mm height interpretation remain unchanged. No hardware, Viam configuration or audio deployment accompanies this experiment.
