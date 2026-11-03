# Broad shot search and shared-policy learning

This study asks whether a shared policy can rank cooperative board-and-drop configurations for the original 35 distinct bucket sites. The immediate goal is to find successful configurations at the sites that previously scored zero, then compare a longer learned policy against its initial policy and the earlier frozen policy. The exact tabletop, bucket positions, arm bases, provisional physics and capture definition remain unchanged.

The [completed study, per-table scores and exact configurations](../../baselines/isaac-farm-shot-learning-20261102/README.md) record 19 search-successful targets, 8,960 PPO attempts and frozen sampled improvement from 96/560 to 183/560. Greedy success stayed at 10/35, so this is not a dependable deterministic inverse-physics policy. A separate no-retraining probe uses new candidate configurations at the same bucket sites; it is not an unseen-bucket test. Frozen source revisions and partial search evidence remain separate. Earlier results remain in [the expanded farm baseline](../../baselines/isaac-expanded-farm-20261102/README.md).

## The learning problem

Each episode is one shot. A policy sees privileged bucket geometry, chooses a board/drop configuration, and receives a terminal reward after native PhysX contact and capture scoring. This is a contextual bandit trained with PPO, rather than a controller making joint commands throughout the flight. The two arms cooperate through their selected endpoints: the board arm supplies the bounce point and surface normal, while the dropper supplies the release location and falling energy.

We reduce the arm configuration space before learning. The board faces the bucket in the horizontal plane, the ball starts vertically above the nominal board center, and the gripper points down. Inverse kinematics resolves the selected endpoints into arm joints. The policy therefore searches five task variables instead of independently exploring every joint of both arms. Arm grasping, transport paths and reactive interception are separate problems.

| Variable | Nominal bounds | Meaning |
|---|---:|---|
| Board-center height, H | 140–540 mm | Height above the tabletop |
| Release gap, G | 80–560 mm | Initial ball-bottom height minus board-center height |
| Board tilt, θ | 12–52° | Surface tilt from horizontal; heading faces the bucket |
| Horizontal distance, D | 280–1200 mm | Board-center XY to bucket-center XY |
| Bucket-relative bearing | −180° to +180° | Location of the bounce point around the bucket |

The release ball-center height is **H + G + 37.5 mm**, using the current provisional ball radius. The dropper flange target is a further 95 mm above that center. These are different quantities; release gap is not flange height or exact free-fall distance to contact. Board thickness and tilt move the actual sphere/board contact-center height above the nominal board center.

In broad search, board XY is sampled across the original tabletop, whose nominal extent is X=−230 to 1570 mm and Y=−875 to 375 mm. Its center must lie within those bounds with 15 mm margin. The arm-held board is allowed to overhang the table; requiring its entire footprint to stay above the tabletop would wrongly exclude previously demonstrated shots. The actual rotated board bottom must still clear the table plane. The effective board-to-bucket separation also includes the board half-diagonal, outer bucket radius and 20 mm clearance, so the practical lower separation can exceed the nominal 280 mm bound. Both arms must also pass their joint limits, geometric reach bounds and strict endpoint IK; not every numerical combination in the table is physically admissible.

The board-arm IK constrains its normal, leaving rotation about that normal unconstrained. The new proposal filter checks the **actual fitted rectangle's vertical clearance**, not only the nominal target rectangle, so this free in-plane spin cannot put its bottom corners into the table plane unnoticed. Overhang remains allowed. This is an endpoint geometric check, not a whole-arm collision or swept-path certificate.

## Why the previous run was limited

The earlier all-table film established release participation: 350/350 releases and 111 qualified captures. It used only five PPO updates. Its 24-option menus were nearby variations around one holding pose per site, chosen to make driven transitions manageable. Typical menu spans were only 20–40 mm in height/gap, 1–4° in tilt and roughly 3–20 mm in radius. Having all five dimensions technically vary did not mean their broad interaction had been explored.

That actor sampled a continuous Gaussian latent and projected it to the closest qualified menu entry. Many different samples chose the same physical shot. Values at action bounds map near ±7.25 in the inverse-tanh latent space, further distorting exploration of a small local menu. This projection is a valid one-shot policy-gradient construction, but it can waste samples and obscure available diversity.

The historical reward also returned exactly −0.5 as soon as any external table/floor/robot contact was recorded. Most misses eventually hit something external, so a near miss and a distant miss often became identical learning signals. The old reward and results are preserved. The new policy and reward have their own versions rather than silently redefining the earlier experiment.

## Searching the relationship efficiently

The broad proposal bank mixes two outcome-free sources in approximately a 3:1 ratio:

1. **Physics-informed proposals:** sample board XY, board height and tilt, then solve an approximate ballistic reflection for the release gap. Restitution priors of 0.45, 0.55, 0.65, 0.75 and 0.85 spread proposals around uncertainty in the uncalibrated bounce model. Three target ball-center heights—rim minus one radius, rim, and rim plus one radius—cover descending whole-ball entry and the preceding approach.
2. **Independent broad proposals:** sample XY, height, gap and tilt independently with seeded scrambled Halton sequences. These cover configurations that an inaccurate analytic prior might otherwise exclude.

Radius and bearing are derived exactly from the sampled board XY and that table's fixed bucket position. The search retains two previous configurations as controls, then leaves the remaining bank capacity for broad proposals. Seeded sampling gives reproducibility and better space filling than repeatedly drawing a tiny neighborhood. It is a bounded sweep, not exhaustive enumeration of a continuous space.

Proposal version `sircus-physics-informed-shot-sweep-v2` follows the user's energy argument. If the incoming vertical speed is w, a frictionless reflection from a static plane tilted by θ gives approximately:

```text
w² = 2 g × free-fall distance
horizontal speed = (1 + e) w sin(θ) cos(θ)
vertical speed   = [(1 + e) cos²(θ) − 1] w
```

A projectile equation then relates distance, board height and desired descending height to the required incoming speed and release gap. The implemented prior accounts approximately for sphere radius and board thickness. Its three plane-height priors are alternative guides, not proof of whole-ball retention. Friction, finite board geometry, robotic links, spin and detailed impacts are omitted. It supplies proposals, never a success label. The independent broad stream and actual native tests remain necessary.

Cheap geometry and link-length bounds discard impossible proposals before numerical IK. Warm starts come from nearby previously solved endpoints, while the acceptance threshold stays at the original `1e-4` weighted residual. Each admitted fit is independently checked by forward kinematics. The current budget allows **512 endpoint IK checks per site** and retains **up to 128 proposals per site**, including controls. Native tests then apply the original measured position, normal, velocity, drift and dropper-position gates before each release. Search failures and invalid endpoints remain visible in the report.

Every candidate that passes native endpoint checks stays available to training, whether it captured or missed. Training does not receive an oracle that removes unsuccessful shots. A separately identified search-selected best setting is repeated four times per table; that is a development search control, not policy inference or evidence that an unseen target was solved.

## New shared policy and PPO

The new actor scores each available candidate using the same network at every table. Its input is:

- Five bucket features: normalized X/Y, distance to the common reference point, and sine/cosine of the bucket-to-reference bearing.
- Six shot features: sine/cosine of the bucket-relative board bearing, normalized distance, board height, release gap and tilt.

The resulting 11 values pass through a shared `11 → 96 → 96 → 1` tanh network. A masked categorical distribution turns the candidate scores into selection probabilities. Padding and unavailable candidates have zero probability. The final actor layer starts at zero, so exploration begins uniformly over each table's valid options. There are no table-ID embeddings, per-table learned parameters, capture labels or images in these inputs. The critic is a separate shared `5 → 64 → 64 → 1` network over bucket geometry.

This policy learns a ranking relationship between target geometry and candidate shot parameters. At deployment it selects a configuration with one network pass over the qualified candidates. It does not run a capture optimizer at inference. It also does not yet produce an unconstrained continuous solution or prove that the learned ranking transfers to unseen bucket positions.

The completed predeclared run used **256 batches × 35 shots = 8,960 on-policy training throws** and **128 PPO updates**, with two batches per update. Each update uses four epochs, Adam learning rate 0.0003, ratio clipping 0.2, value coefficient 0.5, categorical entropy coefficient 0.01 and gradient-norm cap 1.0. Each shot terminates, so its return is its observed reward; there is no next-state bootstrap or GAE. The entropy bonus encourages exploration of available configurations. Successful actions do not become supervised optimizer labels.

The bounded native sweep can add up to 4,480 development attempts before training, plus separate repeats and frozen evaluations. These are different data uses and are reported separately. Headless learning initializes arm endpoints directly and records no RGB frames, avoiding filming waits and storage overhead. Final release checks and 240 Hz physics scoring remain active. A successful headless endpoint shot does not establish that a driven arm path can reach it safely.

## Reward and capture truth

The original capture scorer is unchanged. A qualified complete shot requires a board bounce, whole-ball entry, and one continuous second retained inside the tapered bucket, with no detected external table/floor/robot route. Partial exits reset retention. Rim-assisted captures count. Clean entry and exactly-one-board-contact capture remain separate quality metrics. With a 177 mm opening and a provisional 75 mm ball diameter, the ball center has only about 51 mm of radial mouth clearance; predicting a rough flight range alone is insufficient.

The new versioned learning reward is:

```text
invalid endpoint: −1
no board bounce:  −0.25 − 0.5 × external-contact indicator
otherwise:
  +0.15
  +0.40 × exp(−closest-opening-distance / 0.12 m)
  +0.25 × min(maximum-retention-seconds, 1)
  −0.80 × external-contact indicator
  +1.00 × qualified-capture indicator
  +0.10 × clean-qualified-capture indicator
  +0.05 × exactly-one-board-contact-qualified-capture indicator
```

The larger external-contact penalty keeps an unqualified external route from earning a positive total, while preserving a gradient between nearer and farther misses. The current closest-distance and retention measurements may include post-contact states; they are not claimed to measure pre-contact accuracy. Capture bonuses still require the unchanged qualified success flag. Every row also logs the historical reward, preserving comparisons under its original definition.

## Evaluation and per-table reporting

The final checkpoint is selected by the predeclared training budget, not by testing several checkpoints on evaluation outcomes. Fresh native episodes evaluate:

- The earlier frozen continuous policy using its own original 24-option qualified bank, when that checkpoint is supplied.
- The new categorical policy before training, using the broader native bank.
- The new categorical policy after all 128 updates, using that same broader bank.

Each policy gets **16 stochastic episodes per table** and **one greedy episode per table**. Initial/final sampling uses reset seeds and unchanged buckets. Stochastic repeats vary selected qualified configurations; they are **not** a depth-noise, physics-noise or object-placement robustness test. One greedy episode is not a repeated reliability estimate. The four repeated search-selected settings are labeled separately.

The per-table report separates development sweep, search-selected repeats, on-policy training, frozen legacy, frozen initial and frozen final outcomes. It records attempts, releases, invalids, qualified captures, clean captures, exactly-one-board-contact captures and both reward definitions. Native candidate counts and the exact search-selected settings are included even where no capture was found. The earlier policy's smaller bank is an additional difference, so a changed score is not attributed to learning alone.

All of these stages use the **same 35 development bucket coordinates**. They establish within-layout progress only. A genuine generalization test must reserve different bucket coordinates before further tuning, freeze the policy, construct geometry-qualified candidates without capture search on those held-out layouts, and evaluate direct policy choices. It should separately vary distance, bearing, board height and contact properties, with declared unreachable/unsafe cases. Those unseen-position results are not available from this study.

## Reproduction in the configured Ubuntu environment

Use the pinned Isaac Sim 5.1 / Isaac Lab 2.3.2 environment and existing generated arm assets. Freeze source before execution, use new output paths, acquire the existing scene lock for native work and leave the 4070 workload and services untouched. The CPU search does not launch Isaac. Generated runtime files, raw observations, logs and credentials stay outside Git.

```bash
# Existing contract preserves the original 35 sites and expanded 5-D bounds.
python.sh shot_sweep.py --spec ORIGINAL_EXPANDED_CONTRACT.json \
  --assets /workspace/assets --seed-native PRIOR_NATIVE_MENU.json \
  --workspace table --controls 2 --candidates 4096 \
  --max-checks 512 --accepted 128 --output NEW_OFFLINE_BANK.json

# Headless study: camera identity is bound, but RGB recording is disabled.
python.sh film_cylinder_farm.py --headless --enable_cameras --device cuda:0 \
  --appearance silver --training-camera-json ORIGINAL_CAMERA.json \
  --expanded-pose-menu NEW_OFFLINE_BANK.json --learning-study \
  --study-sweep-limit 128 --study-train-batches 256 \
  --study-eval-repeats 16 --study-seed 20261110 \
  --study-legacy-checkpoint PRIOR_RUN/policy-final.pt \
  --output NEW_STUDY_DIRECTORY

# CPU-only saved-policy / sampling-likelihood audit after completion.
python.sh farm_learning_runtime.py NEW_STUDY_DIRECTORY
```

The legacy checkpoint expects its original `expanded-menu-qualified.json` beside it. The current parallel CPU search divides cell indices across four workers; merged manifests must preserve every original cell and exact source identity. A post-filtered frozen proposal bank records its own source hash and actual fitted-board clearance checks; it does not relabel the earlier bank as native-qualified. The study command explicitly overrides the CLI's 128-batch default with the planned 256 batches.

## Separate frozen-candidate probe

`farm_transfer_probe.py` qualifies a new offline bank and evaluates the exact learned checkpoint without an optimizer or policy updates. Use `--transfer-probe-checkpoint FINAL_POLICY.pt` instead of `--learning-study`, with the same camera, scene and expanded-bank arguments. The checkpoint's completed budget, sealed contract, feature layout and geometry fingerprint are checked. All natively accepted candidates remain available regardless of capture outcome. `farm_transfer_probe.py RUN` audits frozen weights, categorical probabilities, budgets and summaries on CPU.

This tests new shot configurations at seen bucket locations, not unseen bucket layouts. A bank may contain preserved controls, so novel-proposal outcomes and overall policy scores need separate interpretation. The prepared 39-new-configuration attempt did not launch: its 900-second shared-lock wait expired behind another agent's job. Native validation remains outstanding; coordinate GPU ownership before a new attempt. The primary study is complete and unaffected.

Key artifacts are `farm-study-contract.json`, `sweep-proposal-protocol.json`, `native-qualified-study-bank.json`, `stages/…/batch-….json`, sampling/final policies, `farm-study-result.json`, `progress.json` and `candidate-checkpoint-audit.json`. The audit reloads saved policies on CPU, recomputes categorical sampling likelihoods and verifies logged update linkage and artifact hashes. It does not independently reconstruct optimizer mathematics, replay full contact telemetry or establish held-out performance.

The measured 200×300×5 mm board, 1,337.31 mm base spacing and tapered bucket dimensions remain as recorded. The bucket's 175 mm reported height interpretation, assumed base thickness, ball dynamics, mount rigidity and other physical parameters remain provisional. No hardware, Viam configuration, audio or service deployment accompanies this study.
