# Why four placements changed — November 24, 2026

The [50-placement comparison](../mjlab-expanded-selector-20261124/README.md)
finished at **44 captures for each planner**. The expanded planner gained two
placements and lost two. This follow-up examines both saved drops at those four
placements: **eight historical flights, no new trials or training**.

**Four paired movies are published in the mjlab UI at
`#expandedSelectorReplays`.** Each compares the two saved flights with the same
camera at **0.25× speed**. Both arms stay at their recorded endpoints; only the
recorded ball samples are played back. All four movies passed live-browser
playback verification. [Publication receipts](replay-publication.json) record
the movie hashes, camera settings and playback checks. All 13 older media entries
are preserved, bringing the total to 17. The 4070 workload and Isaac setup remain
untouched.

| Bucket X/Y (mm) | Current planner | Expanded planner | Observed difference |
|---|---|---|---|
| 616.5 / −504.4 (04) | Direct capture | Miss | Expanded: table rebound, then near-side bucket contact while rising at 1.56 m/s; no entry. |
| 1179.2 / −488.1 (10) | Miss | Table-assisted capture | Current: near-side bucket contact while rising at 0.84 m/s. Expanded crossed above the opening and retained the ball. |
| 854.6 / −408.6 (19) | Table-assisted capture | Miss | Expanded: table rebound, then near-side bucket contact while rising at 1.43 m/s; no entry. |
| 1178.8 / 216.5 (47) | Miss | Table-assisted capture | Current: robot contact at 0.553 s. Expanded: two board-contact intervals, then table rebound and capture, without robot contact. |

**04, 10, 19 and 47 are row IDs in that 50-placement test**, not locations on the
older 33-point training grid. Capture still requires the board rebound first,
whole-ball entry, and one continuous second inside. Later table or robot contacts
are permitted. The four misses had no whole-ball entry; none was rejected merely
for an allowed later contact.

## What the planners chose

Heights are measured from the simulated table plane; tilt is from horizontal.
Drop gap is the configured ball-bottom height above the board-center plane.
Ball-center release height is board height + gap + the 37.5 mm ball radius.
The table below does not include the simultaneous board XY and joint changes;
the complete paired settings and deltas are in the [diagnostic summary](diagnosis.json).

| Case | Planner | Board height (mm) | Tilt | Drop gap (mm) | Board–bucket XY distance (mm) |
|---|---|---:|---:|---:|---:|
| 04 | Current | 335.3 | 21.80° | 145.6 | 232.2 |
| 04 | Expanded | 502.6 | 12.14° | 179.8 | 428.2 |
| 10 | Current | 433.8 | 31.73° | 133.3 | 598.8 |
| 10 | Expanded | 426.9 | 29.14° | 193.0 | 727.6 |
| 19 | Current | 483.6 | 22.33° | 144.4 | 589.7 |
| 19 | Expanded | 502.6 | 12.14° | 179.8 | 438.4 |
| 47 | Current | 373.7 | 16.02° | 253.5 | 692.3 |
| 47 | Expanded | 539.8 | 17.82° | 65.7 | 549.0 |

Both expanded losses reused the same historical source setting, whose recorded
capture included robot contact. The new shots had **no robot contact**, and that
old source does not guarantee capture at a new bucket location. Their selector
scores were high (0.980 and 0.956), and their flight forecasts predicted positive
opening margins (33.9 and 28.8 mm). The saved flights instead contacted the bucket
on the rising approach and reversed horizontal direction before entry.

Case 47 shows the opposite forecasting limitation: the expanded shot captured
despite a negative predicted opening margin. It touched the board at **0.1117 s
and 0.2600 s**, then the table at **0.6126 s**, entered at **0.9683 s**, and
completed retention at **1.9683 s**. The forecast models one board rebound and
one table rebound; it does not model that second board contact.

## What is established, and what to test next

- Native force intervals establish contact timing and categories. For current
  Case 47, CPU geometry at saved ball states uniquely overlaps
  **Arm14/link3**. That link name is inferred, not a recorded native contact ID.
- Rim crossings interpolate adjacent recorded states. A crossing short of the
  bucket before a planned table rebound is normal; it is not itself a miss.
  Actual entry and retention determine the outcome.
- Several settings changed together. These pairs identify failure mechanisms
  to investigate, but do not prove that one height, tilt or gap change caused
  the improvement.

**Next targeted test:** check the complete table-rebound approach against bucket
collision geometry before accepting a favorable descending opening prediction.
Do not ban table or robot rebounds. Preserve the current planner and successful
shots, and measure whether this contact-aware check rejects the three observed
rising-contact misses while retaining successful table-assisted examples.
Second-board rebounds need separate modeling or an explicit forecast limitation.

## Reproduce the saved-evidence diagnosis

The new `sircus.mjlab.expanded_selector_diagnosis` module verifies the parent
report/audit/manifest, selected requests, request packets, dense native columns
and initial states, then independently recomputes v4 scores for all eight shots.
It reconstructs held endpoints using CPU kinematics, with **no physics steps**.

```sh
CUDA_VISIBLE_DEVICES='' /home/kadir/sircus-mjlab/venv/bin/python -m sircus.mjlab.expanded_selector_diagnosis \
  --study /mnt/robot-training/sircus-mjlab/output/expanded-selector-v1 \
  --audit /mnt/robot-training/sircus-mjlab/output/expanded-selector-v1-audit.json \
  --assets /home/kadir/sircus-isaac/assets \
  --output /path/to/new/diagnosis.json
```

Use a new output path; never write into the parent study. Saved-native videos
replay the recorded ball path and held arm configurations, not a new simulation
or moving-arm trial. The parent 44/50 results remain unchanged. No live planner,
4070 workload, Isaac setup, physical hardware or model weights are changed.

## Frozen evidence

Diagnosis/video producer: **`a9c617a35be7ac49d6405508a487fe72e97a2e9b`**.
UI/publisher: **`c8be9e9`**; draft
[PR #167](https://github.com/kadiru/sircus/pull/167).
Final CPU diagnosis completed **773 evidence checks**; all eight independently
rescored outcomes match the audited parent. The full diagnostic is preserved
outside Git at
`/mnt/robot-training/sircus-mjlab/output/expanded-selector-replays-v1/diagnosis.json`.
The repository summary excludes dense states and raw force arrays.

| Artifact | SHA-256 |
|---|---|
| Final diagnosis | `8dc675a0e3314c7fa2a3794d986b31e2e7dd326555edef38b5fc337cecf82f95` |
| Diagnosis module | `3f370cf03e729a572b2633c51a4445d5c9c5e4abbda2ff2ebf764a7480cfd31f` |
| Parent report | `d72966f031f91ad1f231b0b8c9bed3e64f37032c1f0d20b020c22bd041baa0d2` |
| Parent independent audit | `af5f6d4efdba7829bc3fda48a67c28e655a0306ca613abe5ce55659e1afe585a` |
| Parent evidence manifest | `3117645239b4b56ba2a03ea81ccc616435dfea50b6014db864952630aa10a8ad` |
| Parent choices | `2b9a06cded6a0911ae49e60674069307d70cc36389077bb8e318d0762bacdb8b` |

Validation: the full Python suite passed with **3,138 tests run and 301 skipped**;
**500 JavaScript tests passed**. This includes the diagnosis/contact helper
checks; the final frozen Ubuntu CPU diagnosis passed independently. All four
published movies were verified playing in the live browser. Video generation
and playback verification have separate receipts and add no simulated drops.
GitHub CI passed for `c8be9e9` ([run 45491807126](https://github.com/kadiru/sircus/actions/runs/45491807126)).
No training, live planner promotion, service restart, 4070 workload change or
Isaac setup change accompanied publication.
