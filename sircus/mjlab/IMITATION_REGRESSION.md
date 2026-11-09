# Six imitation regressions: one frozen action-midpoint test

The corrected-imitation comparison captured **38/70**, versus **40/70** for the
unchanged current policy on the same new positions. It gained four captures and
lost six. This study investigates those six lost captures. Their inspected
positions are now development data; the historical scores stay unchanged.

## What the saved evidence shows

| Context suffix | Observed corrected-policy failure | Matched current result |
| --- | --- | --- |
| `r0-c0-n0` | No dropper IK branch found; 14 board branches | Capture |
| `r0-c1-n0` | No board IK branch found; 8 dropper branches | Capture |
| `r0-c6-n1` | Board contact at 0.1167 s, robot contact at 0.1393 s; no whole-ball entry | Table-assisted capture |
| `r1-c1-n0` | Near-rim then far-rim contacts; no whole-ball entry | Capture |
| `r1-c2-n0` | Robot contact at 0.1468 s, followed by entry and one-second retention; route remains disallowed | Capture |
| `r3-c0-n1` | No board IK branch found; 6 dropper branches | Capture |

The full prefix is `corrected-imitation-test-`. Finite IK refusal is not proof
of unreachability. Recorded force channels identify robot contact, not a
particular link without separate reconstruction. All admitted measured board
poses were settled; the saved failures followed changed shot configurations,
not observed board drift. That association does not identify a unique cause.

The authenticated CPU diagnosis retains the exact actions, branch proofs,
measured board/release poses, contacts and recorded ball paths for each pair.
It independently rescores the nine admitted saved flights. The original
55,633-check native proof remains the source validity evidence.

## The single intervention

For each of the six contexts, use:

```
midpoint = corrected_action + 0.5 * (current_action - corrected_action)
action = [board_x_m, board_y_m, board_height_m, tilt_deg, drop_gap_m]
```

This is one fixed joint correction in task coordinates. It changes all five
coordinates together, including derived target-facing yaw and absolute release
height. It tests whether reducing action drift recovers the recipe; it does
not isolate the effect of a particular coordinate, alter either learned model,
or search new weights after seeing a result.

Each context has three variants, each requested three times:

- `current_exact`: preserve the successful current recipe and admitted joints.
- `corrected_exact`: preserve the corrected policy's recipe and original
  admitted joints or cached finite refusal.
- `blend50`: the prespecified midpoint, with a new strict setup proof.

The **54 diagnostic requests** are bracketed by **18 formal controls**, for
**72 total slots**. Each control block has three direct captures, three
accepted table-assisted captures and three expected misses. Formal controls
retain the original 17-source namespace; diagnostics use the unchanged
19-source geometry bank. Every slot is frozen before new outcomes, including
nine cached original refusal slots. Those slots are not repeated independent
IK searches and never release a ball from placeholder zero joints.

A midpoint receives one strict proof under unchanged solver seeds, geometry,
clearance and sampled setup-path checks, reused for its three nominal
requests. Every admitted diagnostic's measured joints and board pose are
compared with its matched successful current recipe. Declines have no native
pose or recorded flight. Successful source recipes are preserved regardless
of the new outcome; no policy is promoted or retrained.

## Scoring, video and execution

Native dynamics remain fixed at **64 substeps** with version 3 scoring: measured
board bounce, whole-ball entry and one continuous second retained. Rim and
table assistance count. Robot, floor and unattributed contacts disallow the
route. Containment after robot contact must stay visibly distinct from a valid
capture; setup refusals and invalid physics remain separate outcomes.

The UI shows exact configurations, repeat outcomes, proof evidence and saved
native replay. Video uses recorded states and admitted held endpoints; it does
not invent arm transit or run new physics. Refused variants are labelled as
having no flight. Native results require the independent audit before final
publication; nominal repeats do not establish noise or timestep robustness.

Run `sircus.mjlab.imitation_regression_pipeline` with authenticated
`--parent-study`, `--parent-audit`, `--diagnosis`, the committed
`--rule`, `--reference`, `--assets`, a fresh `--output` and existing `--gpu-lock`.
The parent is `corrected-imitation-v1`; the reviewed CPU diagnosis is
`imitation-regression-diagnosis-v1`.

Only the 4090 UUID may be acquired: at most 900 seconds queued, 3,600 acquired
worker seconds and 6,300 total pipeline seconds, with 1,800 seconds reserved
for the independent CPU audit and at least 8 GiB free storage. Queue state is
reported separately from acquired work. A busy-GPU timeout stops only this
owned task and preserves its evidence; it does not interrupt another job.

The 4070 workload, Isaac setup, services, existing policy weights, physical
hardware and prior evidence stay unchanged. Raw traces, generated artifacts,
logs and host configuration remain outside Git.
