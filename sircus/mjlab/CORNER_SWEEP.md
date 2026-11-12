# Two locations on the fixed bucket grid

The current robot-base layout has 33 usable bucket locations. All coordinates
below are in the existing table frame.

## Robot contact at (1050, 25) mm

In the audited six-hour follow-up, each of 1,131 releases rebounded off the
board and recorded robot contact before any bucket contact. The first robot
force occurred 187–532 ms after rebound (median 349 ms).

The trace records aggregate robot force. Reconstructing the exact saved joint
pose and ball state in the CPU collision model at that interval identified
overlapping board-arm geometry:

| Board-arm link | Releases |
| --- | ---: |
| 3 | 796 |
| 4 | 252 |
| 2 | 82 |
| 2 and 3 together | 1 |

The same reconstruction matched all 1,131 first board contact intervals and
30 sampled recorded bucket geometry IDs. The analysis verified the hashes
of all 60 input trace archives. Across 389 numeric settings tried with
multiple arm joint configurations, changing the arm pose changed the first
link hit in 48 settings. Every tested configuration still recorded contact.

Gravity-only extrapolation from the pre-contact ball state gave 89 shots
with whole-ball side clearance at the opening. This projects position at entry height
and omits further collisions, spin and retention. In the closest projection,
the ball center crosses 6.45 mm from the bucket center after the recorded
trajectory was intercepted by link 3. This supports testing the ball's
outgoing clearance against the board arm for any further search at this
location.

## Local sweep at (100, -650) mm

The original shot holds the board center at (533.905, -231.173, 306.984) mm,
with tilt 26.088 degrees and a 398.769 mm drop gap. The runner reads the exact
values and joint vectors from the audited parent's preserved trial.

The frozen plan tests 30 single-coordinate offsets and 16 coupled tilt/gap
offsets. The single-coordinate offsets are ±5, ±15 and ±30 mm for each board
position coordinate and the gap, and ±0.5, ±1.5 and ±3 degrees for tilt.
Coupled tests combine tilt offsets of ±1 or ±3 degrees with gap offsets of
±10 or ±25 mm. Both arms must pass the original full-pose and setup-path
checks before each release. Each successful setting selected for confirmation
gets three repeats with the exact same joint vectors. The six available
confirmation slots prioritize the first capture in each offset family.

All 46 candidates passed setup and released the ball. Eight settings captured,
all with rim contact and zero table contact. Each of the six selected settings
captured on all three exact-command repeats.

| Change from the original setting | Fresh repeats |
| --- | ---: |
| Board X +5 mm | 3/3 |
| Board Y +5 mm | 3/3 |
| Board height −5 mm | 3/3 |
| Tilt −0.5° | 3/3 |
| Tilt +0.5° | 3/3 |
| Tilt −1.5° | — |
| Tilt −3° | — |
| Tilt +3° and gap +25 mm | 3/3 |

All six gap-only offsets missed. The original shot and eight discoveries
give nine settings with at least one capture at this location.

The 4090 run completed 88 releases in 789 seconds, including six successful
replays of the original setting, 18 confirmation releases and 18 formal
controls. Both control blocks passed. The final independent CPU audit passed
39,556 checks and rescored all 88 dense traces.

The first audit compared the full archived trace descriptor with a shorter
clock summary. The corrected audit verifies the clock fields and the complete
parent identity separately. Both receipts and the original run are preserved.
The passing audit SHA-256 is
`8a3a842a15bf60a1c6231d4d068686abc6fc84e6a5d7b2abc1ae169aa5fff88c`.

Run in the pinned native environment using the complete audited parent,
matching scene assets, shared 4090 lease and a fresh output directory.

```sh
CUDA_VISIBLE_DEVICES="$AUTHORIZED_4090_UUID" python -m sircus.mjlab.corner_sweep \
  --parent-study "$PARENT_STUDY" --parent-audit "$PARENT_AUDIT" \
  --reference "$NATIVE_REFERENCE" --assets "$SCENE_ASSETS" \
  --output "$CORNER_OUTPUT" --gpu-lock "$SHARED_GPU_LEASE"

CUDA_VISIBLE_DEVICES='' python -m sircus.mjlab.corner_sweep_verify \
  --study "$CORNER_OUTPUT" --parent-study "$PARENT_STUDY" \
  --parent-audit "$PARENT_AUDIT" --assets "$SCENE_ASSETS" --audit "$CORNER_AUDIT"
```
