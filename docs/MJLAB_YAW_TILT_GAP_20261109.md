# Farther bucket-facing shot — tilt/drop-gap integration handoff

November 9, 2026. Simulation-only development search at native64; no policy
training or promotion. [Complete result table](../baselines/mjlab-yaw-tilt-gap-20261109/README.md)
and [frozen protocol](../sircus/mjlab/YAW_TILT_GAP.md).

## Result

The farther yaw-corrected layout `corrected-imitation-test-r0-c6-n1` now has
**ten recipes with 3/3 fresh captures each**. The baseline retains zero captures
in three repeats. The 49-point tilt/gap grid produced 35 native releases,
14 finite IK declines and ten discovery captures; all ten received all three
prespecified confirmation repeats (**30/30 captures**).

Fixed scene: board center **(653.661, −203.412, 449.818) mm**, bucket X/Y
**(1107.027, −536.266) mm**, horizontal distance **562.434 mm**, dropper world
rotation `Rz(−36.285503°) @ diag(1,−1,−1)`. Only board tilt and gap change;
new arm joints realize those poses, with original strict checks retained.

| Grid | Tilt from horizontal | Drop gap | Ball-center release height | Fresh captures |
| --- | ---: | ---: | ---: | ---: |
| 02 | 24.0000° | 115.650 mm | 602.968 mm | 3/3 |
| 03 | 24.0000° | 135.650 mm | 622.968 mm | 3/3 |
| 08 | 26.0000° | 95.650 mm | 582.968 mm | 3/3 |
| 09 | 26.0000° | 115.650 mm | 602.968 mm | 3/3 |
| 15 | 28.0000° | 95.650 mm | 582.968 mm | 3/3 |
| 16 | 28.0000° | 115.650 mm | 602.968 mm | 3/3 |
| 22 | 30.3307° | 95.650 mm | 582.968 mm | 3/3 |
| 29 | 32.0000° | 95.650 mm | 582.968 mm | 3/3 |
| 36 | 34.0000° | 95.650 mm | 582.968 mm | 3/3 |
| 43 | 36.0000° | 95.650 mm | 582.968 mm | 3/3 |

Gap is **ball-bottom Z minus board-center Z**; release-ball center includes
its 37.5 mm radius. At the original 30.3307° tilt, increasing gap only **20 mm**
to **95.650 mm** was sufficient for 3/3 captures in this layout. This is a tested
recipe, not a guaranteed continuous success window.

All 40 discovery/confirmation captures include table contact; no new grid or
confirmation release contacts a robot. For grid 22 repeat zero, the saved route
has board **0.1330 s** → table **0.4777 s** → bucket **0.7208 s** → whole entry
**0.8770 s** → one-second retention **1.8770 s**. Table/rim routes are permitted
by unchanged v3 scoring. The label alone does not identify a causal bounce;
use the native trajectory and timing evidence when describing the route.

## Verification and preservation

- **106 requests / 92 native releases / 14 finite IK declines**, no invalid physics.
- **18 formal controls passed**, including six intentionally colliding misses.
- Preserved farther-success and nearer-yaw references remain **6/6 captures**.
- All fourteen grid refusals occur at gaps 195.650/235.650 mm; finite failure is
  not proof that those poses are unreachable by every solver or joint branch.
- Independent audit: **50,615 passed checks**, including plan/selection lineage,
  strict geometry/full-rotation proofs, native reset FK, dense native scoring,
  source/physics identity and complete request accounting.
- Worker **2000.63 s**, finite 3,600 s worker cap, native64 only; no noise,
  128-step replay, new PPO, imitation fit or policy activation.
- Model weights, successful parent recipes and historical **40/70 versus 38/70**
  policy results remain unchanged. This new tuning uses an inspected development
  layout; it is not a held-out comparison or improvement of the frozen policy.

Producer/UI commit **`c9d0248743790e862cd8053f1681f4ab24867e31`** is pushed in
[draft PR #116](https://github.com/kadiru/sircus/pull/116), stacked on draft #114.
The small [baseline summary](../baselines/mjlab-yaw-tilt-gap-20261109/summary.json)
contains processed outcomes and shot settings; dense samples, logs, runtime
configuration and media stay private and outside Git.

## UI, video and remaining boundary

`#yawTiltGap` displays every discovery proposal, exact reference and confirmation
outcome, with separate refusal/contact/entry/capture accounting. Its recorded
comparison selects the first reference and rotated-baseline repeat, then the
first up to three search-selected recipes in grid order at fresh repeat zero.
It never chooses a later successful repeat to hide a failure. All ten recipes
remain visible in the results table even when only three appear in the movie.

The deployed panel passed visual/scoped-DOM verification with a 226,292-byte
snapshot. The **55.8-second, 1,674-frame** comparison completed on CPU Mesa
llvmpipe, fully decoded, and played in the browser at 1280×720 with no media
error. Seventy-two preview frames were generated; representative contact views
were inspected. [Video provenance](../baselines/mjlab-yaw-tilt-gap-20261109/video.json)
records the exact clips, hashes and playback check. No extra native simulation
steps or interpolated trajectories were introduced for the movie.

Final read-only checks confirmed all 11 inventoried files unchanged outside the
owned HTML/CSS blocks, the original viewer process, and the exited 4090 worker.
Only the Mac SSH tunnel was restored. The earlier post-study check also matched
the original 4070 fountain process. During publishing, the fountain appeared
under a new PID; this task issued no stop/restart command or configuration
change. Therefore final process identity is explicitly recorded as changed,
while the final historical-file/viewer checks still pass. No viewer or hardware
restart was performed, and no 4070 or Isaac workload was launched by this task.

The simulation holds admitted arm endpoints during flight; there is no validated
moving-arm transit, real grasp/opening sequence or physical deployment. Any next
training step should retain these successful recipes, decide explicitly how yaw
enters release behavior, and evaluate frozen models on genuinely new positions.
No further simulation or learning job is launched by this record.
