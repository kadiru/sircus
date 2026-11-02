# Twelve original bank holes: development diagnosis

These positions came from the completed relative-bank study's test set. Once
selected for targeted search, they become **development targets**. They cannot
serve as untouched evaluation layouts for the expanded bank or retrained policy.

| Original test index | Bucket X (m) | Bucket Y (m) |
|---:|---:|---:|
|8|1.2138862453|−0.1562874533|
|9|1.1798593677|−0.5817071648|
|10|1.1155016613|−0.1715135384|
|12|1.1909716579|0.0473628572|
|19|0.1157531523|−0.5274625131|
|21|0.1627525547|−0.6365356467|
|23|1.0821769084|−0.5174710228|
|26|1.1532532351|0.2184567237|
|35|1.2174065448|−0.3998856716|
|39|1.2199134135|0.1802220024|
|41|1.2372435346|−0.6134795749|
|48|1.0052579293|−0.4646715157|

## What the original masks actually establish

Across their 3,072 fixed bank slots,2,716 failed action bounds/minimum distance,
355 failed the loose 842 mm wrist outer bound, and one passed those filters but
failed native endpoint collision checks. Eleven layouts had **no candidate
reaching the IK solver**. Thus these holes do not establish an IK-solver defect,
let alone that a successful bounce is physically impossible.

The original 16 teachers span roughly 170–441 mm board-to-bucket distance,
246–430 mm board height and 96–351 mm drop gap. Their translated/rotated placements
do not explore independent height/energy combinations at the table edges. Fixed
H/gap combinations can exceed dropper wrist reach even when another lower-board
configuration could place both arm endpoints.

## CPU-only candidate checks

New proposals retain original bounds by default and deliberately sample lower
board centers and gaps near the dropper's geometric limit. A seeded scout of
128 proposals at each of the 12 positions produced 109 native endpoint-qualified
configurations across **all 12**. Of the 1,536 proposals,1,319 remained unresolved
by finite full-pose IK and 108 failed native endpoint collision tests. This is
endpoint evidence only; no ball-flight outcome is implied.

An explicitly versioned wider family permits board Y down to−500 mm, board height
down to 120 mm, tilt 8–60° and gap 50–650 mm. A separate 128-per-target CPU scout yielded
86 qualified endpoints across all 12 positions,43 outside the original action
bounds. Wider bounds did not automatically increase the valid fraction and do
not prove that any of those throws will capture.

The new CPU audit reproduces original-bound native acceptance exactly: all
three known endpoint-valid control poses retained identical masks, joint
branches, FK errors and rejection reasons. Joint limits, full native transforms,
bucket collider placement,2 mm/0.5° pose gates and collision gates remain intact.
The old modules, reports, checkpoints and physics/scoring definitions are unchanged.

## Interpretation for the next experiment

Simulate qualified endpoints, confirm newly discovered captures independently,
then freeze a bank. Compare uniform selection and the learned selector on the
same new held-out positions. Keep the 12 targeted development results separate.
Report empty-bank masks, endpoint rejection, released misses and qualified
captures independently; an endpoint is not a successful drop-bounce trajectory.
