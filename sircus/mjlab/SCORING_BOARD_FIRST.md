# Board-first capture, version 4

This opt-in task definition follows the user's rule: the ball must first rebound
from the bouncer board. It may then contact the table, robot bodies, bucket,
floor, or other surfaces and still succeed. Success requires observed whole-ball
entry through the bucket mouth and one continuous second inside.

`board_first_score.py` and its independent oracle use score version
`sircus-board-first-native-capture-v4`, route version
`sircus-board-first-board-route-v4`, and reward version
`sircus-board-first-shot-reward-v4`. Historical v2/v3 source and results are not
changed. `board_first_env.make_environment` explicitly opts into the adapter;
importing it does not start physics or change a running policy.

## What the recorded evidence proves

Native trace time zero is the released ball state. Six force channels record
board, bucket, table, floor, robot and unattributed contacts. A force row is an
interval, not an instantaneous sample; a channel is active only above 0.01 N.
Only intervals before the unchanged scoring terminal are counted.

The first active contact interval must contain **board only**. The existing
kernel confirms a board rebound at the first subsequent observed state with
outward board-normal velocity above 0.05 m/s, within 0.05 seconds of the last
board-contact end and after a board-free interval. No non-board contact interval
may start before this confirmation timestamp. An interval starting exactly at
that timestamp is subsequent and permitted.

A simultaneous first board/other contact or another contact between board touch
and rebound confirmation has unresolved causal order. It is not accepted as a
confirmed board-first route. This conservative evidence check prevents an
outward velocity caused by a robot from being attributed to the board. It does
not establish that every rejected ambiguous trajectory physically violated the
user's intended rule. Saved channels cannot resolve order within one interval.

After confirmed board rebound, **no surface is forbidden**. Surface flags
report contact presence, not proof that a surface caused an additional bounce.
`table_assisted_capture`, `robot_assisted_capture`, and `other_assisted_capture`
can overlap. `route_class` partitions accepted captures into direct, table,
robot, other, or mixed contact routes. Direct means no external contact beyond
board/bucket; a rim-assisted capture can still be direct.

## Unchanged containment and physics

The calibrated v2 containment kernel remains authoritative: downward mouth
entry, whole sphere below the rim, calibrated wall/base contact tolerances,
continuous one-second retention, and partial-exit resets. Invalid physics still
rejects a shot. The original capture/fall/invalid/end-of-record terminal is
unchanged; a saved recording cannot prove a later rebound after its terminal.
No material parameters, collision geometry, solver timestep, arm admission,
or path checks change in this scorer.

`success` preserves the kernel's retained-capture fact. `route_success` and
`complete_success` are the new task result. `board_first_status` explains the
order finding; `route_failure_reason` distinguishes failed order, invalid
physics, and lack of retention. `forbidden_contact_steps` now counts only
non-board intervals before rebound confirmation (all non-board intervals if
there is no confirmation), never subsequent robot/table contacts.

## Reward and data preservation

Reward weights are unchanged: 0.15 for measured board rebound, up to 0.5 for
proximity to the opening, up to 0.5 for retained time, 3 for accepted capture,
0.3 for clean mouth entry, and 0.1 for one board-contact episode. The previous
0.5 forbidden-contact penalty applies only to pre-confirmation non-board
contacts. Later contacts do not remove capture bonuses. Invalid setups/physics
retain reward -1. The one-board bonus does not mean only one physical bounce.

The environment adapter keeps `historical_pre_board_first_score` alongside the
new score and verifies unchanged capture facts, trace identity, and independent
oracle agreement. Re-scoring does not generate new drops or erase earlier
labels. Training must use the experiment's original TRAIN/VAL partition;
rescoring historical TEST results does not make them training examples.
