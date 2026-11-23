# Concurrent board-pose planning

The current planner can spend its entire 30-second allowance on a shot whose
ball path is blocked by an arm. This experiment lets a second CPU worker try
other board poses while the original search continues. Both share the same
deadline. A replacement must pass arm-pose and path checks, predict at least
5 mm of ball-to-arm clearance, and improve on the final original shot.

The test uses five previously missed bucket positions and one known successful
position. Each planner selects one shot per position; each checked shot is then
repeated three times at 64 simulation substeps. These are selected development
cases, not new positions or a table-wide capture-rate estimate.

## Saved planning decisions

[Planning diagnosis](planning-diagnosis.json) records 69 completed alternative
setup checks, 23 admitted arm configurations and **zero selected replacements**.
All 23 admitted alternatives still predict arm obstruction: their ball-to-arm
clearances range from −74.09 to −20.74 mm. Passing an arm setup check means the
arms can assume the poses; it does not mean the released ball clears them.

All six selected commands match their independently planned baseline commands.
The parallel search consumes its full 30 seconds even where the baseline
finishes earlier. Native outcomes remain pending until all drops, controls and
the final independent audit complete.

The frozen source is `de3ecec18cb7d7c491ef152a43c4cafcffc8871e`, study
`parallel-board-v2`. All 55 focused tests and GitHub CI passed; the preparation
audit passed 34,924 checks. The earlier v1 preparation found an auditor subset
bug and ran zero native drops. Its source and evidence remain preserved. V2
uses a corrected verifier and freshly prepared decisions; physics, models,
shot libraries and scoring are unchanged.

## Interpretation and next step

Scheduling alone has not produced a clearer shot in these saved plans. The
next useful candidate-generation change is to start from nearby, repeatedly
successful board poses and joint configurations, then recheck them for the new
bucket position. Preserve diversity in height, tilt and drop gap instead of
ranking every candidate from one blocked starting shot by its predicted aim.
Transferred settings still need independent simulation drops; nearby success
does not establish success at a new target.

See [protocol and reproduction](../../sircus/mjlab/PARALLEL_BOARD_STUDY.md).
This experiment uses simulator geometry and arms held at checked endpoints.
It does not train a model, test driven-arm motion or camera perception, or
change the live planner. The 4070 and Isaac setup are unchanged. Sanitized
Git evidence excludes raw observations, host calibration, checkpoints and logs.
