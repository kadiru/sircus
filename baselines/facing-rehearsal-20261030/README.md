# Bucket-facing operator rehearsal: small synthetic diagnostic

This is separate from the published 33/48 frozen-policy benchmark. The new
`facing_learned` variant makes one existing frozen prediction, then applies an
explicit geometric heading constraint. `facing_search` uses32 numerical proposals
per scene. Neither method changes its selected action during the five fresh-seed
release evaluations. No hardware or native collision path was tested.

The three layouts were declared in `sircus/studies/run_facing_rehearsal.py` before
running; planning seed42, evaluation seeds1001–1005; default tilt interval5–75°.
These are three layouts with repeated noisy releases, not15 independent scenes.

| Layout | Board center XYZ mm | Bucket XY mm | Bounded search | Constrained frozen prediction |
|---|---|---|---:|---:|
| Front |600, −150,350|600, −400|5/5|5/5|
| Left |590, −170,360|530, −430|5/5|5/5|
| Right, farther |620, −130,345|700, −420|5/5|0/5|

The frozen prediction's right/farther failure was retained (`no_entry`). No fit,
fallback optimization, exclusion or baseline overwrite followed these outcomes.
The small sample supports workflow debugging, not a reliability or generalization
claim. All capture outcomes use board bounce + whole-ball entry + continuous1s
retention. Bucket reported175mm height interpretation remains unresolved.

Nominal front search example: heading−90°, tilt30°, board-local U=V=0mm,
gap160mm, release ball center[600,−170,544.64]mm. These are numerical ball-center
coordinates with provisional ball/TCP and board-mount transforms; do not copy them
into a physical arm controller as calibrated targets.

See [results.json](results.json) for every selected action, planner source,
release coordinate, planning outcome and fresh release result. Reproduce with:

```sh
sircus/simulation/.venv/bin/python sircus/studies/run_facing_rehearsal.py \
  --budget 32 --trials 5 --output experiments/facing-smoke-001.json
```
