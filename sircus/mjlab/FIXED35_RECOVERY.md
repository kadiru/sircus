# Recovery search on the 11 unresolved fixed-grid positions

The preserved current policy captured 24 of the 35 canonical grid locations.
This finite study searches only the remaining **11 locations**, with their
bucket coordinates unchanged. It does not train a new policy or add placement
perturbations.

The original 11 requests comprise eight released noncaptures and three setup
declines. Two declines were fixed-base clearance refusals and one was a finite
IK refusal. The two fixed-base clearance blockers stay explicit and are not searched in this pass; the other nine receive a bounded search. The conservative clearance test is not a full mesh-overlap proof. These are distinct outcomes: a declined request had no ball flight,
and a finite search refusal does not prove a pose impossible.

The frozen plan contains 80 proposals for each searchable location, offered
round-robin. Each captured discovery receives three fresh nominal repeats,
up to three candidate recipes per location. Search stops for a location once
one recipe captures all three repeats. The maximum is 720 search requests,
81 confirmation requests, 24 unchanged successful-parent replays and 18 formal
controls: **843 requested slots at most**. Actual execution can finish earlier.
Search starts no new rounds after 5,400 acquired seconds; the owned worker is
bounded to 7,200 seconds, including the remaining repeats and controls.

## Evidence and acceptance

The parent checkpoint, 24 successful recipes, original scene and historical
results are preserved. Both the original diagnosis and new search retain
bucket-facing gripper rotation, full-pose IK/clearance admission, native64
physics and version 3 scoring.

A capture requires a measured board bounce, whole-ball entry and one continuous
second retained. Table and rim assistance count. Robot, floor and unattributed
contacts invalidate capture; invalid physics is reported separately. New
successful recipes receive three fresh nominal repeats. A recipe is marked
confirmed only when all three fresh repeats capture; any failed repeat stays
visible. Confirmation is deterministic nominal repetition, not noise or
128-substep qualification.

Discovery and repeated confirmation are development search. Finding a recipe
at an inspected location does not change the policy's historical **24/35**
score, constitute a fresh held-out test or establish all-table generalization.
No evaluation result is converted into a training update in this task.

## Interface and execution boundary

`#fixed35Recovery` shows 11 compact case cards: original failure, discovery
counts, selected exact board X/Y/height, tilt and drop gap, and fresh confirmation
outcomes. The 24 successful parent recipes appear separately. The full ledger
is collapsed by default. An independent bound audit must pass before new
results appear as completed; queued, running, stopped or unaudited work remains
explicitly provisional.

The finite worker uses only the 4090 with the shared GPU lock and bounded
queue/execution time. The 4070 workload, Isaac setup, running services and
physical hardware are outside this task. Results publish additively, preserving
historical panels and the existing runtime report. Raw native traces, generated
configurations and logs stay outside Git. Policy promotion and another training
run require a subsequent task.
