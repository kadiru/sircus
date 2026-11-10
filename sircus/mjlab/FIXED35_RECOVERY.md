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

## Audited result — 9 November 2026

The study completed **441 requests**: 384 search, 15 fresh confirmation,
24 unchanged-parent preservation and 18 controls. Five discovered recipes
captured **3/3 fresh repeats each**, and all **24/24 parent replays** captured.
Verified recipe coverage is therefore **29/35 fixed locations**. The unchanged
policy's historical score is still **24/35**; no policy fitting or promotion
occurred.

Four searchable locations exhausted their 80 proposals: grid 01-05, 03-04,
03-05 and 04-06. Grid 03-00 and 03-06 remain fixed-base clearance blocked.
There were three invalid-physics requests, all excluded from valid capture
evidence. The study stopped at finite proposal caps, without time truncation.
Worker time was 4,105.75 seconds and pipeline time 4,135.29 seconds.

The independent bound audit passed **141,041 checks**; final isolation passed
**31/31 checks**. All pre/post control gates passed. The owned 4090 worker exited,
and the final check found the 4090 idle with the 4070 workload unchanged. Exact
recipes, per-case failures and evidence hashes are in the
[completed handoff](../../docs/MJLAB_FIXED35_RECOVERY_20261109.md) and
[versioned summary](../../baselines/mjlab-fixed35-recovery-20261109/summary.json).
Frozen producer: `7bcba3980a1baf24d6ac2505e35189b95daead64`,
[draft PR #120](https://github.com/kadiru/sircus/pull/120), based on PR #119.

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
