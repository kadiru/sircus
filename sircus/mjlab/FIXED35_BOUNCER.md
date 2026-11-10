# Saved bouncer-arm alternatives on three remaining cells

The completed six-case run preserved all 29 known recipes and found one more:
30/35 fixed locations now have confirmed recipes. The frozen learned policy
still scores 24/35. This follow-up is bounded recipe development, not training.

The prior two admitted branches at each of the twelve remaining numeric poses
changed the dropper wrist while keeping the same bouncer joints. This pass
examines the saved, endpoint-valid bouncer alternatives that were not flown.
No new IK optimization, numeric shot changes, bucket changes or guard relaxation.

For each pose, use the lowest admitted old branch slot as the matched source.
Filter its saved endpoint candidates to exact equality with that source's
six dropper joints. Exclude bouncer branches equivalent within 0.05 radians
under periodic wrapping; deduplicate equivalent bouncer branches. Select at
most two: a proximal change in the first three bouncer joints first, then a
wrist-only change if available, otherwise the next distinct proximal change.
Every candidate keeps its exact saved joint vector and must pass the unchanged
endpoint and setup-path admission before a native drop. Missing candidates
remain explicit; no new solver fills them in.

Freeze all candidate commands before flight. Search only grid 03-04, 03-05 and
04-06, round-robin, with at most 24 candidate requests. A capture receives three
fresh repeats of its exact admitted branch. Stop that cell after 3/3 captures
or three failed confirmation groups. Replay all 30 successful recipes unchanged
and bracket the study with 18 original direct/table-assisted/negative controls.
The maximum is 24 search + 27 confirmation + 30 preservation + 18 controls =
99 requests; actual candidate availability may lower the count.

Native64 and v3 scoring remain unchanged: board bounce, whole-ball entry and
one continuous second inside; table/rim assistance counts. Robot, floor and
unattributed contacts invalidate capture; invalid physics stays separately
reported. Nominal repeats do not establish noise or timestep robustness.

The worker is limited to 3,600 acquired seconds, stops new search rounds at
2,400 seconds, and reserves 900 seconds for independent CPU audit. Queue limit
600 seconds; total supervisor limit 5,400 seconds. Use the 4090 under its shared
lease; leave the 4070, Isaac and running services unchanged. No UI update,
hardware access, policy refit or promotion. Raw evidence and prior results remain
immutable; the independent audit reconstructs candidate selection and execution.

Frozen parent six-case report:
`629360baed3cd23261c125679202b8d129ffb28b9cec0c1c8709d0ef19eeeb32`;
independent audit:
`ef56098a64bbd640a0c2e72aeaea1d10e7da00f853f1b9bb4b373d06c85c28e7`
(170,755 checks). The two fixed-base overlap cells remain excluded.
