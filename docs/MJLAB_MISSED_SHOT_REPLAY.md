# Replaying feasible shots at missed bucket positions

The [100-position comparison](MJLAB_DIRECT_FIRST_COMPARISON.md) recorded
16 ordinary flight misses for the route selector, five positions where
neither route passed setup, and one invalid trajectory. This follow-up
tests whether a different shot was already available at each ordinary miss.

- Reconstruct the original candidates and selections from the audited
  study. Freeze the complete list of 16 missed positions and their
  admitted commands before running the follow-up.
- Repeat the selected shot once at each position. Run every other distinct
  admitted command once, using the archived pose, joints and path proof.
  Candidate IDs that lead to the same command share one replay.
- Record which commands previously ran under direct-first. Compare their
  repeated results against the old outcomes and count newly tested
  alternatives by requested route. A capture requires a board rebound
  and one continuous second with the whole ball inside the bucket.
- For the five setups that failed before release, report whether each route
  produced any candidates and why the arm checks refused each one.
- Use the original 64-substep simulator and the same nine controls before
  and after. The CPU audit reconstructs the command list, verifies the
  archived setup proofs and rescores the recorded trajectories.

The selected positions serve as development evidence. The producer is
`python -m sircus.mjlab.intergrid_candidate_replay_study --help` and the
verifier is `python -m sircus.mjlab.intergrid_candidate_replay_audit --help`.
Use the audited `intergrid-route-fallback-v1` and `fixed35-sixhour-v2`
archives on Ubuntu. Native simulation uses the 4090 via the shared GPU
lock. Keep the `intergrid-candidate-replay-v1` archive outside Git.
