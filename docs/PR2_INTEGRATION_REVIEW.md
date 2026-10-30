# PR #2 integration review — October 30, 2026

Merged [PR #2](https://github.com/kadiru/sircus/pull/2) after reviewing actual head
`5294fed016746d3574ed20b47ff7c7775d089551` against main
`26ca998f3e93b3dbfa795e5db354d4c07985d703`. The resulting main merge commit is
`6f00ac3faa4e34d508b005891e62d9911ead81cf`.

## Review and verification

Two independent code reviews found no blocking scoring/contact or compatibility
defect. The new complete-shot contract requires board bounce before qualifying
whole-ball entry and one continuous simulated second of containment. Partial
exits reset dwell; full exits require a fresh valid crossing. Rim-assisted and
clean captures both count. Numerical/model-limit failures override success.

Candidate selection and evaluation both call the versioned `shot_rollout.rollout`
and rank success before shaping. The selected policy is written before held-out
evaluation. The native v2 replay guard rejects before any connection attempt.

Executed on the Mac in an isolated checkout, using the existing Python 3.12
environment and `PYTHONDONTWRITEBYTECODE=1`:

```sh
SIRCUS_PYTHON=/path/to/project/python bash scripts/test.sh
/path/to/project/python sircus/two-arm/train_orientation_height.py \
  --budget 4 --tests 2 --seed 42 --output /tmp/unique-pr2-smoke
/path/to/project/python scripts/check-repository.py --tracked
git diff --check
```

- All **99 unittest tests** passed, plus orientation contract assertions.
- Small numerical smoke: four executable candidates; frozen selected action
  captured in **2/2 fresh evaluation episodes**. Selected roll 1.641594°,
  pitch 39.660427°, gap 116.688541 mm. These are fixed-scene synthetic results,
  not a learned cross-scene policy or hardware validation.
- All 127 tracked files passed the repository file/credential scan.
- Both current GitHub checks passed at the reviewed head, including
  [PR CI](https://github.com/kadiru/sircus/actions/runs/41029254925).
- Actual head/base/CI were rechecked immediately before merging with an explicit
  head-commit match. No unreviewed head was merged.

## Preserved boundaries

`simulate` without `capture_mode` remains legacy behavior. New scoring is opt-in;
merging this contribution did not switch the running Viam runtime/dashboard to
it. Historical rectangular/bucket evidence retains its original definitions,
including the earlier 0.2-second success criterion.

The Mac frontend, server, physical controller/adapters, Viam module, scene assets
and baselines were unchanged by PR #2. No hardware connections, robot commands,
service restarts, cloud changes or native-path execution occurred in this review.
The separate audio PR #1 was not merged. Uncommitted dashboard/audio work remains
in its own checkout and branch.

**V2 native replay remains blocked until a matching adapter is implemented.**
Do not remove that guard to make numerical replay appear native. Endpoint IK is
also separate: `shot_rollout.rollout` alone does not call `feasibility`; consumers
must run that check explicitly and must still distinguish endpoints from paths.

## Ownership for the next checkpoint

The planning conversation and published branches were inspected before starting
new work. Ubuntu's published scoring task was complete; no published randomized
study implementation or existing local scene-conditioned checkpoint was found.
The next task branch is `codex/macbook-scene-conditioned-study`, based on the
merge above. It reuses the reviewed scorer and numerical model; it does not
replace the simulator or commission physical execution.
