# Preserve a checked shot when expanded search fails

The previous 50-position comparison captured 43 buckets with the current
planner and 32 with expanded search plus a complete bucket-approach forecast.
Expanded search declined nine setups. This follow-up diagnoses those declines
and tests a single fallback rule before any further training.

All nine declines exhausted the first 16 candidates in 2.15–8.58 seconds;
none reached the 30-second deadline. Of 144 tested candidates, 134 were new
local tilt/gap variants. Rejections comprised 81 analytic wrist-reach failures,
42 unresolved finite IK searches, 20 parameter-bound/minimum-distance
failures, and one endpoint pose/clearance failure. An unresolved finite IK
search is not proof that the arm can never reach the requested pose.

The current planner's selected shot was absent from five expanded menus and
ranked 60th, 70th, 76th and 76th in the other four. Those ranks exceed the
16-candidate initial check budget. No expanded candidate passed setup, so its
later alternative-joint search had no admitted starting point. This diagnoses
a candidate-ordering and fallback problem; it does not establish that the
bucket positions are unreachable. Historical current-planner drops captured
seven of these nine positions. Those outcomes do not choose new candidates.

## Frozen rule

Run the unchanged current and expanded planners in two concurrent CPU lanes.
Each retains its 30-second decision budget and existing arm/path checks. If the
expanded planner returns a checked shot, select it. Otherwise preserve the
current planner's checked shot, including its full command, arm joints, path
proofs and native adapter. If neither finds a checked shot, decline the setup.

The fallback does not solve IK again, relax collision checks, use the outcome
of a test drop, or substitute a previously successful shot at another target.
It preserves a feasible setup, not a guarantee that the ball will score.

This method uses two planning lanes; the current planner uses one. Report the
actual paired request duration and each lane's decision duration. Their sum is
decision-window time, not measured CPU usage. Shared baseline preparation in
the experiment avoids duplicated work but must remain explicit. Native drops
are independent for the two compared methods.

## Test sequence

1. Re-plan the nine known declined positions. Repeat each selected command
   three times for both the current planner and the fallback method. These
   are development cases, reported separately from generalization results.
2. Keep the same rule frozen and evaluate 50 fresh bucket positions excluded
   from the historical training, validation, search and test inventories.
   Each method gets one attempt per position. Declined setups remain in the
   denominator.
3. Use native MuJoCo at 64 substeps with matched capture/miss controls. Check
   recorded dense trajectories independently, including the complete command
   and setup-proof bindings. Report captures, setup declines, flight misses,
   invalid physics, fallback use and planning time.

Capture scoring is unchanged: the first bounce must be on the board, then
the whole ball must remain inside the bucket for one continuous second.
Subsequent table, rim, robot or other rebounds are allowed. Partial exits
reset retention. Invalid physics remains a failed attempt.

The arms are held at checked endpoints during native ball flight. CPU path
checks are not a driven-arm simulation or physical validation. Simulator
geometry supplies inputs; this experiment uses no camera perception.

The RTX 4090 is leased for native execution only. The 4070 workload, Isaac
installation, live interactive planner and existing services stay unchanged.
Prior results, recipes and recordings are preserved. No model is retrained.
