# Round 1 drop pose as home

The Manual operation panel has **Reset → Round 1 drop home**, with its own
empty-gripper, fixed-setup and clear-route confirmation. This is an explicit
physical motion command, separate from starting a round. It does not open or
close either tool, move arm 14, release suction, or start cameras.

Reset requires fresh readback, no active dashboard action, valid taught round
presets, and a matching request epoch. It verifies Stop and planner quiescence,
checks the live held-object sensor is explicitly false, and checks the current
board reference. A held or unknown ball blocks movement. An interrupted record
is retained unlabeled; completed records are not relabeled aborted.

Supported starting poses are a taught pickup, its 100 mm approach, the existing
pickup clearance waypoint, or either saved round drop pose. From a low pickup,
the empty tool retreats vertically with fixed XY/orientation before planner
transfer. Unknown intermediate poses are blocked rather than guessed safe.
Round 1's saved drop pose is the final home target. The board stays where it is;
**Run Round 1** performs the existing board transition after a new confirmation.
Stop cancels reset motion without tool actuation. Audio is muted through the
existing asynchronous observer, and cannot gate motion or stopping.

A successful reset creates a separate home-ready record referencing the prior
round; prior round/trial evidence remains intact. Reload all four pickup marks,
confirm the empty gripper and clear paths, then press **Run Round 1 · four balls**.
After Round 1, reload and explicitly start Round 2 as before.

Validation: 162 physical-controller mock tests, 45 Python server/startup checks,
and 117 JavaScript UI tests passed. The idle Mac hardware dashboard was loaded
with this implementation, camera-off, connected and both arms stationary. The
button was verified served and available despite the retained interrupted-trial
fault; reset itself performs verified recovery before motion. No home movement
or tool actuation was issued during implementation/deployment. Physical reset
execution remains operator initiated and not claimed validated by these mocks.
