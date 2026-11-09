# Watching the recorded height/tilt failures

The completed height/tilt study saved dense native trajectories, but did not
record a movie during execution. The replay renders four of those recordings
in the original MuJoCo scene: three misses and a successful reference control.
The controls do not change the discovery result of **0 captures from 58 drops**.

The ball follows actual recorded center samples at 64 substeps. Both arms stay
at their recorded endpoint joints, as they did during these experimental
flights. There is no invented arm transit, interpolated ball path, new physics
rollout or new scoring. Ball spin was not recorded. Normal-speed and quarter-speed
views include pauses at the exact recorded first robot-contact samples.

| Example | Board height / tilt / gap | What the recording shows |
| --- | --- | --- |
| Case 00, g23 | 460 mm / 28° / 251.819 mm | Board at 0.223 s; robot at 0.476 s; bucket at 0.569 s; no entry |
| Case 18, g23 | 460 mm / 28° / 240.200 mm | Board at 0.218 s; robot at 0.468 s; table at 0.656 s; no entry |
| Case 01, g17 | 440 mm / 24° / 239.833 mm | No robot contact; bucket sidewall at 0.596 s; no entry |
| Table-assisted positive control | See exact video metadata | Board at 0.214 s; table at 0.618 s; bucket at 0.800 s; one-second capture at 2.084 s |

For cases 00 and 18, CPU reconstruction at the saved contact samples identifies
the bouncer's **link 2 and link 3**, respectively, as the intersecting geometry.
The original native log records aggregate robot force rather than the link ID;
the named links are a corroborating geometric reconstruction, not a new force
measurement. At case 01's first bucket impact the ball center was **122.7 mm**
above the tabletop, against the exterior wall below the rim.

This suggests two separate checks before another training run: find admitted
arm configurations that leave the rebound path clear of the bouncer itself,
and adjust the flight of otherwise unobstructed shots to clear the bucket rim.
The videos do not prove which parameter change will solve either problem.
Table-assisted routes remain permitted; the successful control illustrates one.

## Evidence and reproduction

- [Original results](../baselines/mjlab-table-height-tilt-20261108/README.md)
  retain the full 120-proposal denominator and all historical labels.
- [Contact evidence](../baselines/mjlab-table-height-tilt-20261108/video-contact-evidence.json)
  binds the report, independent 36,371-check audit, rows, native archives,
  collision assets and model sources. It includes the exact contact samples and
  CPU-only geometric reconstruction.
- `python -m sircus.mjlab.height_tilt_video --help` describes the renderer.
  It checks those identities, clips every shot at its scoring terminal, and
  requires software OpenGL with both GPUs hidden. `--preview-only` produces
  camera previews; `--cameras` changes only view framing.
- The full movie sidecar records every output frame's source tick, camera
  settings, held joint values, input/output hashes and rendering backend.
  Raw traces, full sidecars, MP4s and runtime HTML remain outside Git.
- `python -m sircus.mjlab.prepare_height_tilt_replay --help` describes the
  additive publisher. It checks the movie and frame-map hashes against the
  audited study, then updates only its own HTML section and display-report key
  using compare-and-swap. No new media route or server restart is required.

The interface section is `#heightTiltReplay`. It provides four chapter buttons,
native video controls, a speed selector, a download link and exact event times.
Event times in its table refer to the original shot, not the edited video clock.
Neither the 4070 workload nor the Isaac setup is used by this replay.
