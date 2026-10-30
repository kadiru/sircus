# ODD Ball evidence from the Mac Mini

## Observed in the previous Logic session

Two balls connected through macOS **Bluetooth LE MIDI**, exposed as separate
CoreMIDI inputs both named **ODD**. In Logic, the actual tap was **channel 1,
note 0**, despite reference documents listing tap note 1. Movement included CC5;
mapping it to a limited Retro Synth cutoff range produced audible brightness
changes. The user confirmed first ball/drums and second ball/synth isolation and
extended clean playback after removing a diagnostic logger and reconnecting.

The correct two-track setup used separate input ports with **I on, R off**.
Record-enabling both had previously sent a ball to both instruments. Input menu
order may change after reconnection; names alone do not identify the physical ball.
Shake/twist mappings and operation with all 9–10 owned balls are unverified.

An intermittent loud scratch/squeal followed one route during troubleshooting.
The old logger emitted repeated `Trace() failed. Try a shorter string`. Removing
logging was followed by clean single-ball, dual-ball and restored-tilt tests, but
reconnections also occurred. This is correlation, not proof of the exact cause.
The new receiver contains no diagnostic logging and consumes unrelated CCs.

## Timing, sensitivity and sensing limits

- Old JavaScript diagnostics used `Date.now()` at host receipt. They did not
  retrieve a sensor acquisition timestamp. BLE transport timestamps must not be
  claimed as measured acquisition time here. The new `odd_tap` boundary asks the
  caller for host monotonic arrival time and a host/session clock ID.
- Four early observed tap velocities were 9–21. This is a small anecdotal sample,
  not calibration, a threshold, a minimum detectable impact or a force estimate.
  Velocity/127 in the converter is explicitly a MIDI-strength proxy.
- No measured latency, jitter, drop rate, sensitivity curve, impact classifier,
  accelerometer stream, calibrated orientation, or sensor sampling rate is available.
- The attempted later ten-tap diagnostic report was unreadable; no counts or
  timing estimates are derived from it. Ball generation/firmware remains unverified.
- ODD gestures cannot identify board versus bucket, count all physical bounces,
  prove retention, or classify a miss. Ball motion after a robot command is not
  automatically a measured trial event. A sensor or an explicit operator label
  must establish the required object/outcome evidence independently.

## References and reuse

The [ODD MIDI page](https://oddballism.com/pages/midi) links the
[MIDI Reference v1.3.0](https://docs.google.com/document/d/14L2wokwEkl3OIqeRpiXxA0IOLzT7xJSZx7kKNLflzVw/edit).
Connected Drive documentation read in the prior work also included v1.0/v1.1
copies. Their CC orientation and note timeout descriptions differ. Older desktop
Max example routing used notes 0/1/2. Observed messages on this Mac take precedence
over assuming one document applies to every ball.

The project reuses the existing Scripter MIDI path and note-release ownership
pattern, with original procedural MIDI gestures added in `score.py`. It adds no
firmware changes, Max/Ableton dependencies or system-wide MIDI filter.
Apple documents the [Scripter API](https://support.apple.com/guide/logicpro/scripter-api-overview-lgce3905a48c/mac)
and [ProcessMIDI callback](https://support.apple.com/guide/logicpro/processmidi-function-lgce225e4d89/mac).
`coremidi.swift` uses the installed macOS MIDI-1 APIs for Logic compatibility and
was compiled/tested locally. Its virtual source carries **outgoing cue MIDI**;
it does not subscribe to or merge the two physical ODD input ports.
