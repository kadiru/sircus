# Ghost Booth loudness

The native Mac playback path has an optional **3× boost** beside Master volume.
It adds up to +9.54 dB of signal gain before a peak limiter on the combined
output. This is a signal-amplitude setting, not a claim of three times the
perceived loudness; overlapping sounds may be reduced by the limiter.

- Master volume still controls the output, including zero/silence.
- Boost starts off and applies only to native Mac playback. Logic MIDI routing
  keeps its existing volume behavior.
- The UI distinguishes a requested setting from the native helper's applied
  acknowledgment. Boost requires a ready native limiter; unsupported or failed
  requests must not appear successful.
- Changing boost does not enable FX, preview a sound, arm a trial, or command a
  robot. FX off, cancellation, Stop playback and Mute all retain their meanings.
- Raw MIDI observations, per-ball bindings, sound choices and trial evidence are
  unchanged. Software gain does not identify a contact surface or an outcome.

For listening, set the desired master level, enable the boost, then explicitly
enable FX and arm the manual audio test or the approved show routine. Normal
show impacts still use their bounded post-release windows. A louder output does
not bypass those windows. Software gain cannot increase the speaker's physical
maximum output.

The original four-ball manual tests at maximum master and system volume were
audible but reported too quiet in the venue. The operator reported balls reaching
the bucket; this is an aggregate observation, not an automatically scored trial
set. Host-specific MIDI events and listening evidence remain outside Git.

After the boost was deployed to Ghost Booth on this MacBook, the operator
confirmed the manual-drop sounds were loud enough and clear. The test does not
measure sound-pressure level or establish robot-cycle timing. The selected boost
can remain on while FX is muted; a service restart resets it to normal gain.
