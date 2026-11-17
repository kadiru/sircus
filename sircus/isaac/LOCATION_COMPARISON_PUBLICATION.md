# Read-only location-comparison publication

This package prepares a separate viewer for the [location comparison](LOCATION_COMPARISON.md).
Its implementation was checked with invented fixtures only. It does not establish
that native execution finished, that an audit passed, or that either continued
model improved. It starts no simulator, training, shots, reader or other service
on import. Publishing a report never promotes a model.

The historical six-run sweep/PPO benchmark and its existing reader remain separate.
The native run uses its previously pinned production source; these presentation
files do not replace that source, alter its manifest, or change the experiment.

## What the comparison means

The saved seed-22 model is the common parent. Two independently restored copies
receive 8,960 additional TRAIN requests each: one repeats the 31 usable original
bucket locations, and the other uses 8,960 distinct broader locations. Four of
the original 35 locations overlap arm bases in this Isaac scene. This compares
two continuations of one starting model, not three independently trained seeds.

The original parent and both final models make one direct prediction at each of
70 fresh broad TEST positions and 25 nearby TEST positions. The nearby positions
are 50 mm from selected original locations and remain a separate sensitivity
check. There is no pooled 95-position headline. Rejections stay in every request
denominator, and clean captures are a subset of captures using the unchanged
stricter clean-entry score. No held-out search, learning or final-model selection
is permitted by this publication.

The complete schedule is 17,920 TRAIN requests, 315 frozen monitoring requests,
and 285 final TEST requests: **18,520 total**. Exploratory TRAIN curves contain
256 batches of 35 requests per continued model. Their rates may rise or fall;
they are not cumulative rates or generalization results. Frozen monitoring uses
the same 35 monitoring positions at 0, 2,240, 4,480, 6,720 and 8,960 additional
TRAIN requests. Both curves share the original parent measurement at zero.

## Audit and publication gates

`location_comparison_publication.py` is a read-only certificate consumer, not a
replacement for the [independent model/physics audit](LOCATION_COMPARISON_PRODUCTION.md).
The caller supplies the expected runtime-manifest fingerprint, prepared-comparison
fingerprint and session ID. These identify one actual prepared attempt.

Outcome publication additionally requires an **externally trusted SHA256 of the
completed independent audit envelope**. Obtain it from the verified production
audit completion, not by trusting an arbitrary `audit.json` encountered in a
directory. The builder checks that pin and the associated fingerprints, complete
native and CPU-only audit process receipts, native child exit, accounting and
trace hash chain. It also rechecks final checkpoint bytes and the ten recording
packets and their source JPEG bytes. It refuses changed, linked, escaped,
nonfinite, duplicate-key or inconsistent evidence rather than relabeling it.

Without the audit pin and successful process gate, the builder reads only
allowlisted progress counters. TRAIN, monitor and TEST outcomes are not opened
or emitted. Status distinguishes unknown, running, audit pending and stopped;
native completion by itself does not establish audited results. A failed or
partial attempt remains preserved and has no automatic retry route here.

The browser separately requires the native completed-audit flags, explicit
`model_promoted: false`, both exact 8,960-request TRAIN summaries, and all three
70-position and 25-position TEST summaries. Until those checks pass, it hides
outcomes, learning curves, compute measurements and video players. Missing
measurements display as unknown rather than invented zero values. A failed
refresh clears previously displayed claims instead of silently retaining them.

## Time and memory scopes

The viewer shows observed bounded-phase elapsed time, native watchdog elapsed
time and separate audit-watchdog elapsed time. These scopes overlap and must not
be added into a new total. Recorded operation rows distinguish restoring,
checkpoint saves, PPO updates, exploratory TRAIN, frozen monitoring and final
TEST calls. Rollout process CPU time is separate from wall time.

Geometry time is already inside episode time. Sampling-state checkpoint writes
are included in proposal-call wall time; explicit initial/milestone/final saves
have their own checkpoint-call total. The reader does not subtract, extrapolate
or invent missing operations.

Memory is shown as maxima of the recorded samples grouped by model and phase,
with each group's sample count. Process RSS is a lifetime high-water mark;
GPU occupancy is a whole-device sample. Neither is a model-only memory measure
or an exact phase-local peak. Audit CPU use and audit memory are not measured
by this report.

## Ten recordings, then a separate encoding step

The experiment scheduled exactly these native recording cohorts:

| Model | Phase | New TRAIN requests completed before recording | Active / parked slots |
| --- | --- | --- | --- |
| Original-location copy | TRAIN | 0, 2,240, 4,480, 6,720 | 35 / 0 each |
| Broader-location copy | TRAIN | 0, 2,240, 4,480, 6,720 | 35 / 0 each |
| Original parent | Broad TEST subset | Not applicable | 24 / 11 |
| Original parent | Nearby TEST subset | Not applicable | 9 / 26 |

These are scheduled snapshots, not selected successes. **No final-model TEST
videos were scheduled.** They must not be inferred from the parent-only clips.

The builder verifies packet and ordered JPEG hashes, then emits frame counts,
frame-sequence identities and labels with `media: null`. It does not launch an
encoder or create playable media. Until a separately verified encoding is
attached, the UI says that source frames are verified but no playable video is
published.

That encoding step must use the audited packet's ordered JPEG sequence, without
rerendering or new shots, and preserve the model, phase, request milestone and
active/parked-slot labels. Bind the resulting file back to the packet and frame
sequence, verify its bytes/duration and attach a same-origin media descriptor:

```json
{
  "url": "/media/original-00000.mp4",
  "sha256": "<verified 64-character lowercase SHA256>",
  "duration_s": 4.0,
  "verified": true
}
```

The duration above illustrates the schema; it is not a measured clip duration.
Chapter entries identify the corresponding native phase and begin at valid
times inside that same file. Browser chapter buttons seek without autoplay.
Actual playback and chapter labels still require verification after real media
is available; mock tests cannot establish playback.

## Command-line interfaces

The following are documentation examples, not commands executed by this task.
Use actual verified identities and a separate publication directory. Keep raw
evidence, generated reports, media and runtime configuration out of Git.

Build the compact publication JSON on stdout:

```bash
python3 sircus/isaac/location_comparison_publication.py \
  --output /absolute/path/to/native-attempt \
  --audit-output /absolute/path/to/independent-audit \
  --expected-manifest-fingerprint "$COMPARISON_MANIFEST_SHA" \
  --expected-comparison-fingerprint "$COMPARISON_PLAN_SHA" \
  --expected-session-id "$COMPARISON_SESSION_ID" \
  --expected-audit-sha256 "$COMPARISON_COMPLETED_AUDIT_SHA"
```

Omitting `--expected-audit-sha256` produces status/progress only. A deliberately
staged `report.json` and any verified `media/*.mp4` belong in the new reader root,
not inside the preserved native evidence directory.

For an authorized reader start, first check that the chosen port is unused:

```bash
python3 sircus/isaac/location_comparison_server.py \
  --output /absolute/path/to/new-publication-root \
  --port 18803
```

The default **18803** is a separate proposed port, not evidence that it is
currently free. The command binds only `127.0.0.1`. It never restarts or replaces
the historical reader on 18797 or the existing continuation interface on 18802.

The exact read-only routes are `/`, `/location_comparison_panel.js`,
`/location_comparison.css`, `/api/location-comparison` (also `/report.json`), and
uniquely allowlisted `/media/<filename>.mp4` entries. The server supports GET,
HEAD and single byte ranges for playback. It checks media bytes against their
published SHA256 and rejects unaudited/incomplete/synthetic media, invalid
durations, symlinks, traversal and unlisted files. It exposes no raw observation,
checkpoint, trace or control endpoint. The report is an explicitly staged local
publication; the server does not itself reproduce the independent physics audit.

## Mock checks

```bash
python3 -m unittest \
  sircus/isaac/tests/test_location_comparison_publication.py \
  sircus/isaac/tests/test_location_comparison_server.py
node --test sircus/isaac/tests/test_location_comparison_panel.mjs
python3 scripts/check-repository.py
```

Tests use invented certificates, traces, frame bytes and HTTP requests. They
verify the presentation boundary without opening a port, loading a real model,
starting a simulator or producing a native performance result. The panel test
is also part of `scripts/test.sh`'s existing Node test command.

November 17 validation passed: 17 publisher cases, eight mocked HTTP cases and
12 DOM cases, plus an actual publisher-fixture-to-UI-export integration check.
The full repository suite passed with 1,692 Isaac cases (1,571 passed and 121
optional dependency skips), 71 Node cases and all other required stages.
Independent review found no remaining presentation/server integration blocker.
Native outcome publication, actual encoding and browser playback remain pending.
