"""Operator-run exports/previews. Default commands never produce sound."""
import argparse
import json
from pathlib import Path
import sys
import time

from .adapter import AudioAdapter
from .midi import MidiSink
from .preview import write_preview
from .replay import demo_events, load_events, plan, schedule
from .score import config
from .sources import operator_outcome, simulation_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="write a silent-until-clicked browser preview and three JSONL traces")
    demo.add_argument("--output", default=".runtime/sfx-preview.html")
    demo.add_argument("--config")
    convert = sub.add_parser("convert", help="convert a saved record, without SDK calls")
    convert.add_argument("input")
    convert.add_argument("--source", choices=["simulation", "operator"], required=True)
    convert.add_argument("--trial-id")
    convert.add_argument("--output", required=True)
    replay = sub.add_parser("replay", help="plan a JSONL trace; audible Logic playback requires --play")
    replay.add_argument("trace")
    replay.add_argument("--config")
    replay.add_argument("--play", action="store_true")
    replay.add_argument("--midi-helper", type=Path)
    replay.add_argument("--setup-seconds", type=float, default=10,
                        help="silent time to select the Sircus SFX input in Logic before playback")
    args = parser.parse_args()
    if args.command == "convert":
        record = json.loads(Path(args.input).read_text())
        if args.source == "simulation":
            if not args.trial_id:
                parser.error("simulation export requires a stable --trial-id")
            events = simulation_events(record.get("result", record), args.trial_id)
        else:
            events = [operator_outcome(record)]
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(json.dumps(e.to_dict(), allow_nan=False) + "\n" for e in events))
        print(f"Wrote {len(events)} events to {target}; no audio played")
        return
    mapping = config(args.config)
    if args.command == "demo":
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_preview(target, mapping)
        for outcome in ("captured", "missed", "unknown"):
            target.with_name(outcome + ".jsonl").write_text(
                "".join(json.dumps(e.to_dict()) + "\n" for e in demo_events(outcome)))
        print(f"Open {target.resolve()} and click a cue or replay. Nothing has played automatically.")
        return
    events = load_events(args.trace)
    planned = plan(events, mapping)
    if not args.play:
        print(json.dumps({"dry_run": True, "source": events[0].source,
                          "trial_id": events[0].trial_id, "midi_messages": planned}, indent=2))
        return
    if not args.midi_helper:
        parser.error("--play requires --midi-helper .runtime/sircus-midi")
    if not 0 <= args.setup_seconds <= 300:
        parser.error("setup-seconds must be 0–300")
    # Construction and startup wait happen here in the separate operator CLI,
    # never in a robot controller callback.
    sink = MidiSink(args.midi_helper.resolve())
    adapter = AudioAdapter(sink, mapping)
    try:
        print(f"Sircus SFX port ready. Playback begins in {args.setup_seconds:g}s; Ctrl-C cancels.", flush=True)
        time.sleep(args.setup_seconds)
        adapter.start_trial(events[0].trial_id, "simulation" if events[0].source == "simulation" else "physical")
        rows = schedule(events)
        for delay, event in rows:
            if not adapter.submit(event, delay):
                raise RuntimeError("trace rejected: duplicate IDs or queue capacity")
        deadline = time.monotonic() + rows[-1][0] + 2
        while time.monotonic() < deadline:
            time.sleep(.02)
        if adapter.errors:
            raise RuntimeError(f"audio output failed ({adapter.last_error}); no retry performed")
    finally:
        adapter.close()
        sink.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Cancelled; pending audio cleared.")
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(f"Audio only: {error}", file=sys.stderr)
        raise SystemExit(1)
