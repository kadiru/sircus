"""Deterministic offline replay planning. No wall-clock sleeps or audio output."""
import json
from pathlib import Path

from .adapter import AudioAdapter
from .contract import AudioEvent
from .score import config
from .sources import simulation_events


def load_events(path):
    if Path(path).stat().st_size > 2_000_000:
        raise ValueError("audio trace exceeds 2 MB")
    return [AudioEvent.from_dict(json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]


def schedule(events):
    if not events or len(events) > 4096:
        raise ValueError("replay needs 1–4096 events")
    first = events[0]
    for event in events:
        if (event.trial_id, event.time_basis, event.clock_id) != (first.trial_id, first.time_basis, first.clock_id):
            raise ValueError("replay must use one trial and one explicit clock; do not merge unsynchronized clocks")
        if (event.source == "simulation") != (first.source == "simulation"):
            raise ValueError("simulation and real events must not share a replay")
    start = 0 if first.time_basis == "simulation_elapsed_s" else min(e.timestamp for e in events)
    unique = {}
    for event in events:
        key = (event.source, event.event_id)
        if key in unique and unique[key] != event:
            raise ValueError("one event ID cannot describe different observations")
        unique[key] = event
    ordered = sorted(unique.values(), key=lambda e: e.timestamp)
    if ordered[-1].timestamp - start > 3600:
        raise ValueError("replay exceeds one hour")
    return [(event.timestamp - start, event) for event in ordered]


def plan(events, mapping=None):
    rows = schedule(events)
    now, packets = [0.0], []

    class FakeSink:
        def send(self, payload):
            packets.append([round(now[0], 6), list(payload)])

    adapter = AudioAdapter(FakeSink(), mapping or config(), clock=lambda: now[0], threaded=False)
    adapter.start_trial(events[0].trial_id, "simulation" if events[0].source == "simulation" else "physical")
    for delay, event in rows:
        if not adapter.submit(event, delay):
            raise ValueError("trace exceeds configured capacity")
    adapter.pump()
    while adapter.pending or adapter.panic:
        if adapter.pending:
            now[0] = max(now[0], adapter.pending[0][0])
        adapter.pump()
    adapter.close()
    return packets


def demo_events(outcome="captured"):
    """Hand-authored synthetic scenario; never a physical/simulation-engine result."""
    raw = {"captured": "retained", "missed": "miss", "unknown": "model_limit"}[outcome]
    result = {"contacts": [{"t": .3, "object": "board"}, {"t": 1.1, "object": "table"},
                           {"t": 1.65, "object": "bucket-rim"}, {"t": 2.2, "object": "bucket-floor"}],
              "duration_s": 3.0, "outcome": raw, "retained": outcome == "captured"}
    events = simulation_events(result, "synthetic-demo-" + outcome)
    return [AudioEvent.from_dict({**e.to_dict(), "provenance": "hand-authored synthetic audio demo; no robot or physics run"})
            for e in events]
