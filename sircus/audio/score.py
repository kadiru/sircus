"""Small original MIDI sound gestures; no samples or recordings are required."""
import hashlib
import json
import math
from pathlib import Path

from .contract import number

CUES = ("boing", "squeak", "plop", "spring", "celebration", "wong")


def config(path=None):
    value = json.loads(Path(path or Path(__file__).with_name("mapping.example.json")).read_text())
    return validate_config(value)


def validate_config(value):
    if not isinstance(value, dict):
        raise ValueError("mapping must be an object")
    if set(value) != {"gain", "muted", "channel", "first_board", "bounces", "captured", "missed",
                      "max_events", "max_pending_messages"}:
        raise ValueError("unexpected or missing mapping keys")
    number(value["gain"], "gain", 0, 1)
    if type(value["muted"]) is not bool:
        raise ValueError("muted must be boolean")
    for name, low, high in (("channel", 1, 16), ("max_events", 1, 100000),
                            ("max_pending_messages", 1, 100000)):
        if type(value[name]) is not int or not low <= value[name] <= high:
            raise ValueError(f"invalid {name}")
    if not isinstance(value["bounces"], list) or not 1 <= len(value["bounces"]) <= 16:
        raise ValueError("bounces needs 1–16 cue names")
    if any(c not in CUES for c in [value["first_board"], value["captured"], value["missed"], *value["bounces"]]):
        raise ValueError("unknown cue name")
    return {**value, "bounces": list(value["bounces"])}


def choose(event, mapping, board_seen=False):
    if event.event_type == "impact":
        if event.contact_object == "board" and not board_seen:
            return mapping["first_board"]
        index = int(hashlib.sha256(event.event_id.encode()).hexdigest()[:8], 16)
        return mapping["bounces"][index % len(mapping["bounces"])]
    if event.event_type == "outcome" and event.outcome != "unknown":
        if event.source == "simulation" or event.confirmed:
            return mapping[event.outcome]
    return None


def score(cue, intensity=None, gain=0.55, channel=16):
    """Return sorted (seconds, three MIDI bytes). Pitch bend assumes ±12 semitones.

    Each cue is monophonic. The dispatcher preempts the previous gesture on a
    new impact so a new bounce is immediate and old bends cannot affect it.
    """
    if cue not in CUES:
        raise ValueError("unknown cue")
    number(gain, "gain", 0, 1)
    if type(channel) is not int or not 1 <= channel <= 16:
        raise ValueError("channel must be 1–16")
    if intensity is not None:
        number(intensity, "intensity", 0, 1)
    velocity = round(100 * gain * (0.55 + 0.45 * (0.6 if intensity is None else intensity)))
    if velocity == 0:
        return []
    ch = channel - 1
    rows = []

    def note(t, pitch, duration):
        rows.extend([(t, [0x90 | ch, pitch, velocity]), (t + duration, [0x80 | ch, pitch, 0])])

    def bend(t, semitones):
        v = max(0, min(16383, round(8192 + semitones * 8191 / 12)))
        rows.append((t, [0xE0 | ch, v & 127, v >> 7]))

    if cue in {"boing", "spring", "squeak", "plop"}:
        pitch, duration = {"boing": (55, .65), "spring": (64, .42),
                           "squeak": (76, .23), "plop": (43, .19)}[cue]
        note(0, pitch, duration)
        for i in range(25):
            u = i / 24
            semitones = (10 * math.exp(-4*u) * math.cos(3*math.pi*u) if cue in {"boing", "spring"}
                         else 9*(1-2*u) if cue == "squeak" else -8*u)
            bend(duration * u, semitones)
        bend(duration + .01, 0)
    elif cue == "celebration":
        for i, pitch in enumerate((60, 64, 67, 72, 79)):
            note(i * .13, pitch, .11 if i < 4 else .45)
    else:
        for i, pitch in enumerate((55, 52, 48)):
            t = i * .4
            note(t, pitch, .3)
            for j in range(10):
                bend(t + j * .03, -j * .6)
            bend(t + .32, 0)
    return sorted(rows, key=lambda row: row[0])
