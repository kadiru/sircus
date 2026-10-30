"""Pure, explicit adapters for existing saved records. Never connect to a robot."""
from datetime import datetime
import hashlib
import json

from .contract import AudioEvent, identifier, number


def _id(trial, kind, payload):
    raw = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256((trial + ":" + kind + ":" + raw).encode()).hexdigest()[:24]


def simulation_events(result, trial_id):
    """Convert a complete physics result, not gripper commands or native task events.

    Contacts are predicted records, not an exhaustive measured bounce sequence.
    No bounce_index or impact intensity is invented from their list positions.
    """
    identifier(trial_id, "trial_id")
    if not isinstance(result, dict) or "contacts" not in result:
        raise ValueError("expected a complete two-arm physics result with contacts")
    events = []
    for i, contact in enumerate(result["contacts"]):
        t = number(contact["t"], "contact time")
        obj = identifier(contact["object"], "contact object")
        events.append(AudioEvent(_id(trial_id, "contact", [i, contact]), trial_id,
                                 "simulation", "impact", obj, t, "simulation_elapsed_s",
                                 trial_id, "two-arm/physics.py contacts (predicted; may omit bounces)",
                                 "model_prediction"))
    raw = result.get("outcome")
    # False retained, a floor touch, timeout, model limit or absent label alone is NOT a miss.
    outcome = ("captured" if raw == "retained" and result.get("retained") is True else
               "missed" if raw in {"miss", "board_miss", "floor_contact_not_retained"} else "unknown")
    t = number(result.get("duration_s", max((e.timestamp for e in events), default=0)), "duration")
    if events and t < max(e.timestamp for e in events):
        raise ValueError("duration precedes a contact")
    events.append(AudioEvent(_id(trial_id, "outcome", [raw, t]), trial_id, "simulation", "outcome",
                             "unknown", t, "simulation_elapsed_s", trial_id,
                             "two-arm/physics.py terminal outcome; current retained definition is 0.2s",
                             "model_prediction", outcome=outcome))
    return sorted(events, key=lambda event: event.timestamp)


def operator_outcome(trial):
    """Only a saved explicit human label can produce a physical outcome cue."""
    if trial.get("schema") != "sircus-physical-trial-v1":
        raise ValueError("unsupported physical trial schema")
    trial_id = identifier(trial["id"], "trial id")
    stamp = trial.get("annotated_at") or trial.get("saved_at") or trial.get("created_at")
    dt = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("operator timestamp must include a timezone")
    labeled = (trial.get("saved") is True and trial.get("label_status") != "unlabeled"
               and bool(trial.get("annotated_at")) and trial.get("outcome") in {"captured", "missed"})
    outcome = trial["outcome"] if labeled else "unknown"
    # Dry-run labels must never masquerade as physical observations.
    if trial.get("mode") != "hardware":
        raise ValueError("physical label adapter requires mode=hardware; use synthetic demo for mocks")
    return AudioEvent(_id(trial_id, "label", [stamp, outcome]), trial_id, "operator", "outcome",
                      "unknown", dt.timestamp(), "unix_s", "operator-record-wall-clock",
                      "physical_trials/controller.py saved operator annotation; not automatic detection",
                      "operator_label", outcome=outcome, confirmed=labeled)


def odd_tap(trial_id, event_id, velocity, received_monotonic, clock_id, device_label="ODD"):
    """Call only for observed positive note-0 NoteOn on the verified ball/channel.

    Velocity/127 is a MIDI-strength proxy, not force or calibrated acceleration.
    The caller supplies stable per-observation IDs and a host/session clock ID.
    """
    number(velocity, "MIDI velocity", 1, 127)
    return AudioEvent(event_id, trial_id, "sensor", "impact", "unknown", received_monotonic,
                      "host_monotonic_s", clock_id,
                      f"{device_label}: BLE MIDI ch1 note0; host arrival time; velocity/127 proxy",
                      "midi_gesture", intensity=velocity / 127)
