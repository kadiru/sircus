"""Versioned boundary between evidence producers and optional sound playback."""
from dataclasses import asdict, dataclass
import math

SCHEMA = "sircus.audio.v1"


def number(value, name, low=0, high=1e15):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} outside range")
    return float(value)


def identifier(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{name} must be 1–256 characters")
    return value


@dataclass(frozen=True)
class AudioEvent:
    event_id: str
    trial_id: str
    source: str
    event_type: str
    contact_object: str
    timestamp: float
    time_basis: str
    clock_id: str
    provenance: str
    evidence: str
    outcome: str = "unknown"
    confirmed: bool = False
    confidence: float | None = None
    bounce_index: int | None = None
    intensity: float | None = None
    schema: str = SCHEMA

    def __post_init__(self):
        if self.schema != SCHEMA:
            raise ValueError("unsupported audio schema")
        for key in ("event_id", "trial_id", "clock_id", "provenance"):
            identifier(getattr(self, key), key)
        if self.source not in {"simulation", "sensor", "operator"}:
            raise ValueError("invalid source")
        if self.event_type not in {"impact", "outcome", "stop", "pickup", "carry_start", "carry_stop",
                                   "release", "flight_start", "flight_stop"}:
            raise ValueError("invalid event_type")
        identifier(self.contact_object, "contact_object")
        if self.time_basis not in {"simulation_elapsed_s", "host_monotonic_s", "unix_s"}:
            raise ValueError("invalid time_basis")
        number(self.timestamp, "timestamp")
        if self.source == "simulation" and self.time_basis != "simulation_elapsed_s":
            raise ValueError("simulation must retain its simulation clock")
        if self.source != "simulation" and self.time_basis == "simulation_elapsed_s":
            raise ValueError("physical observations cannot use a simulation clock")
        allowed = {"simulation": {"model_prediction"}, "sensor": {"midi_gesture", "sensor_detection"},
                   "operator": {"operator_label", "operator_action"}}
        if self.evidence not in allowed[self.source]:
            raise ValueError("source/evidence mismatch")
        if self.outcome not in {"unknown", "captured", "missed"}:
            raise ValueError("invalid outcome")
        if type(self.confirmed) is not bool:
            raise ValueError("confirmed must be boolean")
        if self.source == "simulation" and self.confirmed:
            raise ValueError("a prediction is not a confirmed physical observation")
        if self.evidence == "operator_action" and (self.confirmed or self.outcome != "unknown"):
            raise ValueError("an action request is not a confirmed physical outcome")
        if self.evidence == "midi_gesture" and (self.confirmed or self.contact_object != "unknown"
                                                 or self.outcome != "unknown"):
            raise ValueError("ODD MIDI alone cannot identify contact or confirm an outcome")
        if self.event_type != "outcome" and self.outcome != "unknown":
            raise ValueError("outcome belongs only on an outcome event")
        if self.confidence is not None:
            number(self.confidence, "confidence", 0, 1)
        if self.intensity is not None:
            number(self.intensity, "intensity", 0, 1)
        if self.bounce_index is not None:
            if type(self.bounce_index) is not int or not 1 <= self.bounce_index <= 100000:
                raise ValueError("bounce_index must be a positive known ordinal or null")

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("event must be an object")
        try:
            return cls(**value)
        except TypeError as exc:
            raise ValueError("missing or unexpected event fields") from exc

    def to_dict(self):
        return asdict(self)
