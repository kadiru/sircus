"""Optional audio only: no Viam, robot, network or playback side effects on import."""

from .contract import AudioEvent
from .adapter import AudioAdapter

__all__ = ["AudioEvent", "AudioAdapter"]
