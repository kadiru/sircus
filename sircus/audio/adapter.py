"""Bounded, best-effort audio scheduling, deliberately outside robot control."""
import heapq
import threading
import time

from .contract import AudioEvent, identifier, number
from .score import choose, config, score, validate_config


class AudioAdapter:
    """submit/cancel never wait on Logic, MIDI, network, filesystem or a worker.

    One explicitly started trial at a time. Deduplication lasts for that entire
    trial (no eviction); exceeding its cap fails closed. start_trial is an
    explicit new replay generation and permits intentional re-audition.
    The short state lock never covers output calls. This is not a hard-real-time
    robot interlock. Output failure drops audio; it cannot stop/control a robot.
    """
    def __init__(self, sink, mapping=None, clock=time.monotonic, threaded=True):
        self.sink, self.mapping, self.clock = sink, validate_config(mapping if mapping is not None else config()), clock
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.closed = False
        self.trial = None
        self.mode = None
        self.generation = 0
        self.sequence = 0
        self.pending = []
        self.seen = set()
        self.board_seen = False
        self.terminal = False
        self.panic = False
        self.muted = self.mapping["muted"]
        self.errors = 0
        self.dropped = 0
        self.last_error = None
        self.worker = None
        if threaded:
            self.worker = threading.Thread(target=self._run, daemon=True, name="sircus-audio")
            self.worker.start()

    def start_trial(self, trial_id, mode):
        identifier(trial_id, "trial_id")
        if mode not in {"simulation", "physical"}:
            raise ValueError("mode must be simulation or physical")
        with self.lock:
            if self.closed:
                return False
            self.generation += 1
            self.pending.clear()
            self.seen.clear()
            self.trial, self.mode = trial_id, mode
            self.board_seen = self.terminal = False
            self.panic = True
        self.wake.set()
        return True

    def submit(self, event, delay_s=0):
        """Explicit host-relative delay; never subtract unrelated source clocks.

        Return False for bad input, another trial/domain, duplicates or capacity.
        Conversion of large source records belongs in a separate observer/replay.
        """
        try:
            if not isinstance(event, AudioEvent):
                event = AudioEvent.from_dict(event)
            number(delay_s, "delay_s", 0, 3600)
        except (ValueError, TypeError):
            return False
        with self.lock:
            if self.closed or event.trial_id != self.trial:
                return False
            if (event.source == "simulation") != (self.mode == "simulation"):
                return False
            key = (event.source, event.event_id)
            if key in self.seen or len(self.seen) >= self.mapping["max_events"]:
                return False
            if len(self.pending) >= self.mapping["max_pending_messages"]:
                self.dropped += 1
                return False
            self.seen.add(key)
            if self.muted:
                # Consume muted input permanently; unmute must not replay a backlog.
                if event.event_type == "stop":
                    self.generation += 1
                    self.pending.clear()
                    self.trial = None
                    self.panic = True
                if event.event_type == "impact" and event.contact_object == "board":
                    self.board_seen = True
                if event.event_type == "outcome":
                    self.terminal = True
                return True
            self.sequence += 1
            heapq.heappush(self.pending, (self.clock() + delay_s, self.sequence,
                                          self.generation, "event", event))
        self.wake.set()
        return True

    def cancel(self, trial_id=None):
        with self.lock:
            if trial_id is not None and trial_id != self.trial:
                return False
            self.generation += 1
            self.pending.clear()
            self.trial = None
            self.panic = True
        self.wake.set()
        return True

    def set_muted(self, muted):
        if type(muted) is not bool:
            raise ValueError("mute must be boolean")
        with self.lock:
            self.muted = muted
            if muted:
                self.generation += 1
                self.pending.clear()
                self.panic = True
        self.wake.set()

    def set_gain(self, gain):
        number(gain, "gain", 0, 1)
        with self.lock:
            self.mapping["gain"] = gain
        # Previously queued notes retain their velocity; mute/stop is immediate.

    def _send(self, payload):
        try:
            self.sink.send(payload)
        except Exception as exc:
            self.errors += 1
            self.last_error = type(exc).__name__
            # No retries: replaying a note after an unknown send outcome duplicates it.

    def _panic(self):
        ch = self.mapping["channel"] - 1
        for payload in ([0xB0 | ch, 120, 0], [0xB0 | ch, 123, 0], [0xE0 | ch, 0, 64]):
            self._send(payload)

    def pump(self):
        """Single-worker tick; public only to permit deterministic fake-clock tests."""
        with self.lock:
            panic, self.panic = self.panic, False
        if panic:
            self._panic()
        # A tick has a hard work-count cap, even if many events are overdue.
        for _ in range(256):
            with self.lock:
                if self.panic or not self.pending or self.pending[0][0] > self.clock():
                    break
                due, _, generation, kind, item = heapq.heappop(self.pending)
                if generation != self.generation or self.muted:
                    continue
                if kind == "event":
                    if item.event_type == "stop":
                        self.pending.clear()
                        self.generation += 1
                        self.trial = None
                        self.panic = True
                        break
                    if self.terminal:
                        continue
                    cue = choose(item, self.mapping, self.board_seen)
                    if item.event_type == "impact" and item.contact_object == "board":
                        self.board_seen = True
                    if item.event_type == "outcome":
                        self.terminal = True
                    if cue is None:
                        continue
                    # Keep future events, but replace the previous monophonic gesture.
                    self.pending = [p for p in self.pending if p[3] == "event"]
                    heapq.heapify(self.pending)
                    packets = score(cue, item.intensity, self.mapping["gain"], self.mapping["channel"])
                    if len(self.pending) + len(packets) > self.mapping["max_pending_messages"]:
                        self.dropped += 1
                        self.panic = True
                        continue
                    # Anchor late events at NOW, not their stale due time (no bend burst).
                    anchor = self.clock()
                    for offset, packet in packets:
                        self.sequence += 1
                        heapq.heappush(self.pending, (anchor + offset, self.sequence,
                                                      generation, "midi", packet))
                    do_panic, payload = True, None
                else:
                    do_panic, payload = False, item
            # Sink code NEVER runs under the state lock. An already-in-flight
            # send may finish during cancel; the next worker tick sends panic.
            if generation != self.generation:
                continue
            if do_panic:
                self._panic()
            else:
                self._send(payload)

    def _run(self):
        while not self.closed:
            self.pump()
            self.wake.wait(.005)
            self.wake.clear()
        self._panic()

    def close(self):
        self.cancel()
        self.closed = True
        self.wake.set()
        if self.worker:
            self.worker.join(timeout=.1)
        else:
            self.pump()
        # A hung custom sink cannot make close wait forever. The caller owns
        # sink disposal; the provided MIDI sink uses nonblocking pipe writes.
