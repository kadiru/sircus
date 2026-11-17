"""Inert budget guards for the prepared location comparison.

Bounds is a cooperative in-process guard. SnapshotWatchdog is an independent
clock/resource observer for a future identity-owned process supervisor; neither
class starts or kills a process. Mock checks are not a native watchdog test.
"""

import copy
import math
import time

ARMS = ("original_usable", "broader")
TRAIN_CAP = 8960
ASSESSMENT_CAP = 600
ARM_WALL_S = 28800
SHARED_WALL_S = 3600
MIN_FREE_BYTES = 6 * 1024**3
SCHEMA = "sircus-location-comparison-bounds-v1"


class BoundsExceeded(RuntimeError):
    """Latched stop: retain all evidence and charged reservations; never retry."""


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _limits(plan):
    protocol = plan["protocol"]
    expected = dict(proposed_train_requests_per_arm=TRAIN_CAP,
                    proposed_total_train_requests=2 * TRAIN_CAP,
                    proposed_wall_seconds_per_training_arm=ARM_WALL_S,
                    proposed_shared_assessment_wall_seconds=SHARED_WALL_S,
                    minimum_free_storage_bytes=MIN_FREE_BYTES)
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("Unsupported comparison budget")
    # The generic preparation may have another local cohort size. Never widen
    # the actual prepared 600-request assessment ceiling through that input.
    cap = protocol.get("total_assessment_requests")
    if type(cap) is not int or not 1 <= cap <= ASSESSMENT_CAP:
        raise ValueError("Unsupported assessment cap")
    fingerprint = plan.get("fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 \
            or any(c not in "0123456789abcdef" for c in fingerprint):
        raise ValueError("Missing comparison identity")
    return fingerprint, cap


class Bounds:
    """Charge each batch before dispatch and account every elapsed phase second.

    The caller creates this before setup. Initial/shared startup and final
    assessment share one cumulative hour. Each arm owns its startup, TRAIN and
    four monitor passes. A phase switch never resets its accumulated clock.
    No restore, refund, retry or resume API is provided.
    """

    def __init__(self, plan, clock=time.monotonic, resources=None):
        self.fingerprint, self.assessment_cap = _limits(plan)
        if not callable(clock) or not callable(resources):
            raise ValueError("Explicit clock and resource observations required")
        self.clock, self.resources = clock, resources
        start = clock()
        if not _number(start) or start < 0:
            raise ValueError("Invalid initial monotonic clock")
        self.started = self.last_time = float(start)
        self.phase = "shared"
        self.phase_index = 0
        self.elapsed = dict(shared=0., original_usable=0., broader=0.)
        self.requested = {arm: 0 for arm in ARMS}
        self.completed = {arm: 0 for arm in ARMS}
        self.assessment_requested = self.assessment_completed = 0
        self.pending = None
        self.reservations = 0
        self.stopped = None
        self.last_free_bytes = None
        self.check()

    def _stop(self, reason):
        if self.stopped is None:
            self.stopped = reason
        raise BoundsExceeded(self.stopped)

    def _tick(self):
        now = self.clock()
        if not _number(now) or now < self.last_time:
            self._stop("monotonic clock invalid or moved backwards")
        if self.stopped is None:
            self.elapsed[self.phase] += float(now) - self.last_time
            self.last_time = float(now)
        return float(now)

    def check(self):
        if self.stopped is not None:
            raise BoundsExceeded(self.stopped)
        self._tick()
        for phase, elapsed in self.elapsed.items():
            limit = SHARED_WALL_S if phase == "shared" else ARM_WALL_S
            if elapsed >= limit:
                self._stop("wall limit reached: " + phase)
        try:
            observation = self.resources()
        except Exception as error:
            self._stop("resource observation failed: " + type(error).__name__)
        free = observation.get("free_storage_bytes") if isinstance(observation, dict) else None
        if type(free) is not int or free < MIN_FREE_BYTES:
            self._stop("storage below 6 GiB or unverified")
        self.last_free_bytes = free
        return True

    def enter_phase(self, phase):
        self.check()
        if phase == self.phase:
            return
        sequence = ("shared", *ARMS, "shared")
        if self.pending is not None:
            self._stop("cannot switch phase with an unresolved reservation")
        if self.phase_index == 3 or phase != sequence[self.phase_index + 1]:
            self._stop("phase order changed or repeated")
        if self.phase in ARMS and (self.requested[self.phase] != TRAIN_CAP
                                   or self.completed[self.phase] != TRAIN_CAP):
            self._stop("cannot leave an incomplete TRAIN arm")
        self.phase_index += 1
        self.phase = phase

    def reserve(self, kind, count, arm=None):
        self.check()
        if type(count) is not int or not 1 <= count <= 35:
            self._stop("invalid bounded batch count")
        if self.pending is not None:
            self._stop("unresolved reservation; no retry or overlapping dispatch")
        if kind == "train":
            if arm not in ARMS or arm != self.phase:
                self._stop("TRAIN dispatched outside its arm phase")
            if self.requested[arm] + count > TRAIN_CAP:
                self._stop("TRAIN request cap exceeded")
            self.requested[arm] += count
        elif kind == "assessment":
            if arm != (None if self.phase == "shared" else self.phase):
                self._stop("assessment charged to the wrong phase")
            if self.assessment_requested + count > self.assessment_cap:
                self._stop("assessment request cap exceeded")
            self.assessment_requested += count
        else:
            self._stop("unknown request kind")
        reservation = dict(index=self.reservations, kind=kind, count=count, arm=arm)
        self.reservations += 1
        self.pending = reservation
        return copy.deepcopy(reservation)

    def complete(self, reservation):
        if self.stopped is not None:
            raise BoundsExceeded(self.stopped)
        if type(reservation) is not dict or reservation != self.pending:
            self._stop("completion does not match the one pending reservation")
        # Terminal evidence can arrive after its deadline. Record completed work
        # before latching the failure, but never authorize the next batch.
        if reservation["kind"] == "train":
            self.completed[reservation["arm"]] += reservation["count"]
        else:
            self.assessment_completed += reservation["count"]
        self.pending = None
        self.check()

    def fail(self, reason):
        if self.stopped is None:
            self._tick()
            self.stopped = str(reason) or "explicit failure"

    def snapshot(self):
        if self.stopped is None:
            self._tick()
        return dict(schema=SCHEMA, comparison_fingerprint=self.fingerprint,
            started_monotonic=self.started, observed_monotonic=self.last_time,
            active_phase=self.phase, phase_index=self.phase_index,
            phase_elapsed_s=copy.deepcopy(self.elapsed),
            train_requested=copy.deepcopy(self.requested),
            train_completed=copy.deepcopy(self.completed),
            assessment_requested=self.assessment_requested,
            assessment_completed=self.assessment_completed,
            pending=copy.deepcopy(self.pending), reservations=self.reservations,
            stopped=self.stopped, free_storage_bytes=self.last_free_bytes)


class SnapshotWatchdog:
    """Observe a child with an independent clock, even if progress stalls.

    Use only snapshots from an identity-bound child in a future OS supervisor.
    This class does not create that supervisor, signal a process, or establish
    child identity. Its local clock must share the child's monotonic domain.
    """

    def __init__(self, plan, *, started_monotonic, clock=time.monotonic, resources=None):
        self.fingerprint, self.assessment_cap = _limits(plan)
        if not _number(started_monotonic) or started_monotonic < 0 or not callable(resources):
            raise ValueError("Invalid externally owned start/resource observations")
        self.started = started_monotonic
        self.last_time = started_monotonic
        self.clock, self.resources = clock, resources
        self.previous = None
        self.stopped = None

    def check(self, snapshot=None):
        if self.stopped:
            raise BoundsExceeded(self.stopped)
        try:
            return self._check(snapshot)
        except Exception as error:
            self.stopped = str(error)
            raise BoundsExceeded(self.stopped) from error

    def _check(self, snapshot):
        now = self.clock()
        if not _number(now) or now < self.last_time:
            raise ValueError("independent monotonic clock invalid")
        self.last_time = now
        observed = self.resources()
        free = observed.get("free_storage_bytes") if isinstance(observed, dict) else None
        if type(free) is not int or free < MIN_FREE_BYTES:
            raise ValueError("independent storage floor reached")
        if snapshot is not None:
            value = copy.deepcopy(snapshot)
            if value.get("schema") != SCHEMA or value.get("comparison_fingerprint") != self.fingerprint:
                raise ValueError("watchdog snapshot identity changed")
            if value.get("started_monotonic") != self.started or value.get("stopped") is not None:
                raise ValueError("watchdog start changed or child stopped")
            stamp = value.get("observed_monotonic")
            if not _number(stamp) or not self.started <= stamp <= now:
                raise ValueError("watchdog snapshot clock invalid")
            elapsed = value.get("phase_elapsed_s")
            if not isinstance(elapsed, dict) or set(elapsed) != {"shared", *ARMS} \
                    or not all(_number(x) and x >= 0 for x in elapsed.values()) \
                    or not math.isclose(sum(elapsed.values()), stamp-self.started, abs_tol=1e-6):
                raise ValueError("watchdog phase accounting invalid")
            phase_index = value.get("phase_index")
            if type(phase_index) is not int or not 0 <= phase_index <= 3 \
                    or value.get("active_phase") != ("shared", *ARMS, "shared")[phase_index]:
                raise ValueError("watchdog phase order invalid")
            for key in ("train_requested", "train_completed"):
                if not isinstance(value.get(key), dict) or set(value[key]) != set(ARMS) \
                        or any(type(v) is not int or not 0 <= v <= TRAIN_CAP for v in value[key].values()):
                    raise ValueError("watchdog TRAIN counter invalid")
            if any(value["train_completed"][arm] > value["train_requested"][arm] for arm in ARMS):
                raise ValueError("watchdog TRAIN completion exceeds reservation")
            for index, arm in enumerate(ARMS, start=1):
                if phase_index > index and value["train_completed"][arm] != TRAIN_CAP:
                    raise ValueError("watchdog advanced past an incomplete arm")
            for key in ("assessment_requested", "assessment_completed"):
                if type(value.get(key)) is not int or not 0 <= value[key] <= self.assessment_cap:
                    raise ValueError("watchdog assessment counter invalid")
            if value["assessment_completed"] > value["assessment_requested"]:
                raise ValueError("watchdog assessment completion exceeds reservation")
            if self.previous is not None:
                prior = self.previous
                if stamp < prior["observed_monotonic"] or phase_index < prior["phase_index"]:
                    raise ValueError("watchdog progress moved backwards")
                for key in ("train_requested", "train_completed"):
                    if any(value[key][arm] < prior[key][arm] for arm in ARMS):
                        raise ValueError("watchdog TRAIN counter moved backwards")
                if any(value[key] < prior[key] for key in ("assessment_requested", "assessment_completed")) \
                        or any(elapsed[phase] < prior["phase_elapsed_s"][phase] for phase in elapsed):
                    raise ValueError("watchdog cumulative accounting moved backwards")
            self.previous = value
        value = self.previous
        if value is None:
            if now - self.started >= SHARED_WALL_S:
                raise ValueError("watchdog initial shared phase deadline reached")
            return True
        elapsed = copy.deepcopy(value["phase_elapsed_s"])
        elapsed[value["active_phase"]] += now-value["observed_monotonic"]
        if any(elapsed[phase] >= (SHARED_WALL_S if phase == "shared" else ARM_WALL_S)
               for phase in elapsed):
            raise ValueError("independent watchdog wall deadline reached")
        return True
