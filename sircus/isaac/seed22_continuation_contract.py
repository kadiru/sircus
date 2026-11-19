"""Pure accounting and outcome-blind layouts for direct seed-22 continuation.

No policy, geometry solver, simulator, native process or file writer is imported.
Every new TRAIN request, including a geometry decline, spends one request.  A
reservation is returned before dispatch and must be completed exactly once.
This contract cannot authorize execution, overwrite evidence or resume a run.
"""

import copy
import hashlib
import json
import math
import random
import re


SCHEMA = "sircus-seed22-continuation-protocol-v1"
SPLIT_SCHEMA = "sircus-seed22-continuation-layouts-v1"
LEDGER_SCHEMA = "sircus-seed22-continuation-ledger-v1"
TRAIN_REQUEST_CAP = 30000
WALL_SECONDS = 28800
ASSESSMENT_REQUEST_CAP = 525
NATIVE_SLOTS = 35
PARENT_SEED = 20261122
PARENT_REQUESTS = 10000
PARENT_UPDATES = 73
PARENT_OPTIMIZER_STEPS = 292
SPLIT_SEED = 20261130
TRAIN_ORDER_SEED = 20261130
MONITOR_MILESTONES = (0, 1000, 2000, 5000, 10000, 20000, 30000)
BOUNDS = ((-.10, -.70), (1.40, .20))
MIN_SEPARATION_M = .025
MAX_DRAWS = 1000000
PARENT_CHECKPOINT_SHA256 = "57d7658e9ed2f72ad09cf4e8523ab497ccbe0708364f1b8ec4a4c5c1aef1fe9f"
ORIGINAL_EXPERIMENT_SHA256 = "77afe173a07917dd87ce43f37809a9ac6ef99beebd7e8e35f703c0e5cf1e3a26"
ORIGINAL_SPLIT_SHA256 = "f46c690d80bd2079b27c8d3c72267c39acf1cee61fb4625f37994bff92292785"
GEOMETRY_LAYER_SHA256 = "c6489a44e268610470edd2a4c37bd4cfb1a78c89989883113f76996bd8fbad23"
ZERO = "0" * 64
ROLES = ("train", "validation", "test")
CELL_FIELDS = {"id", "split", "bucket_xy_m", "tile_index", "grid_row", "grid_column"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def signed(value):
    result = copy.deepcopy(value)
    require("fingerprint" not in result, "Cannot sign an already signed object")
    result["fingerprint"] = canonical_hash(result)
    return result


def validate_fingerprint(value):
    require(type(value) is dict and value.get("fingerprint") == canonical_hash(
        {key: item for key, item in value.items() if key != "fingerprint"}), "Changed artifact fingerprint")
    return value["fingerprint"]


def _sha(value, label):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid " + label + " SHA256")
    return value


def make_protocol(parent_checkpoint_sha256, original_experiment_sha256,
                  original_split_sha256, geometry_layer_sha256, source_sha256,
                  exclusions_fingerprint):
    """Freeze the exact parent, unchanged geometry layer, new source and caps."""
    expected = ((parent_checkpoint_sha256, PARENT_CHECKPOINT_SHA256, "parent checkpoint"),
                (original_experiment_sha256, ORIGINAL_EXPERIMENT_SHA256, "original experiment"),
                (original_split_sha256, ORIGINAL_SPLIT_SHA256, "original split"),
                (geometry_layer_sha256, GEOMETRY_LAYER_SHA256, "geometry layer"))
    for actual, wanted, label in expected:
        require(_sha(actual, label) == wanted, "Changed " + label)
    _sha(exclusions_fingerprint, "coordinate exclusion fingerprint")
    require(type(source_sha256) is dict and bool(source_sha256), "Missing source membership")
    for name, digest in source_sha256.items():
        require(type(name) is str and name and not name.startswith("/")
                and ".." not in name.split("/"), "Invalid source member path")
        _sha(digest, "source member")
    return signed(dict(schema=SCHEMA, parent_seed=PARENT_SEED,
        inherited=dict(requests=PARENT_REQUESTS, updates=PARENT_UPDATES,
                       optimizer_steps=PARENT_OPTIMIZER_STEPS),
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        original_experiment_sha256=original_experiment_sha256,
        original_split_sha256=original_split_sha256, geometry_layer_sha256=geometry_layer_sha256,
        source_sha256=dict(sorted(source_sha256.items())), exclusions_fingerprint=exclusions_fingerprint,
        additional_train_request_cap=TRAIN_REQUEST_CAP, absolute_wall_seconds=WALL_SECONDS,
        assessment_request_cap=ASSESSMENT_REQUEST_CAP, native_slots=NATIVE_SLOTS,
        monitor_milestones=list(MONITOR_MILESTONES), train_order_seed=TRAIN_ORDER_SEED,
        split_seed=SPLIT_SEED, train_contexts=35, validation_contexts=35, test_contexts=70,
        minimum_separation_m=MIN_SEPARATION_M, min_free_bytes=6 * 1024**3,
        min_available_ram_bytes=1024**3, min_free_gpu_bytes=512*1024**2,
        final_assessment_reserve_seconds=1200,
        learning=dict(method="original continuous terminal PPO", buffer_target=140, epochs=4,
                      learning_rate=.0003, clip=.2, value_coef=.5, entropy_coef=.005, grad_norm=1.,
                      restore_parent_optimizer=True, imitation_steps=0,
                      likelihood_action="Original sampled latent, never adjusted execution settings"),
        accounting="Every reserved TRAIN request including all geometry/native declines; no replacement or retry",
        selection="Monitoring is diagnostic; final artifact follows the first fixed cap/time stop, never the best score",
        geometry="Frozen deterministic geometry wrapper; original proposal and executed action remain distinct",
        execution_authorized=False, release_authorized=False))


def validate_protocol(protocol):
    validate_fingerprint(protocol)
    fields = ("parent_checkpoint_sha256", "original_experiment_sha256", "original_split_sha256",
              "geometry_layer_sha256", "source_sha256", "exclusions_fingerprint")
    require(all(key in protocol for key in fields), "Missing protocol binding")
    require(protocol == make_protocol(*(protocol[key] for key in fields)), "Unsupported continuation protocol")
    return protocol["fingerprint"]


def _coordinates(values):
    require(type(values) in (list, tuple), "Expected coordinate pairs")
    result = []
    for point in values:
        require(type(point) in (list, tuple) and len(point) == 2
                and all(type(x) in (int, float) and math.isfinite(x) for x in point),
                "Invalid coordinate pair")
        result.append((float(point[0]), float(point[1])))
    return sorted(set(result))


def _train(cells):
    require(type(cells) in (list, tuple) and len(cells) == 35, "Exactly 35 original TRAIN cells required")
    result = copy.deepcopy(list(cells))
    for index, cell in enumerate(result):
        require(type(cell) is dict and {"id", "split", "bucket_xy_m"}.issubset(cell)
                and set(cell).issubset(CELL_FIELDS) and cell["id"] == f"train-{index:03d}"
                and cell["split"] == "train", "Original TRAIN membership/order changed")
        point = _coordinates([cell["bucket_xy_m"]])[0]
        require(all(BOUNDS[0][j] <= point[j] < BOUNDS[1][j] for j in (0, 1)),
                "Original TRAIN coordinate outside bounds")
        if "tile_index" in cell:
            require(type(cell["tile_index"]) is int and cell["tile_index"] == index,
                    "Original TRAIN tile changed")
        for field, wanted in (("grid_row", index // 7), ("grid_column", index % 7)):
            if field in cell:
                require(type(cell[field]) is int and cell[field] == wanted, "Original TRAIN grid changed")
    points = _coordinates([cell["bucket_xy_m"] for cell in result])
    require(len(points) == 35, "Repeated original TRAIN coordinate")
    require(all(math.dist(point, old) >= MIN_SEPARATION_M
                for i, point in enumerate(points) for old in points[:i]), "Original TRAIN separation changed")
    return result


def create_fresh_splits(original_train, excluded_coordinates, *, split_seed=SPLIT_SEED,
                        max_draws=MAX_DRAWS):
    """Retain TRAIN exactly and draw 35 monitoring/70 TEST sites without outcomes."""
    train = _train(original_train)
    require(type(split_seed) is int and split_seed == SPLIT_SEED, "Changed continuation split seed")
    require(type(max_draws) is int and 105 <= max_draws <= MAX_DRAWS, "Invalid finite draw cap")
    excluded = _coordinates(list(_coordinates(excluded_coordinates)) +
                            [cell["bucket_xy_m"] for cell in train])
    previous = list(excluded)
    rng = random.Random(split_seed)
    groups, draws = {}, 0
    width, height = (BOUNDS[1][0] - BOUNDS[0][0]) / 7, (BOUNDS[1][1] - BOUNDS[0][1]) / 5
    for role, count in (("validation", 35), ("test", 70)):
        groups[role] = []
        for index in range(count):
            tile = index % 35
            row, column = divmod(tile, 7)
            lower = (BOUNDS[0][0] + column * width, BOUNDS[0][1] + row * height)
            while draws < max_draws:
                point = (lower[0] + rng.random() * width, lower[1] + rng.random() * height)
                draws += 1
                if any(math.dist(point, old) < MIN_SEPARATION_M for old in previous):
                    continue
                previous.append(point)
                groups[role].append(dict(id=f"continuation-{role}-{index:03d}", split=role,
                    bucket_xy_m=list(point), tile_index=tile, grid_row=row, grid_column=column))
                break
            require(len(groups[role]) == index + 1,
                    "Fresh coordinate draw cap reached; no spacing relaxation or tile replacement")
    return signed(dict(schema=SPLIT_SCHEMA, seed=split_seed, bounds_xy_m=[list(x) for x in BOUNDS],
        minimum_separation_m=MIN_SEPARATION_M, max_draws=max_draws, draws=draws,
        original_train_fingerprint=canonical_hash(train),
        excluded_coordinates=[list(x) for x in excluded],
        exclusions_fingerprint=canonical_hash([list(x) for x in excluded]),
        selection="Seeded uniform coordinates in fixed 7x5 tiles; separation only, no IK or outcome filtering",
        train=train, validation=groups["validation"], test=groups["test"],
        outcome_files_read=0, native_requests=0, optimizer_steps=0,
        execution_authorized=False, release_authorized=False))


def validate_splits(manifest, *, original_train=None, excluded_coordinates=None):
    """Rebuild exact geometry; callers bind the original TRAIN/history externally."""
    validate_fingerprint(manifest)
    require(manifest.get("schema") == SPLIT_SCHEMA, "Unknown continuation layout schema")
    train = manifest.get("train") if original_train is None else original_train
    history = manifest.get("excluded_coordinates") if excluded_coordinates is None else excluded_coordinates
    rebuilt = create_fresh_splits(train, history, split_seed=manifest.get("seed"),
                                  max_draws=manifest.get("max_draws"))
    require(manifest == rebuilt, "Continuation layout, original TRAIN or exclusion history changed")
    return manifest["fingerprint"]


def batch_schedule(max_requests=TRAIN_REQUEST_CAP, seed=PARENT_SEED, milestones=MONITOR_MILESTONES):
    """Additional-only schedule, cut at milestones and each 35-context round.

    The parent label remains seed22.  A separate documented order stream starts
    here; it does not replay the original training order or its old proposals.
    """
    require(type(max_requests) is int and 1 <= max_requests <= TRAIN_REQUEST_CAP, "Invalid additional TRAIN cap")
    require(type(seed) is int and seed == PARENT_SEED, "Changed parent seed")
    require(tuple(milestones) == MONITOR_MILESTONES, "Changed monitoring milestones")
    rng = random.Random((TRAIN_ORDER_SEED << 32) | seed)
    order = []
    while len(order) < max_requests:
        row = list(range(NATIVE_SLOTS))
        rng.shuffle(row)
        order.extend(row)
    order = order[:max_requests]
    result, start = [], 0
    while start < max_requests:
        next_monitor = next((x for x in MONITOR_MILESTONES if x > start), max_requests)
        stop = min(start + NATIVE_SLOTS - start % NATIVE_SLOTS, next_monitor, max_requests)
        indices = order[start:stop]
        result.append(dict(call_index=len(result), batch_index=len(result), logical_round=start // NATIVE_SLOTS,
            attempt_start=start, attempt_stop=stop, attempts=stop - start,
            site_indices=indices, enabled=[index in indices for index in range(NATIVE_SLOTS)],
            milestone_after=stop if stop in MONITOR_MILESTONES else None))
        start = stop
    return result


class Ledger:
    """In-memory one-use accounting; caller must persist reserve receipts first.

    No restore/from-dict API exists.  This class performs no file or native I/O.
    Runtime persistence is responsible for fsync before exposing a reservation.
    """

    def __init__(self, session):
        require(type(session) is str and re.fullmatch(r"[A-Za-z0-9_-]{1,96}", session), "Invalid unique session")
        self.session = session
        self.pending = None
        self.closed = False
        self.stop_reason = None
        self.train_reserved = self.train_terminal = 0
        self.assessment_reserved = self.assessment_terminal = 0
        self._used = set()
        self._events = []
        self._head = ZERO

    def _event(self, kind, payload):
        receipt = signed(dict(schema=LEDGER_SCHEMA, session=self.session, seq=len(self._events),
                              previous_sha256=self._head, kind=kind, payload=copy.deepcopy(payload)))
        self._events.append(receipt)
        self._head = receipt["fingerprint"]
        return copy.deepcopy(receipt)

    def reserve(self, batch_index, role, enabled_count):
        require(not self.closed and self.pending is None, "Ledger closed or reservation still pending")
        require(type(batch_index) is int and batch_index >= 0 and role in ROLES, "Invalid batch identity or role")
        require((role, batch_index) not in self._used, "Repeated reserved batch")
        require(type(enabled_count) is int and 1 <= enabled_count <= NATIVE_SLOTS, "Invalid paid slot count")
        current, cap = ((self.train_reserved, TRAIN_REQUEST_CAP) if role == "train" else
                        (self.assessment_reserved, ASSESSMENT_REQUEST_CAP))
        require(current + enabled_count <= cap, "Request cap exceeded")
        request_ids = [f"{self.session}:{role}:{batch_index:06d}:{slot:02d}" for slot in range(enabled_count)]
        self.pending = dict(batch_index=batch_index, role=role, enabled_count=enabled_count, request_ids=request_ids)
        self._used.add((role, batch_index))
        if role == "train":
            self.train_reserved += enabled_count
        else:
            self.assessment_reserved += enabled_count
        return self._event("reserve", self.pending)

    def complete(self, batch_index, role, terminal_count):
        require(not self.closed and self.pending is not None, "No open reservation")
        require(type(terminal_count) is int and terminal_count == self.pending["enabled_count"]
                and type(batch_index) is int and batch_index == self.pending["batch_index"]
                and role == self.pending["role"], "Terminal outcomes do not exactly cover the reserved batch")
        completed = copy.deepcopy(self.pending)
        completed["terminal_count"] = terminal_count
        if role == "train":
            self.train_terminal += terminal_count
        else:
            self.assessment_terminal += terminal_count
        self.pending = None
        return self._event("complete", completed)

    def finish(self, reason):
        require(not self.closed and self.pending is None, "Cannot finish closed or unresolved ledger")
        require(type(reason) is str and bool(reason.strip()), "Stop reason required")
        self.closed, self.stop_reason = True, reason
        return self._event("finish", dict(reason=reason))

    def fail(self, reason, *, terminal_count=0):
        """Close permanently; a pending prefix remains charged, never refundable."""
        require(not self.closed and type(reason) is str and bool(reason.strip()), "Ledger closed or missing failure reason")
        count = self.pending["enabled_count"] if self.pending else 0
        require(type(terminal_count) is int and 0 <= terminal_count <= count, "Invalid failed terminal prefix")
        if self.pending:
            if self.pending["role"] == "train":
                self.train_terminal += terminal_count
            else:
                self.assessment_terminal += terminal_count
            self.pending["terminal_count"] = terminal_count
        self.closed, self.stop_reason = True, reason
        return self._event("failure", dict(reason=reason, pending=copy.deepcopy(self.pending)))

    def snapshot(self):
        return signed(dict(schema=LEDGER_SCHEMA, session=self.session, closed=self.closed,
            stop_reason=self.stop_reason, inherited=dict(seed=PARENT_SEED, requests=PARENT_REQUESTS,
                updates=PARENT_UPDATES, optimizer_steps=PARENT_OPTIMIZER_STEPS),
            train_reserved=self.train_reserved, train_terminal=self.train_terminal,
            train_unresolved=self.train_reserved - self.train_terminal,
            assessment_reserved=self.assessment_reserved, assessment_terminal=self.assessment_terminal,
            assessment_unresolved=self.assessment_reserved - self.assessment_terminal,
            pending=copy.deepcopy(self.pending), event_count=len(self._events), journal_head=self._head))

    def history(self):
        return copy.deepcopy(self._events)
