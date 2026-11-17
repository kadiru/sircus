"""Outcome-blind, offline preparation for a paired TRAIN-location comparison.

This module cannot load a policy, simulate, learn, or authorize execution. It
creates coordinates and a proposed protocol only. Production callers bind its
inputs to the original split, checkpoint, geometry and coordinate inventory.
"""

import copy
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict

SCHEMA = "sircus-seed22-location-comparison-preparation-v1"
BOUNDS = ((-.10, -.70), (1.40, .20))
TRAIN_REQUESTS = 8960
BUFFER = 140
SLOTS = 35
SEPARATION = .025
OFFSET = .050
MAX_DRAWS_PER_POINT = 100000
SEEDS = dict(heldout=2026111601, broad_coordinates=2026111602,
             target_order=2026111603, action_noise=2026111604,
             optimizer=2026111605, local_offsets=2026111606)
PARENT_SHA = "57d7658e9ed2f72ad09cf4e8523ab497ccbe0708364f1b8ec4a4c5c1aef1fe9f"
SPLIT_SHA = "f46c690d80bd2079b27c8d3c72267c39acf1cee61fb4625f37994bff92292785"
GEOMETRY_SHA = "c6489a44e268610470edd2a4c37bd4cfb1a78c89989883113f76996bd8fbad23"


def require(condition, message):
    if not condition:
        raise ValueError(message)


class CoordinateDrawExhausted(ValueError):
    """No point found within the fixed finite geometry-only draw allowance."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def signed(value):
    require("fingerprint" not in value, "Already signed")
    return dict(copy.deepcopy(value), fingerprint=digest(value))


def coordinates(values):
    require(type(values) in (list, tuple), "Expected coordinates")
    result = []
    for point in values:
        require(type(point) in (list, tuple) and len(point) == 2
                and all(type(x) in (int, float) and math.isfinite(x) for x in point),
                "Invalid coordinate pair")
        result.append(tuple(map(float, point)))
    return sorted(set(result))


def _sha_map(values):
    require(type(values) is dict and bool(values), "Missing provenance")
    for name, value in values.items():
        require(type(name) is str and bool(name) and type(value) is str
                and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid provenance SHA256")


def validate_placement(placement):
    """Use explicit conservative enclosures, not distance from a point robot."""
    require(type(placement) is dict, "Missing placement geometry")
    require(set(placement) == {"table_bounds_xy_m", "bucket_radius_m", "bases",
                              "clearance_m", "source_sha256", "method"},
            "Unknown or missing placement field")
    lo, hi = placement["table_bounds_xy_m"]
    coordinates([lo, hi])
    require(all(lo[j] < hi[j] for j in (0, 1)), "Inverted table bounds")
    for name in ("bucket_radius_m", "clearance_m"):
        value = placement[name]
        require(type(value) in (float, int) and math.isfinite(value) and value > 0,
                "Invalid placement radius/clearance")
    require(len(placement["bases"]) == 2, "Both fixed bases required")
    require(len({base["id"] for base in placement["bases"]}) == 2, "Repeated base")
    for base in placement["bases"]:
        require(set(base) == {"id", "xy_m", "radius_m"}, "Invalid base enclosure")
        coordinates([base["xy_m"]])
        require(type(base["radius_m"]) in (int, float) and math.isfinite(base["radius_m"])
                and base["radius_m"] > 0, "Invalid base radius")
    _sha_map(placement["source_sha256"])
    require(type(placement["method"]) is str and placement["method"], "Missing enclosure method")


def placement_check(point, placement):
    """A conservative footprint admission check, not proof of shot reachability.

    Failure means withheld by this enclosure rule. Overlap between two bounding
    circles alone is not proof that their enclosed collision shapes intersect.
    """
    xy = coordinates([point])[0]
    lo, hi = placement["table_bounds_xy_m"]
    radius = placement["bucket_radius_m"]
    table_margin = min(*(xy[j] - lo[j] - radius for j in (0, 1)),
                       *(hi[j] - xy[j] - radius for j in (0, 1)))
    margins = {base["id"]: math.dist(xy, base["xy_m"]) - radius - base["radius_m"]
               for base in placement["bases"]}
    reasons = []
    if not all(BOUNDS[0][j] <= xy[j] < BOUNDS[1][j] for j in (0, 1)):
        reasons.append("outside_original_goal_domain")
    if table_margin < 0:
        reasons.append("bucket_footprint_outside_table")
    reasons.extend("base_enclosure_clearance:" + name for name, margin in margins.items()
                   if margin < placement["clearance_m"])
    return dict(admitted=not reasons, reasons=reasons, table_margin_m=table_margin,
                base_clearance_lower_bounds_m=margins)


class _SeparationIndex:
    """Fixed-radius spatial index, so broad TRAIN generation does not need IK."""

    def __init__(self, points=()):
        self.cells = defaultdict(list)
        for point in points:
            self.add(point)

    def _key(self, point):
        return tuple(math.floor(x / SEPARATION) for x in point)

    def add(self, point):
        self.cells[self._key(point)].append(point)

    def clear(self, point):
        x, y = self._key(point)
        return all(math.dist(point, other) >= SEPARATION
                   for i in range(x - 1, x + 2) for j in range(y - 1, y + 2)
                   for other in self.cells.get((i, j), ()))


def _domain_point(rng):
    return [rng.uniform(BOUNDS[0][j], BOUNDS[1][j]) for j in (0, 1)]


def _tile(point):
    column = min(6, int((point[0] - BOUNDS[0][0]) / 1.5 * 7))
    row = min(4, int((point[1] - BOUNDS[0][1]) / .9 * 5))
    return 7 * row + column


def _original_train(cells):
    require(type(cells) is list and len(cells) == 35, "Expected all 35 original TRAIN cells")
    for i, cell in enumerate(cells):
        require(type(cell) is dict and cell.get("id") == f"train-{i:03d}"
                and cell.get("split") == "train", "Changed original TRAIN order/identity")
        coordinates([cell["bucket_xy_m"]])
    require(len(coordinates([cell["bucket_xy_m"] for cell in cells])) == 35,
            "Duplicate original TRAIN point")
    return copy.deepcopy(cells)


def build_comparison(original_train, history, placement, bindings):
    """Build a fixed, reproducible proposal. No outcomes enter this function."""
    train = _original_train(original_train)
    validate_placement(placement)
    _sha_map(bindings)
    require(bindings.get("parent_checkpoint") == PARENT_SHA, "Changed seed22 parent")
    require(bindings.get("original_split_file") == SPLIT_SHA, "Changed original split")
    require(bindings.get("continuous_geometry_layer") == GEOMETRY_SHA, "Changed shot checker")
    require(bindings.get("original_train") == digest(train), "Changed original TRAIN contents")
    historical = coordinates(list(history) + [cell["bucket_xy_m"] for cell in train])
    require(bindings.get("historical_coordinates") == digest([list(x) for x in historical]),
            "Changed coordinate history")
    require(bindings.get("placement_geometry") == digest(placement), "Changed placement geometry")

    audit = [dict(id=cell["id"], bucket_xy_m=cell["bucket_xy_m"],
                  **placement_check(cell["bucket_xy_m"], placement)) for cell in train]
    eligible = [cell for cell, result in zip(train, audit) if result["admitted"]]
    require(bool(eligible), "No original TRAIN locations admitted")
    occupied = _SeparationIndex(historical)
    draw_reasons = Counter()
    heldout_rng = random.Random(SEEDS["heldout"])

    def draw(propose, *, reserve=True, unique=None, separation_index=None):
        for _ in range(MAX_DRAWS_PER_POINT):
            point = propose()
            result = placement_check(point, placement)
            if not result["admitted"]:
                draw_reasons["placement"] += 1
                continue
            if not (separation_index or occupied).clear(point):
                draw_reasons["history_or_heldout_separation"] += 1
                continue
            if unique is not None and tuple(point) in unique:
                draw_reasons["duplicate_broad_train"] += 1
                continue
            if reserve:
                occupied.add(point)
            return point
        raise CoordinateDrawExhausted("Finite coordinate draw cap reached; do not relax rules or change seed")

    cohorts = {}
    for role, count in (("monitor", 35), ("test_broad", 70)):
        cohorts[role] = []
        for i in range(count):
            point = draw(lambda: _domain_point(heldout_rng))
            cohorts[role].append(dict(id=f"{role}-{i:03d}", split=role,
                                     tile_index=_tile(point), bucket_xy_m=point))
    offset_rng = random.Random(SEEDS["local_offsets"])
    cohorts["test_local"] = []
    unavailable_local_anchors = []
    for cell in eligible:
        def nearby():
            angle = offset_rng.random() * 2 * math.pi
            return [cell["bucket_xy_m"][0] + OFFSET * math.cos(angle),
                    cell["bucket_xy_m"][1] + OFFSET * math.sin(angle)]
        try:
            point = draw(nearby)
        except CoordinateDrawExhausted:
            unavailable_local_anchors.append(dict(anchor_id=cell["id"], offset_m=OFFSET,
                reason="No valid 50mm-offset sample found within finite draw cap after placement/history/heldout exclusions",
                attempted_directions=MAX_DRAWS_PER_POINT))
            continue
        cohorts["test_local"].append(dict(id="test_local-" + cell["id"], split="test_local",
            anchor_id=cell["id"], offset_m=OFFSET, bucket_xy_m=point))

    # Coordinates are fully fixed before policy calls. The action-noise stream
    # is never consumed by coordinate draws, placement or target shuffling.
    coordinate_rng = random.Random(SEEDS["broad_coordinates"])
    heldout_only = _SeparationIndex([cell["bucket_xy_m"]
        for cells in cohorts.values() for cell in cells])
    order_a = random.Random(SEEDS["target_order"])
    a, b, unique = [], [], set(historical)
    while len(a) < TRAIN_REQUESTS:
        cycle = list(eligible)
        order_a.shuffle(cycle)
        a.extend(dict(original_id=c["id"], bucket_xy_m=c["bucket_xy_m"]) for c in cycle)
    a = a[:TRAIN_REQUESTS]
    for _ in range(TRAIN_REQUESTS):
        point = draw(lambda: _domain_point(coordinate_rng), reserve=False, unique=unique,
                     separation_index=heldout_only)
        unique.add(tuple(point))
        b.append(dict(tile_index=_tile(point), bucket_xy_m=point))
    for arm, schedule in (("original_usable", a), ("broader", b)):
        for index, row in enumerate(schedule):
            row.update(id=f"{arm}-{index:05d}", request_index=index, physical_slot=index % SLOTS,
                       batch_index=index // SLOTS, update_index=index // BUFFER)

    milestones = list(range(0, TRAIN_REQUESTS + 1, TRAIN_REQUESTS // 4))
    test_count = len(cohorts["test_broad"]) + len(cohorts["test_local"])
    protocol = dict(
        status="prepared_only_not_launchable", execution_authorized=False, release_authorized=False,
        question="Does broader TRAIN target coverage improve direct seed22 PPO at the same request budget?",
        arms=[dict(id="original_usable", target_count=len(eligible)),
              dict(id="broader", target_count=TRAIN_REQUESTS)],
        proposed_train_requests_per_arm=TRAIN_REQUESTS, proposed_total_train_requests=2 * TRAIN_REQUESTS,
        inherited_per_copy=dict(train_requests=10000, updates=73, optimizer_steps=292),
        shared_learning=dict(restore_weights_and_adam=True, fresh_paired_action_noise=True,
            network="unchanged 5-64-64-5 actor / 5-64-64-1 critic", buffer_requests=BUFFER,
            updates=TRAIN_REQUESTS // BUFFER, epochs=4, learning_rate=.0003, clip=.2,
            value_coef=.5, entropy_coef=.005, grad_norm=1., imitation_steps=0,
            bank=False, old_shot_replay=False,
            likelihood_action="original sampled latent/logp, never the repaired execution settings"),
        shared_environment="Original physics, materials, reward, action bounds, shot geometry wrapper, motion/settling/release guards",
        accounting="Invalid bucket placements are filtered before schedules exist. All subsequent policy requests including geometry/native rejections cost one; no retries/refunds.",
        target_selection="A balances admitted original locations within one request; B uses seeded uniform domain draws conditioned on placement and 25mm reserved heldout bands; excludes exact historical coordinates but no distance band around old TRAIN; no outcomes or IK in target selection",
        heldout_selection="Seeded uniform domain draws conditioned on placement and 25mm separation from known history and each other; secondary local cohort attempts one 50mm offset per admitted original anchor. Unsampled anchors are recorded, never replaced or given relaxed spacing. Report this conditional spatial scope, not uniform coverage of the entire rectangle.",
        train_order=["original_usable", "broader"],
        order_limitation="Fixed sequential A then B may confound wall time with host conditions; no causal speed claim",
        monitor_milestones=milestones, monitor_positions=35,
        monitor_requests=35 * (1 + 2 * (len(milestones) - 1)),
        monitor_baseline="One shared untouched-parent+checker measurement at zero",
        selection="Final terminal checkpoint at exactly 8960 requests; monitor results cannot select, stop, or tune",
        proposed_wall_seconds_per_training_arm=28800,
        proposed_shared_assessment_wall_seconds=3600,
        proposed_cpu_audit_wall_seconds=7200,
        minimum_free_storage_bytes=6 * 1024**3,
        incomplete_rule="Any time/resource/failure stop before exact budgets marks the full comparison incomplete. Preserve all charged work; no automatic restart or TEST of unmatched terminal models.",
        budget_status="All bounds are proposals, not reuse of the completed 30000/8h authorization. Each eight-hour arm includes startup and its four monitor passes; one additional shared hour covers baseline/final assessment, and two CPU-only hours bound audit. No execution is authorized.",
        evaluation_models=["untouched_parent_plus_checker", "original_usable_final_plus_checker", "broader_final_plus_checker"],
        test_positions=dict(broad=70, local=test_count - 70),
        final_assessment_requests=3 * test_count,
        total_assessment_requests=35 * (1 + 2 * (len(milestones) - 1)) + 3 * test_count,
        final_test_rule="Freeze all checkpoint SHAs before TEST; one deterministic mean shot/model/position, fresh reset, rotate model order by target index, no search/retry/update",
        primary_metric="captures / all 70 broad TEST requests; rejections are failures",
        secondary_metrics=["clean captures / all requests", "released misses", "rejections by cause",
            "local-offset cohort results separately", "conditional capture/release labelled separately",
            "phase-separated proposal, geometry, native rollout, PPO, checkpoint, monitor, TEST and audit time",
            "RSS, sampled GPU memory, requests/minute and releases/minute; no double-counting nested time"],
        uncertainty="One paired continuation from one seed22 parent; not three-seed reliability or hardware validation",
        native_video_schedule="First native batch after each monitor boundary; preserve all selected outcomes",
        prerequisite="A new production adapter, independent audit and bounded launcher must validate this schema; old fixed35 launcher cannot consume it")
    return signed(dict(schema=SCHEMA, bindings=bindings, seeds=SEEDS,
        original_train=train, placement_geometry=placement, original_placement_audit=audit,
        historical_coordinates=[list(x) for x in historical], cohorts=cohorts,
        unavailable_local_anchors=unavailable_local_anchors,
        schedules=dict(original_usable=a, broader=b), protocol=protocol,
        preparation_rejected_draws=dict(sorted(draw_reasons.items())),
        native_requests=0, optimizer_steps=0, outcome_fields_used=False))


def validate_comparison(plan, *, original_train, history, placement, bindings):
    """Externally rebound regeneration, not merely trust in a resealed JSON."""
    require(plan == build_comparison(original_train, history, placement, bindings),
            "Comparison differs from externally bound deterministic preparation")
    return plan["fingerprint"]


def summarize(plan):
    audit = plan["original_placement_audit"]
    a = plan["schedules"]["original_usable"]
    counts = Counter(x["original_id"] for x in a)
    return dict(schema=SCHEMA, fingerprint=plan["fingerprint"], bindings=plan["bindings"],
        original_locations=len(audit), admitted_original_locations=len(counts),
        withheld_original_locations=[row for row in audit if not row["admitted"]],
        original_requests_per_location_range=[min(counts.values()), max(counts.values())],
        broader_unique_locations=len({tuple(row["bucket_xy_m"]) for row in plan["schedules"]["broader"]}),
        historical_coordinate_count=len(plan["historical_coordinates"]),
        unavailable_local_anchors=plan["unavailable_local_anchors"],
        tile_counts={key: dict(sorted(Counter(row["tile_index"] for row in value).items()))
                     for key, value in list(plan["cohorts"].items())[:2]},
        broader_tile_counts=dict(sorted(Counter(row["tile_index"] for row in plan["schedules"]["broader"]).items())),
        cohort_fingerprints={key: digest(value) for key, value in plan["cohorts"].items()},
        schedule_fingerprints={key: digest(value) for key, value in plan["schedules"].items()},
        protocol=plan["protocol"], native_requests=0, optimizer_steps=0)
