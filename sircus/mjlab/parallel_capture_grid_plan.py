"""Seed local searches at the 13 least-covered fixed bucket locations."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from itertools import count, product
import json
import math
from pathlib import Path

import numpy as np

from .benchmark_verify import contained, read, sha
from .cell1050_followup_plan import geometry_key
from .corrected_contract import grid_layouts
from .expanded_actions import WIDE_BOUNDS
from .table_assisted_score import ROUTE_VERSION

VERSION = "sircus-parallel-capture-grid-plan-v1"
TARGET_COUNT = 25
BASELINE_COUNT = 911
BASELINE_STUDIES = (
    "bouncer-clearance-v1", "case00-tilt-gap-v1", "corrected-imitation-v1",
    "coverage-recovery-v1", "coverage-training7h-v2", "fixed35-bouncer-v1",
    "fixed35-broad1000-v1", "fixed35-imitation-v1", "fixed35-recovery-v1",
    "fixed35-six-v1", "fixed35-sixhour-v2", "gripper-yaw-v1",
    "recovery-regression-v1", "regression-drop-gap-v1", "regression-ik-v1",
    "table-height-tilt-v1", "table-training-tilt-correction-v1",
    "table-training-v1", "yaw-tilt-gap-v1", "fixed35-corner-sweep-v1",
    "cell1050-rebound-v1", "cell1050-followup-v1",
)
FIXED_BASE_BLOCKS = {"fixed35-dev-grid-03-00", "fixed35-dev-grid-03-06"}
MAX_SEEDS = 4
MAX_HEADING_DEG = 45.0
SEED_DISTANCE_SCALE = np.array([.008, .008, .008, .8, .010])


def _grid_key(xy):
    return tuple(round(value * 1000, 7) for value in xy)


def _cell_id(layout):
    return "fixed35-" + layout["id"]


def _archive_manifest(directory, ledger_sha256):
    report_path = directory / "report.json"
    manifest_path = directory / "evidence-manifest.json"
    report = read(report_path)
    manifest = read(manifest_path)
    manifest_sha256 = sha(manifest_path)
    if (
        report.get("status") != "completed"
        or manifest.get("rows.jsonl") != ledger_sha256
        or not (
            report.get("evidence_sha256") == manifest
            or report.get("evidence_manifest_sha256") == manifest_sha256
        )
    ):
        raise ValueError(f"The completed report does not bind the ledger for {directory.name}")
    return manifest, {
        "report_sha256": sha(report_path),
        "manifest_sha256": manifest_sha256,
        "ledger_sha256": ledger_sha256,
    }


def _joint_vectors(row):
    q10 = row.get("selected_q10")
    q14 = row.get("selected_q14")
    if q10 is None:
        q10 = row.get("q10")
    if q14 is None:
        q14 = row.get("q14")
    first, second = np.asarray(q10, dtype=float), np.asarray(q14, dtype=float)
    if first.shape != (6,) or second.shape != (6,):
        return None
    q = np.stack((first, second))
    return q if np.isfinite(q).all() else None


def _joint_key(q):
    values = np.round((q.ravel() + np.pi) % (2 * np.pi) - np.pi, 4)
    values[values == round(np.pi, 4)] = round(-np.pi, 4)
    return tuple(values.tolist())


def _valid_capture(row):
    return (
        row.get("native_release") is True
        and row.get("route_version") == ROUTE_VERSION
        and row.get("route_success") is True
        and not row.get("invalid")
        and not row.get("invalid_physics")
    )


def _seed_rank(seed, observations):
    clearance = seed["minimum_mouth_clearance_mm"]
    return (
        observations["miss"] > 0,
        -observations["capture"],
        -seed["route_quality"],
        -(clearance if clearance is not None else -math.inf),
        seed["source_study"], seed["id"],
    )


def _seed_record(row, directory, raw, manifest, joints):
    adapter = row.get("adapter_directory")
    cache = row.get("native_cache_file")
    if not adapter or not cache:
        return None
    trace = (Path(adapter) / cache).as_posix()
    trace_sha256 = row.get("native_cache_sha256")
    if not trace_sha256 or trace_sha256 != manifest.get(trace):
        return None
    return {
        "id": row["id"],
        "values": row["values"],
        "bucket_xy_m": row["bucket_xy_m"],
        "q10": joints[0].tolist(),
        "q14": joints[1].tolist(),
        "board_heading_offset_deg": row.get("board_heading_offset_deg", 0.0),
        "board_twist_deg": row.get("board_twist_deg", 0.0),
        "route_quality": row.get("route_quality", 0),
        "route_class": row.get("route_class"),
        "minimum_mouth_clearance_mm": row.get("minimum_mouth_clearance_mm"),
        "source_study": directory.name,
        "source_phase": row["phase"],
        "source_row_sha256": hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest(),
        "source_trace": trace,
        "source_trace_sha256": trace_sha256,
        "source_trace_column": row["native_cache_row"],
        "context_resolution_file": row.get("context_resolution_file"),
        "context_resolution_sha256": row.get("context_resolution_sha256"),
        "connection_file": row.get("connection_file"),
        "connection_sha256": row.get("connection_sha256"),
    }


def _best_seeds(records, observations):
    """Favor repeated captures and avoid filling seed slots with one geometry."""
    candidates = sorted(records.items(), key=lambda item: _seed_rank(item[1], observations[item[0]]))
    chosen, deferred = [], []
    used_settings = set()
    for command, seed in candidates:
        settings = tuple(seed["values"])
        if settings in used_settings:
            continue
        used_settings.add(settings)
        seed = dict(seed, observed=dict(observations[command]))
        values = np.asarray(settings)
        if any(np.linalg.norm((values - prior["values"]) / SEED_DISTANCE_SCALE) < 1 for prior in chosen):
            deferred.append(seed)
        else:
            chosen.append(seed)
        if len(chosen) == MAX_SEEDS:
            break
    return (chosen + deferred)[:MAX_SEEDS]


def load_grid_plan(output_root: Path):
    """Bind baseline captures and exclude old geometries at each exact grid point."""
    root = Path(output_root)
    layouts = {
        _grid_key(layout["bucket_xy_m"]): layout for layout in grid_layouts()
        if _cell_id(layout) not in FIXED_BASE_BLOCKS
    }
    baseline_studies = set(BASELINE_STUDIES)
    ledgers = {path.parent.name: path for path in root.glob("*/rows.jsonl")}
    missing = baseline_studies - ledgers.keys()
    if missing:
        raise ValueError("Missing completed grid ledgers: " + ", ".join(sorted(missing)))

    source_bindings, history_bindings, preparation_bindings = {}, {}, {}
    manifests = {}
    known_settings, excluded = defaultdict(set), defaultdict(set)
    records, observations = defaultdict(dict), defaultdict(Counter)
    for study, path in sorted(ledgers.items()):
        checksum = sha(path)
        history_bindings[str(path.relative_to(root))] = checksum
        if study in baseline_studies:
            manifests[study], source_bindings[study] = _archive_manifest(path.parent, checksum)
        with path.open("rb") as stream:
            for raw in stream:
                row = json.loads(raw)
                xy = row.get("bucket_xy_m")
                values = row.get("values")
                if xy is None or values is None or len(xy) != 2 or len(values) != 5:
                    continue
                key = _grid_key(xy)
                if key not in layouts or not all(math.isfinite(value) for value in values):
                    continue
                geometry = geometry_key(row)
                excluded[key].add(geometry)
                if study not in baseline_studies:
                    continue
                captured = _valid_capture(row)
                if captured:
                    known_settings[key].add(tuple(values))
                if not row.get("native_release") or row.get("route_version") != ROUTE_VERSION:
                    continue
                joints = _joint_vectors(row)
                if joints is None:
                    continue
                command = geometry + _joint_key(joints)
                outcome = observations[key].setdefault(command, Counter())
                outcome["capture" if captured else "miss"] += 1
                if captured and row.get("contact_protocol", {}).get("substeps") == 64:
                    seed = _seed_record(row, path.parent, raw, manifests[study], joints)
                    if seed is not None:
                        previous = records[key].get(command)
                        if previous is None or _seed_rank(seed, outcome) < _seed_rank(previous, outcome):
                            records[key][command] = seed

    for path in sorted(root.glob("*/preparations/*.json")):
        study = path.parent.parent.name
        relative_path = str(path.relative_to(path.parent.parent))
        checksum = sha(path)
        if study in manifests and manifests[study].get(relative_path) != checksum:
            raise ValueError(f"Historical preparation changed: {study}/{relative_path}")
        preparation_bindings[str(path.relative_to(root))] = checksum
        for proposal in read(path).get("proposals", []):
            key = _grid_key(proposal["bucket_xy_m"])
            if key in layouts:
                excluded[key].add(geometry_key(proposal))

    total = sum(map(len, known_settings.values()))
    if total != BASELINE_COUNT or len(known_settings) != len(layouts):
        raise ValueError(f"Expected {BASELINE_COUNT} captured settings across the 33 available cells, found {total}")

    cells = {}
    verified_traces = {}
    for key, layout in layouts.items():
        settings = known_settings[key]
        if len(settings) >= TARGET_COUNT:
            continue
        seeds = _best_seeds(records[key], observations[key])
        if not seeds:
            raise ValueError(f"No bound native64 capture to seed {_cell_id(layout)}")
        for seed in seeds:
            source = root / seed["source_study"]
            trace = contained(source, seed["source_trace"])
            if trace not in verified_traces:
                verified_traces[trace] = sha(trace)
            if verified_traces[trace] != seed["source_trace_sha256"]:
                raise ValueError(f"Recorded native trace changed for seed {seed['id']}")
        context = _cell_id(layout)
        cells[context] = {
            "context_id": context,
            "bucket_xy_m": layout["bucket_xy_m"],
            "prior_count": len(settings),
            "prior_settings": sorted(settings),
            "target_count": TARGET_COUNT,
            "seeds": seeds,
            "excluded_geometry": excluded[key],
        }
    if len(cells) != 13:
        raise ValueError(f"Expected the 13 under-covered locations, found {len(cells)}")
    return cells, {
        "version": VERSION,
        "baseline_count": total,
        "baseline_sources": source_bindings,
        "history_ledger_sha256": history_bindings,
        "preparation_sha256": preparation_bindings,
    }


def _axis_probes(seeds):
    """Keep the same arm hint and change one physical shot coordinate at a time."""
    scales = (
        (.002, .002, .002, .25, .003),
        (.005, .005, .005, .50, .008),
        (.010, .010, .010, 1.0, .015),
    )
    for magnitudes in scales:
        for axis, direction in product(range(5), (-1, 1)):
            for source in seeds:
                values = np.asarray(source["values"], dtype=float).copy()
                values[axis] += direction * magnitudes[axis]
                yield source, values, 0.0, 0.0, "single_parameter"
        # A height/gap pair moves the bounce while keeping the release height fixed.
        for direction, source in product((-1, 1), seeds):
            values = np.asarray(source["values"], dtype=float).copy()
            delta = direction * magnitudes[2]
            values[2] += delta
            values[4] -= delta
            yield source, values, 0.0, 0.0, "fixed_release_height"


def _widening_probes(seeds, seed):
    rng = np.random.default_rng(seed)
    for index in count():
        source = seeds[index % len(seeds)]
        cycle = index // len(seeds)
        width = (.006, .012, .024, .040)[cycle % 4]
        base = np.asarray(source["values"], dtype=float)
        values = base + rng.uniform(-1, 1, 5) * [width, width, width, width * 100, width]
        if cycle % 4 < 2:
            yield source, values, 0.0, 0.0, "local_combination"
        else:
            heading = rng.uniform(-1, 1) * width * 100
            twist = rng.choice([0.0, 0.0, -6.0, 6.0, -12.0, 12.0])
            yield source, values, heading, twist, "local_orientation"


def cell_candidates(cell, *, seed):
    """Yield deterministic HeadingEngine proposals for one canonical bucket cell."""
    context = cell["context_id"]
    bucket = list(cell["bucket_xy_m"])
    low, high = np.asarray(WIDE_BOUNDS["low"]), np.asarray(WIDE_BOUNDS["high"])
    tried = {tuple(key) for key in cell["excluded_geometry"]}
    known = {tuple(settings) for settings in cell["prior_settings"]}
    index = 0
    for stream in (_axis_probes(cell["seeds"]), _widening_probes(cell["seeds"], seed)):
        for source, values, heading_delta, twist_delta, family in stream:
            heading = source["board_heading_offset_deg"] + heading_delta
            twist = source["board_twist_deg"] + twist_delta
            if (
                not ((low <= values) & (values <= high)).all()
                or np.linalg.norm(values[:2] - bucket) <= .15
                or abs(heading) > MAX_HEADING_DEG
                or tuple(values) in known
            ):
                continue
            proposal = dict(
                id=f"capture-grid-{context.removeprefix('fixed35-dev-grid-')}-p{index:06d}",
                proposal_index=index, context_id=context, bucket_xy_m=bucket.copy(),
                phase="search", method="parallel_capture_grid", role="development",
                values=values.tolist(), board_heading_offset_deg=float(heading),
                board_twist_deg=float(twist), group=family,
                seed_id=source["id"], seed_q10=source["q10"], seed_q14=source["q14"],
                hint={key: source[key] for key in ("id", "values", "bucket_xy_m", "q10", "q14")},
                source_study=source["source_study"], source_row_sha256=source["source_row_sha256"],
                source_trace_sha256=source["source_trace_sha256"],
            )
            key = geometry_key(proposal)
            if key in tried:
                continue
            tried.add(key)
            yield proposal
            index += 1
