"""Geometry-only Sobol feasibility survey for a separate continuous shot task.

No capture outcomes enter sampling, IK qualification, splits or action bounds.
A solved endpoint and coarse capsule check are not a collision-certified path.
Numerical IK failures remain unresolved, never proofs of impossibility.
"""

import copy
import concurrent.futures
import hashlib
import json
import math
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from expanded_pose_menu import endpoint_targets, outer_bounds
from expanded_pose_native import verified_fit
from prepare_assets import chain, fk, matrix
from robust_ik import normal_jacobian

VERSION = "sircus-continuous-feasibility-v1"
SEED = 20261112
ACTION_NAMES = ("bearing_offset_rad", "radius_m", "height_m", "gap_m", "tilt_rad")
LOW = np.array([-math.pi, 0.28, 0.14, 0.08, math.radians(12.0)])
HIGH = np.array([math.pi, 1.20, 0.60, 0.56, math.radians(52.0)])
VALIDATION_INDICES = (3, 11, 19, 27, 34)
TEST_INDICES = (0, 8, 16, 24, 32)
CAPSULE_RADIUS_M = 0.035
CLEARANCE_M = 0.010
SCOPE = (
    "Geometry-only finite Sobol endpoint feasibility survey. Independent FK and "
    "joint-limit verification; provisional joint-chain capsule clearance checks. "
    "No ball release outcomes or rewards, native settling, self-collision proof, "
    "collision-certified paths, exhaustive coverage or impossibility claims."
)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def split_for_index(index):
    if type(index) is not int or not 0 <= index < 35:
        raise ValueError("Expected original cell index in 0..34")
    return "validation" if index in VALIDATION_INDICES else "test" if index in TEST_INDICES else "train"


def decode_action(cell, values, identifier="continuous"):
    """Bucket-relative bearing offset; gap is ball-bottom Z minus board center Z."""
    values = np.asarray(values, dtype=float)
    if values.shape != (5,) or not np.isfinite(values).all() or np.any(values < LOW) or np.any(values > HIGH):
        raise ValueError("Expected five finite continuous actions inside the signed bounds")
    offset, radius, height, gap, tilt = map(float, values)
    bucket = np.asarray(cell["bucket_xy_m"], dtype=float)
    if bucket.shape != (2,) or not np.isfinite(bucket).all():
        raise ValueError("Invalid bucket XY")
    reference = np.array([0.67, -0.10]) - bucket
    reference_bearing = math.atan2(reference[1], reference[0])
    bearing = reference_bearing + offset
    return dict(
        id=identifier, bearing_offset_rad=offset, bearing_rad=bearing, radius_m=radius,
        board_x_m=float(bucket[0] + radius * math.cos(bearing)),
        board_y_m=float(bucket[1] + radius * math.sin(bearing)),
        height_m=height, gap_m=gap, tilt_rad=tilt, tilt_deg=math.degrees(tilt),
        azimuth_deg=math.degrees(bearing) % 360,
        azimuth_offset_deg=math.degrees(offset),
    )


def _values(cell, action):
    values = np.array([action[k] for k in ACTION_NAMES], dtype=float)
    expected = decode_action(cell, values, action.get("id", "continuous"))
    for key in ("board_x_m", "board_y_m", "azimuth_deg", "azimuth_offset_deg", "tilt_deg"):
        if not math.isfinite(float(action[key])) or abs(float(action[key]) - expected[key]) > 1e-8:
            raise ValueError("Continuous action geometry disagrees with its parameters")
    return values


def geometry_check(scene, cell, action):
    """Conservative board OBB/table and bucket bounding-cylinder checks only."""
    _values(cell, action)
    targets = endpoint_targets(scene, cell, action)
    board = targets["board_world"]
    half = np.asarray(scene["board"]["size_m"], dtype=float) / 2
    projected_half = np.abs(board[:3, :3]) @ half
    center = board[:3, 3]
    bottom = float(center[2] - projected_half[2])
    table = scene["table"]
    table_top = float(table["center_m"][2] + table["size_m"][2] / 2)
    rim = float(scene["bucket"]["assumed_interior_depth_m"] + scene["bucket"]["assumed_bottom_thickness_m"])
    bucket_radius = scene["bucket"]["outer_top_diameter_m"] / 2
    radius = action["radius_m"]
    horizontal_extent = float(np.linalg.norm(projected_half[:2]))
    release = targets["release_center_m"]
    normal = targets["board_normal_world"]
    sphere_plane_clearance = float((release - center) @ normal - scene["ball"]["radius_m"] - half[2])
    reasons = []
    if bottom < table_top + CLEARANCE_M:
        reasons.append("board_table_clearance")
    # Only disjoint enclosing volumes are claimed safe. A reject can be a false
    # positive because the outer cylinder and projected board sphere are coarse.
    if bottom < rim + CLEARANCE_M and radius < horizontal_extent + bucket_radius + CLEARANCE_M:
        reasons.append("board_bucket_proxy_overlap")
    if sphere_plane_clearance < CLEARANCE_M:
        reasons.append("release_board_clearance")
    if release[2] <= rim + scene["ball"]["radius_m"]:
        reasons.append("release_below_rim_energy_floor")
    return dict(
        accepted=not reasons, reasons=reasons,
        metrics=dict(board_bottom_z_m=bottom, table_top_z_m=table_top, bucket_rim_z_m=rim,
                     board_bucket_horizontal_distance_m=float(radius),
                     projected_board_half_extent_m=projected_half.tolist(),
                     release_ball_center_m=release.tolist(),
                     ball_board_plane_clearance_m=sphere_plane_clearance),
        scope="Conservative enclosing-volume checks; no full robot/link mesh collision certificate",
    )


def bounded_solve(target, links, initial, normal_only, max_nfev=40):
    """Two fixed starts, bounded work; same independent original residual gate."""
    lo, hi = np.array([link[2] for link in links]), np.array([link[3] for link in links])
    initial = np.clip(np.asarray(initial, dtype=float), lo + 1e-6, hi - 1e-6)
    starts = [initial, np.clip(initial + [0.25, -0.25, 0.2, 0, 0.15, 0], lo + 1e-6, hi - 1e-6)]
    best, log = None, []

    def residual(q):
        current = fk(q, links)
        angular = current[:3, 2] - target[:3, 2] if normal_only else Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()
        return np.r_[current[:3, 3] - target[:3, 3], 0.25 * angular]

    for start in starts:
        if normal_only:
            unpack = lambda v: np.r_[v, start[-1]]
            fit = least_squares(lambda v: residual(unpack(v)), start[:-1],
                                bounds=(lo[:-1], hi[:-1]),
                                jac=lambda v: normal_jacobian(unpack(v), links)[:, :-1],
                                max_nfev=max_nfev, ftol=1e-10, gtol=1e-10, xtol=1e-10)
            q = unpack(fit.x)
        else:
            fit = least_squares(residual, start, bounds=(lo, hi), max_nfev=max_nfev,
                                ftol=1e-10, gtol=1e-10, xtol=1e-10)
            q = fit.x
        error = residual(q)
        norm = float(np.linalg.norm(error))
        current = fk(q, links)
        angular = float(np.arccos(np.clip(current[:3, 2] @ target[:3, 2], -1, 1))) if normal_only else float(np.linalg.norm(error[3:]) / 0.25)
        record = dict(q=q.tolist(), solved=norm < 1e-4, residual=norm,
                      stage="bounded_two_start", position_error_mm=float(np.linalg.norm(error[:3]) * 1000),
                      orientation_error_deg=math.degrees(angular))
        log.append(dict(nfev=int(fit.nfev), residual=norm))
        if best is None or norm < best["residual"]:
            best = record
        if record["solved"]:
            break
    return dict(best, attempts=log)


def joint_points(q, links, base=None):
    """Joint-chain centers for coarse clearance; physical meshes may protrude."""
    transform = np.eye(4) if base is None else np.asarray(base, dtype=float).copy()
    points = [transform[:3, 3].copy()]
    for angle, (origin, axis, *_) in zip(q, links):
        transform = transform @ origin
        points.append(transform[:3, 3].copy())
        rotate = np.eye(4)
        rotate[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
        transform = transform @ rotate
    points.append(transform[:3, 3].copy())
    return np.asarray(points)


def segment_distance(a, b, c, d):
    """Exact finite line-segment distance, including degenerate segments."""
    a, b, c, d = map(lambda p: np.asarray(p, dtype=float), (a, b, c, d))
    u, v, w = b - a, d - c, a - c
    aa, bb, cc = float(u @ u), float(u @ v), float(v @ v)
    dd, ee = float(u @ w), float(v @ w)
    eps = 1e-15
    if aa <= eps and cc <= eps:
        return float(np.linalg.norm(a - c))
    if aa <= eps:
        t = np.clip(ee / cc, 0, 1)
        return float(np.linalg.norm(a - (c + t * v)))
    if cc <= eps:
        s = np.clip(-dd / aa, 0, 1)
        return float(np.linalg.norm((a + s * u) - c))
    denominator = aa * cc - bb * bb
    s = np.clip((bb * ee - cc * dd) / denominator, 0, 1) if denominator > eps else 0.0
    t = (bb * s + ee) / cc
    if t < 0:
        t, s = 0.0, np.clip(-dd / aa, 0, 1)
    elif t > 1:
        t, s = 1.0, np.clip((bb - dd) / aa, 0, 1)
    return float(np.linalg.norm(w + s * u - t * v))


def approximate_clearance(scene, links, fits):
    """Two-arm joint-chain capsule and below-table checks, not mesh collision."""
    bases = dict(Arm10=np.eye(4), Arm14=matrix([scene["arms"]["base_spacing_m"], 0, 0], [0, 0, math.pi]))
    points = {name: joint_points(fits[name]["q"], links[name], bases[name]) for name in bases}
    table = scene["table"]
    top = table["center_m"][2] + table["size_m"][2] / 2
    # The base and first two chain origins intentionally meet the table. The
    # distal centerline must stay above it, wherever the table footprint lies.
    table_lower = np.asarray(table["center_m"][:2]) - np.asarray(table["size_m"][:2]) / 2
    table_upper = np.asarray(table["center_m"][:2]) + np.asarray(table["size_m"][:2]) / 2
    reasons = []
    minimum_table = float("inf")
    for name, p in points.items():
        # Sample each distal segment; this is a coarse proxy, not an exact
        # swept capsule–box intersection test.
        for a, b in zip(p[2:-1], p[3:]):
            samples = a[None] + np.linspace(0, 1, 9)[:, None] * (b - a)[None]
            mask = np.all(samples[:, :2] >= table_lower, axis=1) & np.all(samples[:, :2] <= table_upper, axis=1)
            if mask.any():
                clearance = float(samples[mask, 2].min() - top - CAPSULE_RADIUS_M)
                minimum_table = min(minimum_table, clearance)
                if clearance < CLEARANCE_M:
                    reasons.append("arm_table_proxy_overlap_" + name)
    minimum_between = min(
        segment_distance(a, b, c, d)
        for a, b in zip(points["Arm10"][:-1], points["Arm10"][1:])
        for c, d in zip(points["Arm14"][:-1], points["Arm14"][1:])
    ) - 2 * CAPSULE_RADIUS_M
    if minimum_between < CLEARANCE_M:
        reasons.append("inter_arm_proxy_overlap")
    return dict(accepted=not reasons, reasons=sorted(set(reasons)),
                min_arm_table_clearance_m=minimum_table if math.isfinite(minimum_table) else None,
                min_inter_arm_clearance_m=float(minimum_between),
                capsule_radius_m=CAPSULE_RADIUS_M, margin_m=CLEARANCE_M,
                scope="Provisional joint-chain capsules; no self-collision/link mesh or path certificate")



def _board_proxy_at_joints(scene, cell, links, q):
    base = matrix([scene["arms"]["base_spacing_m"], 0, 0], [0, 0, math.pi])
    mount = matrix([0, 0, scene["board"]["tool_offset_m"]], [0, 0, scene["board"]["mount_yaw_rad"]])
    board = base @ fk(q, links) @ mount
    half = np.abs(board[:3, :3]) @ (np.asarray(scene["board"]["size_m"]) / 2)
    bottom = float(board[2, 3] - half[2])
    table = scene["table"]
    top = table["center_m"][2] + table["size_m"][2] / 2
    reasons = []
    if bottom < top + CLEARANCE_M:
        reasons.append("board_table_proxy_overlap")
    if cell is not None:
        rim = scene["bucket"]["assumed_interior_depth_m"] + scene["bucket"]["assumed_bottom_thickness_m"]
        radius = float(np.linalg.norm(board[:2, 3] - cell["bucket_xy_m"]))
        if bottom < rim + CLEARANCE_M and radius < np.linalg.norm(half[:2]) + scene["bucket"]["outer_top_diameter_m"] / 2 + CLEARANCE_M:
            reasons.append("board_bucket_proxy_overlap")
    return dict(reasons=reasons, board_bottom_z_m=bottom)


def approximate_path_check(scene, links, starts, goals, samples=16, cell=None):
    """Sample joint interpolation using provisional capsules and board bounds.

    These are finite geometric proxy checks. They do not cover between-sample
    excursions, actual drive tracking, all mesh protrusions or self collisions.
    No simulator/robot movement occurs here.
    """
    if type(samples) is not int or not 2 <= samples <= 128:
        raise ValueError("Path proxy needs 2..128 samples")
    starts = {name: np.asarray(starts[name], dtype=float) for name in ("Arm10", "Arm14")}
    goals = {name: np.asarray(goals[name], dtype=float) for name in starts}
    if any(q.shape != (6,) or not np.isfinite(q).all() for q in [*starts.values(), *goals.values()]):
        raise ValueError("Expected finite six-joint starts/goals")
    checks, reasons = [], []
    for fraction in np.linspace(0, 1, samples):
        qs = {name: start + fraction * (goals[name] - start) for name, start in starts.items()}
        clear = approximate_clearance(scene, links, {name: dict(q=q) for name, q in qs.items()})
        board = _board_proxy_at_joints(scene, cell, links["Arm14"], qs["Arm14"])
        row_reasons = clear["reasons"] + board["reasons"]
        checks.append(dict(fraction=float(fraction), reasons=row_reasons,
                           min_inter_arm_clearance_m=clear["min_inter_arm_clearance_m"],
                           min_arm_table_clearance_m=clear["min_arm_table_clearance_m"],
                           board_bottom_z_m=board["board_bottom_z_m"]))
        reasons.extend(row_reasons)
    return dict(accepted=not reasons, reasons=sorted(set(reasons)), checks=checks,
                bucket_proxy_checked=cell is not None,
                scope="Finite joint-interpolation capsule/board proxy; not certified collision-free motion, self collision or native tracking")

def qualify_action(scene, cell, action, links, seeds, *, solver=bounded_solve):
    """Return every requested action unchanged, with explicit geometry/IK status."""
    started = time.monotonic()
    geometry = geometry_check(scene, cell, action)
    record = dict(action=copy.deepcopy(action), accepted=False, reason=None,
                  geometry=geometry, ik={}, approximate_clearance=None)
    if not geometry["accepted"]:
        record["reason"] = geometry["reasons"][0]
    else:
        targets = endpoint_targets(scene, cell, action)
        bounds = outer_bounds(targets, links)
        record["outer_bounds"] = bounds
        if not bounds["eligible"]:
            record["reason"] = "outer_link_length_bound"
        else:
            for name in ("Arm10", "Arm14"):
                fit = solver(targets[name], links[name], seeds[name], name == "Arm14")
                record["ik"][name] = fit
                if not fit["solved"]:
                    record["reason"] = "endpoint_ik_unresolved_" + name
                    break
                verified_fit(targets[name], fit, links[name], name == "Arm14")
            if record["reason"] is None:
                clearance = approximate_clearance(scene, links, record["ik"])
                record["approximate_clearance"] = clearance
                record["reason"] = clearance["reasons"][0] if not clearance["accepted"] else None
                record["accepted"] = clearance["accepted"]
    record["elapsed_s"] = time.monotonic() - started
    return record


def make_spec(source_spec, samples_per_cell=128):
    if type(samples_per_cell) is not int or not 1 <= samples_per_cell <= 4096 or samples_per_cell & (samples_per_cell - 1):
        raise ValueError("Sobol samples per cell must be a power of two in 1..4096")
    cells = copy.deepcopy(source_spec["cells"])
    if len(cells) != 35 or len({c["id"] for c in cells}) != 35 or len({tuple(c["bucket_xy_m"]) for c in cells}) != 35:
        raise ValueError("Expected the unchanged 35 distinct source cells")
    scene = copy.deepcopy(source_spec["scene"])
    value = dict(version=VERSION, source_fingerprint=source_spec["fingerprint"],
                 scene=scene, scene_sha256=canonical_hash(scene), cells=cells,
                 seed=SEED, samples_per_cell=samples_per_cell, total_requests=35*samples_per_cell,
                 action_names=list(ACTION_NAMES), action_low=LOW.tolist(), action_high=HIGH.tolist(),
                 sampling="Independent scrambled Sobol five-dimensional sequence per original cell, all requests retained",
                 split_assignment={c["id"]: split_for_index(i) for i, c in enumerate(cells)},
                 split_scope="25 train,5 validation,5 untouched test bucket locations; feasibility geometry is not a reward label",
                 scope=SCOPE)
    value["fingerprint"] = canonical_hash(value)
    return value


def sample_values(spec, index):
    if not 0 <= index < len(spec["cells"]):
        raise ValueError("Unknown cell index")
    engine = qmc.Sobol(d=5, scramble=True, seed=spec["seed"] + index)
    return LOW + engine.random_base2(int(math.log2(spec["samples_per_cell"]))) * (HIGH - LOW)


def _cell_map(job):
    spec, asset_path, i = job
    assets = Path(asset_path)
    links = {name: chain(ET.parse(assets / file).getroot()) for name, file in (("Arm10", "dropper.urdf"), ("Arm14", "board_arm.urdf"))}
    poses = json.loads((assets / "poses.json").read_text())
    seeds = dict(Arm10=np.asarray(poses["dropper_joint_rad"]), Arm14=np.asarray(poses["board_joint_rad"]))
    cell = spec["cells"][i]
    records = []
    for j, values in enumerate(sample_values(spec, i)):
        action = decode_action(cell, values, f"sobol-{i:02d}-{j:04d}")
        record = qualify_action(spec["scene"], cell, action, links, seeds)
        record.update(cell_id=cell["id"], cell_index=i, proposal=j, split=spec["split_assignment"][cell["id"]])
        records.append(record)
    return records


def build_map(spec, assets, progress=None, workers=1):
    unsigned = {k: v for k, v in spec.items() if k != "fingerprint"}
    if spec.get("version") != VERSION or canonical_hash(unsigned) != spec.get("fingerprint"):
        raise ValueError("Feasibility contract changed")
    assets = Path(assets)
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("Choose 1..16 bounded CPU workers")
    records, started = [], time.monotonic()
    jobs = [(spec, str(assets), i) for i in range(len(spec["cells"]))]
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        groups = pool.map(_cell_map, jobs) if pool else map(_cell_map, jobs)
        for i, group in enumerate(groups):
            records.extend(group)
            if progress:
                progress(dict(completed_cells=i+1, attempts=len(records), qualified=sum(r["accepted"] for r in records), elapsed_s=time.monotonic()-started))
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    summary = []
    for i, cell in enumerate(spec["cells"]):
        group = [r for r in records if r["cell_index"] == i]
        accepted = [r for r in group if r["accepted"]]
        summary.append(dict(cell_id=cell["id"], cell_index=i, bucket_xy_m=cell["bucket_xy_m"],
                            split=spec["split_assignment"][cell["id"]], requests=len(group), qualified=len(accepted),
                            reasons=dict(Counter(r["reason"] or "endpoint_proxy_qualified" for r in group)),
                            qualified_ranges={k: [min(r["action"][k] for r in accepted), max(r["action"][k] for r in accepted)] for k in ACTION_NAMES} if accepted else {}))
    result = dict(schema="sircus-continuous-feasibility-result-v1", spec_fingerprint=spec["fingerprint"],
                  source_fingerprint=spec["source_fingerprint"], records=records, cells=summary,
                  requests=len(records), qualified=sum(r["accepted"] for r in records),
                  elapsed_s=time.monotonic()-started, scope=SCOPE,
                  asset_sha256={file:hashlib.sha256((assets/file).read_bytes()).hexdigest() for file in ("dropper.urdf","board_arm.urdf","poses.json")})
    result["fingerprint"] = canonical_hash(result)
    return result



def geometry_coverage(feasibility, source_cells, batches=4):
    """Farthest-first geometry-only filming schedule, never PPO/evaluation data.

    No rewards/outcomes are accepted. Native qualification and driven path gates
    must run again; finite map acceptance alone cannot enable a release.
    """
    if type(batches) is not int or not 1 <= batches <= 32:
        raise ValueError("Choose 1..32 geometry coverage batches")
    if len(source_cells) != 35:
        raise ValueError("Coverage requires the unchanged 35 source cells")
    expected_ids = {cell["id"] for cell in source_cells}
    if set(c["cell_id"] for c in feasibility["cells"]) != expected_ids:
        raise ValueError("Geometry coverage source cells disagree")
    by_cell = {identifier: [] for identifier in expected_ids}
    for row in feasibility["records"]:
        if row["cell_id"] not in by_cell:
            raise ValueError("Unexpected geometry coverage cell")
        if row["accepted"]:
            by_cell[row["cell_id"]].append(row)
    selected = {}
    for cell in source_cells:
        options = by_cell[cell["id"]]
        if not options:
            selected[cell["id"]] = []
            continue
        values = np.array([_values(cell, row["action"]) for row in options])
        scaled = (values - LOW) / (HIGH - LOW)
        # First pick is the point farthest from its menu centroid. Subsequent
        # choices maximize minimum distance from already selected geometries.
        first = int(np.argmax(np.linalg.norm(scaled-scaled.mean(axis=0), axis=1)))
        indices = [first]
        while len(indices) < min(batches, len(options)):
            distances = np.min(np.linalg.norm(scaled[:, None, :] - scaled[indices][None, :, :], axis=2), axis=1)
            distances[indices] = -1
            indices.append(int(np.argmax(distances)))
        selected[cell["id"]] = [options[i] for i in indices]
    result = []
    for batch in range(batches):
        actions, enabled, proposal_ids, fits = [], [], [], []
        for cell in source_cells:
            options = selected[cell["id"]]
            row = options[batch % len(options)] if options else None
            fallback = decode_action(cell, [0, .4, .35, .24, math.radians(29)], "disabled-coverage")
            actions.append(copy.deepcopy(row["action"]) if row else fallback)
            enabled.append(row is not None)
            proposal_ids.append(row["action"]["id"] if row else None)
            fits.append(copy.deepcopy(row.get("ik")) if row else None)
        result.append(dict(batch=batch, mode="geometry_coverage", actions=actions,
                           enabled=enabled, proposal_ids=proposal_ids, fits=fits,
                           scope="Geometry-only farthest-first coverage, no policy updates or evaluation; fresh native admission still required"))
    return result

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-spec", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory; previous studies are preserved")
    args.output.mkdir(parents=True)
    spec = make_spec(json.loads(args.source_spec.read_text()), args.samples)
    (args.output / "contract.json").write_text(json.dumps(spec, allow_nan=False))
    def progress(value):
        print(json.dumps(value), flush=True)
        (args.output / "progress.json").write_text(json.dumps(value, allow_nan=False))
    result = build_map(spec, args.assets, progress, workers=args.workers)
    (args.output / "feasibility.json").write_text(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
