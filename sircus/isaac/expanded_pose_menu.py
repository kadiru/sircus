"""Bounded geometry-only proposal/IK search for the unchanged 35 bucket sites.

CPU IK is an endpoint filter, never a success label or a native settling/path
certificate. Native qualification must precede training. No GPU is initialized.
"""

import copy
import hashlib
import itertools
import json
import math
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
from prepare_assets import chain, matrix
from scipy.spatial.transform import Rotation

VERSION = "sircus-expanded-pose-menu-v1"
REFERENCE = np.array([0.67, -0.10])
HEIGHTS = (0.30, 0.22, 0.38, 0.14, 0.46, 0.54)
GAPS = (0.08, 0.14, 0.22, 0.32, 0.44, 0.56)
TILTS = (12.0, 20.0, 28.0, 36.0, 44.0, 52.0)
RADII = (0.32, 0.40, 0.52, 0.68, 0.84, 1.0, 1.2)
BEARINGS = (0, -15, 15, -30, 30, -60, 60, -90, 90, -135, 135, 180)
SEED = 20261109


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def cell_identity(cells):
    return [dict(id=c["id"], bucket_xy_m=list(c["bucket_xy_m"])) for c in cells]


def geometry_signature(action):
    return tuple(
        round(float(action[k]), 9)
        for k in ("board_x_m", "board_y_m", "height_m", "gap_m", "tilt_deg")
    )


def endpoint_targets(scene, cell, action):
    """Exact existing Isaac targets; role keys map directly to env._ik cache."""
    center = np.array([action["board_x_m"], action["board_y_m"], action["height_m"]])
    from feasibility_contract import dynamic_cell

    heading = dynamic_cell(cell, action)["heading_deg"]
    board = np.eye(4)
    board[:3, :3] = (
        Rotation.from_euler("z", heading, degrees=True).as_matrix()
        @ Rotation.from_euler("x", action["tilt_deg"], degrees=True).as_matrix()
        @ Rotation.from_euler("z", scene["board"]["mount_yaw_rad"]).as_matrix()
    )
    board[:3, 3] = center
    mount = matrix(
        [0, 0, scene["board"]["tool_offset_m"]], [0, 0, scene["board"]["mount_yaw_rad"]]
    )
    base = matrix([scene["arms"]["base_spacing_m"], 0, 0], [0, 0, math.pi])
    release = center + [0, 0, scene["ball"]["radius_m"] + action["gap_m"]]
    return dict(
        Arm10=matrix(release + [0, 0, 0.095], [math.pi, 0, 0]),
        Arm14=np.linalg.inv(base) @ board @ np.linalg.inv(mount),
        board_world=board,
        release_center_m=release,
        board_normal_world=board[:3, 2],
    )


def outer_bounds(targets, links):
    """Conservative URDF origin-length bound, not a numerical IK conclusion."""
    result = {}
    for name in ("Arm10", "Arm14"):
        robot = links[name] if isinstance(links, dict) else links
        pivot = robot[0][0][:3, 3]
        radius = sum(float(np.linalg.norm(link[0][:3, 3])) for link in robot[1:])
        distance = float(np.linalg.norm(targets[name][:3, 3] - pivot))
        result[name] = dict(
            distance_m=distance, outer_radius_m=radius, within=distance <= radius + 1e-6
        )
    return dict(eligible=all(v["within"] for v in result.values()), arms=result)


def geometry_reason(scene, cell, action):
    """Cheap new conservative geometry gates; unchanged measured dimensions."""
    vector = np.array(
        [action[k] for k in ("board_x_m", "board_y_m", "height_m", "gap_m", "tilt_deg")]
    )
    if not np.isfinite(vector).all():
        return "nonfinite_geometry"
    x, y, h, gap, tilt = vector
    if not (
        0.38 <= x <= 0.94
        and -0.50 <= y <= 0.30
        and 0.14 <= h <= 0.54
        and 0.08 <= gap <= 0.56
        and 12 <= tilt <= 52
    ):
        return "proposal_bounds"
    radius = float(np.linalg.norm(np.array(cell["bucket_xy_m"]) - [x, y]))
    separation = (
        np.linalg.norm(np.array(scene["board"]["size_m"][:2]) / 2)
        + scene["bucket"]["outer_top_diameter_m"] / 2
        + 0.02
    )
    if radius < max(0.28, separation) or radius > 1.2 + 1e-9:
        return "board_bucket_separation"
    target = endpoint_targets(scene, cell, action)["board_world"]
    bottom = h - float(np.abs(target[2, :3]) @ (np.array(scene["board"]["size_m"]) / 2))
    if bottom < 0.015:
        return "board_table_clearance"
    return None


def _action(cell, x, y, h, gap, tilt, identifier):
    dx, dy = np.array([x, y]) - cell["bucket_xy_m"]
    bearing = math.atan2(dy, dx)
    base = math.atan2(*(REFERENCE - cell["bucket_xy_m"])[::-1])
    return dict(
        id=identifier,
        board_x_m=float(x),
        board_y_m=float(y),
        height_m=float(h),
        gap_m=float(gap),
        tilt_deg=float(tilt),
        radius_m=math.hypot(dx, dy),
        azimuth_deg=math.degrees(bearing) % 360,
        azimuth_offset_deg=(math.degrees(bearing - base) + 180) % 360 - 180,
    )


def proposals(scene, cell, limit=512):
    """Stable common-overlap grid plus bucket-ring seeds, without outcome data.

    Each XY offers six height strata. Alternate ballistic-ranked and seeded
    broad gap/tilt choices so a bad analytic approximation cannot exclude them.
    """
    if not 1 <= limit <= 4096:
        raise ValueError("Proposal limit must be 1..4096")
    from feasibility_contract import proposed_range

    grid = [
        (float(x), float(y))
        for x, y in itertools.product(
            np.linspace(0.38, 0.94, 8), np.linspace(-0.5, 0.3, 9)
        )
    ]
    grid.sort(key=lambda xy: float(np.linalg.norm(np.array(xy) - REFERENCE)))
    bucket = np.array(cell["bucket_xy_m"])
    towards = math.atan2(*(REFERENCE - bucket)[::-1])
    rings = [
        tuple(
            bucket
            + r
            * np.array(
                [
                    math.cos(towards + math.radians(a)),
                    math.sin(towards + math.radians(a)),
                ]
            )
        )
        for r, a in itertools.product(RADII, BEARINGS)
    ]
    points, seen = [], set()
    for x, y in grid + rings:
        key = (round(x, 7), round(y, 7))
        if 0.38 <= x <= 0.94 and -0.5 <= y <= 0.3 and key not in seen:
            seen.add(key)
            points.append((x, y))
    rng = np.random.default_rng(SEED + int(cell.get("index", 0)))
    portfolios = []
    profiles = list(itertools.product(GAPS, TILTS))
    for x, y in points:
        distance = float(np.linalg.norm(bucket - [x, y]))
        if distance > 1.2 or distance < 0.28:
            continue
        options = []
        for h in HEIGHTS:
            ranked = sorted(
                profiles, key=lambda gt: abs(proposed_range(scene, h, *gt) - distance)
            )
            broad = [profiles[j] for j in rng.permutation(len(profiles))]
            options.extend(
                [(h, *ranked[0]), (h, *broad[0]), (h, *ranked[1]), (h, *broad[1])]
            )
        portfolios.append((x, y, options))
    result, values_seen = [], set()
    for depth in range(24):
        for point_index, (x, y, options) in enumerate(portfolios):
            # Rotate the height stratum across XY in the first pass rather than
            # spending the entire finite budget at one height.
            k = (depth + 4 * (point_index % len(HEIGHTS))) % len(options)
            h, gap, tilt = options[k]
            key = tuple(round(v, 8) for v in (x, y, h, gap, tilt))
            if key in values_seen:
                continue
            values_seen.add(key)
            action = _action(cell, x, y, h, gap, tilt, f"expanded-{len(result):04d}")
            if geometry_reason(scene, cell, action) is None:
                result.append(action)
                if len(result) >= limit:
                    return result
    return result


def make_manifest(
    spec,
    menus,
    *,
    fits=None,
    checks=None,
    native_checks=None,
    qualification="offline_endpoint_ik",
    **metadata,
):
    """Seal geometry/qualification evidence; native checks must be real readbacks."""
    count = len(spec["cells"])
    result = dict(
        metadata,
        version=VERSION,
        qualification=qualification,
        spec=copy.deepcopy(spec),
        source_fingerprint=spec["fingerprint"],
        scene_sha256=canonical_hash(spec["scene"]),
        cells_sha256=canonical_hash(cell_identity(spec["cells"])),
        menus=copy.deepcopy(menus),
        fits=copy.deepcopy(fits if fits is not None else [[] for _ in range(count)]),
        checks=copy.deepcopy(
            checks if checks is not None else [[] for _ in range(count)]
        ),
        native_checks=copy.deepcopy(
            native_checks if native_checks is not None else [[] for _ in range(count)]
        ),
    )
    result["fingerprint"] = canonical_hash(result)
    return result


def load_native_menu(path, spec):
    """Reject offline-only, incomplete, changed-scene or unsteady native menus."""
    from expanded_farm import validate_action
    from settling_contract import accepted

    manifest = json.loads(Path(path).read_text())
    unsigned_spec = copy.deepcopy(spec)
    if (
        canonical_hash({k: v for k, v in unsigned_spec.items() if k != "fingerprint"})
        != spec["fingerprint"]
    ):
        raise ValueError("Expanded experiment fingerprint changed")
    payload = dict(manifest)
    fingerprint = payload.pop("fingerprint", None)
    if canonical_hash(payload) != fingerprint or manifest.get("version") != VERSION:
        raise ValueError("Expanded menu checksum/version mismatch")
    if manifest.get("qualification") != "native_settled_endpoints":
        raise ValueError("Expanded menu has not passed native settling qualification")
    if (
        manifest.get("source_fingerprint") != spec["fingerprint"]
        or manifest.get("spec") != spec
    ):
        raise ValueError("Expanded menu contract changed")
    if manifest.get("scene_sha256") != canonical_hash(spec["scene"]) or manifest.get(
        "cells_sha256"
    ) != canonical_hash(cell_identity(spec["cells"])):
        raise ValueError("Expanded scene or fixed bucket assignments changed")
    menus, checks = manifest["menus"], manifest["native_checks"]
    if len(menus) != 35 or len(checks) != 35 or len(spec["cells"]) != 35:
        raise ValueError("Expected all 35 unchanged workcells")
    try:
        for cell, options, records in zip(spec["cells"], menus, checks):
            if len(options) < 2 or len({a["id"] for a in options}) != len(options):
                raise ValueError(
                    "Every workcell needs two distinct native-qualified actions"
                )
            if len({geometry_signature(a) for a in options}) != len(options):
                raise ValueError("Native menu repeats the same physical configuration")
            by_id = {r["action_id"]: r for r in records if r.get("accepted")}
            for action in options:
                validate_action(cell, action)
                if geometry_reason(spec["scene"], cell, action):
                    raise ValueError(
                        "Expanded action violates conservative geometry limits"
                    )
                record = by_id[action["id"]]
                if record.get("action") != action:
                    raise ValueError(
                        "Native settling evidence belongs to different action geometry"
                    )
                measured = record["settled"]
                numeric = [
                    measured[k]
                    for k in (
                        "position_error_mm",
                        "normal_error_deg",
                        "joint_speed_rad_s",
                        "position_drift_mm",
                        "normal_drift_deg",
                        "dropper_position_error_mm",
                    )
                ]
                if (
                    record.get("invalid") is not None
                    or not np.isfinite(numeric).all()
                    or min(numeric) < 0
                    or not accepted(measured)
                    or measured["dropper_position_error_mm"] > 2
                ):
                    raise ValueError(
                        "Native endpoint did not pass original measured release guards"
                    )
    except (KeyError, TypeError) as error:
        raise ValueError("Incomplete native endpoint evidence") from error
    return menus, dict(
        version=VERSION,
        qualification=manifest["qualification"],
        fingerprint=fingerprint,
        file_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        accepted_per_cell=[len(m) for m in menus],
        source_fingerprint=spec["fingerprint"],
    )


def qualify(
    scene,
    cells,
    assets,
    existing_menus=None,
    *,
    spec,
    max_checks_per_cell=24,
    accepted_per_cell=4,
    progress=None,
    previous=None,
    cell_indices=None,
    candidates_per_cell=512,
    **metadata,
):
    """Finite CPU endpoint search with warm starts, shared target cache and bounds.

    Reuse ``previous`` to extend only unresolved cells without rerunning tested
    candidates. Budgets count candidates sent to IK; cheap geometric rejects are
    reported separately. Solved endpoints still require native hold validation.
    """
    from robust_ik import solve

    if not 1 <= max_checks_per_cell <= 512 or not 2 <= accepted_per_cell <= 32:
        raise ValueError(
            "Bounded search needs 1..512 checks and 2..32 accepted actions"
        )
    if cell_identity(cells) != cell_identity(spec["cells"]) or scene != spec["scene"]:
        raise ValueError("Search geometry does not match the expanded contract")
    assets = Path(assets)
    links = {
        name: chain(ET.parse(assets / filename).getroot())
        for name, filename in (("Arm10", "dropper.urdf"), ("Arm14", "board_arm.urdf"))
    }
    poses = json.loads((assets / "poses.json").read_text())
    seeds = {
        "Arm10": np.array(poses["dropper_joint_rad"]),
        "Arm14": np.array(poses["board_joint_rad"]),
    }
    count = len(cells)
    menus, fits, checks = ([[] for _ in cells] for _ in range(3))
    if previous:
        payload = dict(previous)
        fingerprint = payload.pop("fingerprint", None)
        if (
            canonical_hash(payload) != fingerprint
            or previous["source_fingerprint"] != spec["fingerprint"]
        ):
            raise ValueError("Previous offline search changed")
        menus, fits, checks = (
            copy.deepcopy(previous[k]) for k in ("menus", "fits", "checks")
        )
    cache, started = {}, time.monotonic()
    for i, (options, solved) in enumerate(zip(menus, fits)):
        for action, fit in zip(options, solved):
            targets = endpoint_targets(scene, cells[i], action)
            for name in ("Arm10", "Arm14"):
                cache[(name, tuple(np.round(targets[name].ravel(), 10)))] = fit[name]
    active = list(range(count)) if cell_indices is None else list(cell_indices)
    for i in active:
        if i not in range(count):
            raise ValueError("Unknown workcell index")
        candidates = []
        if existing_menus:
            for a in existing_menus[i]:
                candidates.append(
                    _action(
                        cells[i],
                        a["board_x_m"],
                        a["board_y_m"],
                        a["height_m"],
                        a["gap_m"],
                        a["tilt_deg"],
                        "legacy-" + a["id"],
                    )
                )
        candidates += proposals(scene, cells[i], candidates_per_cell)
        seen = {r["action_id"] for r in checks[i]}
        used = sum(r.get("ik_checked", False) for r in checks[i])
        warm = {
            name: (np.array(fits[i][-1][name]["q"]) if fits[i] else seeds[name])
            for name in seeds
        }
        for action in candidates:
            if len(menus[i]) >= accepted_per_cell or used >= max_checks_per_cell:
                break
            if action["id"] in seen:
                continue
            if geometry_signature(action) in {geometry_signature(a) for a in menus[i]}:
                checks[i].append(
                    dict(
                        action_id=action["id"],
                        reason="duplicate_physical_geometry",
                        ik_checked=False,
                    )
                )
                continue
            reason = geometry_reason(scene, cells[i], action)
            targets = endpoint_targets(scene, cells[i], action)
            bound = outer_bounds(targets, links)
            if reason or not bound["eligible"]:
                checks[i].append(
                    dict(
                        action_id=action["id"],
                        reason=reason or "outer_link_length_bound",
                        ik_checked=False,
                        bounds=bound,
                    )
                )
                continue
            used += 1
            result = {}
            for name in ("Arm10", "Arm14"):
                key = (name, tuple(np.round(targets[name].ravel(), 10)))
                if key not in cache:
                    cache[key] = solve(
                        targets[name], links[name], warm[name], name == "Arm14"
                    )
                result[name] = cache[key]
                if not result[name]["solved"]:
                    reason = "endpoint_ik_" + name
                    break
            checks[i].append(
                dict(action_id=action["id"], reason=reason, ik_checked=True, ik=result)
            )
            if reason is None:
                menus[i].append(action)
                fits[i].append(result)
                warm = {name: np.array(result[name]["q"]) for name in result}
            if progress:
                progress(
                    dict(
                        cell=i,
                        cell_id=cells[i]["id"],
                        checks=used,
                        accepted=len(menus[i]),
                        elapsed_s=time.monotonic() - started,
                    )
                )
    return make_manifest(
        spec,
        menus,
        fits=fits,
        checks=checks,
        **metadata,
        proposal_protocol=dict(
            version=VERSION,
            heights_m=HEIGHTS,
            gaps_m=GAPS,
            tilts_deg=TILTS,
            radii_m=RADII,
            max_checks_per_cell=max_checks_per_cell,
            accepted_per_cell=accepted_per_cell,
            candidates_per_cell=candidates_per_cell,
            geometry_only=True,
        ),
        summary=dict(
            accepted_per_cell=[len(m) for m in menus],
            cells_with_endpoint=sum(bool(m) for m in menus),
            reasons=dict(
                Counter(
                    r["reason"] or "endpoint_ik_qualified"
                    for group in checks
                    for r in group
                )
            ),
            elapsed_s=time.monotonic() - started,
            cached_endpoint_count=len(cache),
        ),
        scope="Offline CPU IK only; no success labels, native settling, collision-free paths or impossibility claims.",
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-source", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--cells", type=int, nargs="+")
    parser.add_argument("--max-checks", type=int, default=24)
    parser.add_argument("--accepted", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file; prior search evidence is preserved")
    spec = json.loads(args.spec.read_text())
    previous = json.loads(args.previous.read_text()) if args.previous else None
    existing = None
    provenance = {}
    if args.legacy_source:
        from action_training import load_geometry_menu

        # Use the pinned source identity/cells. Regenerating the old floating-
        # point layout can change its hash between NumPy/platform versions.
        existing, provenance = load_geometry_menu(
            args.legacy_source,
            dict(fingerprint=spec["camera_source_fingerprint"], cells=spec["cells"]),
        )

    def progress(row):
        print(json.dumps(row), flush=True)

    result = qualify(
        spec["scene"],
        spec["cells"],
        args.assets,
        existing,
        spec=spec,
        max_checks_per_cell=args.max_checks,
        accepted_per_cell=args.accepted,
        candidates_per_cell=args.candidates,
        previous=previous,
        cell_indices=args.cells,
        progress=progress,
        legacy_menu_provenance=provenance,
        source_original_geometry_fp=spec.get("camera_source_fingerprint"),
    )
    args.output.write_text(json.dumps(result, allow_nan=False))
    print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
