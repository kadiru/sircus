"""Freeze a geometry-only holdout between the 33 usable fixed bucket positions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .corrected_contract import grid_layouts
from .learning_contract import digest
from .table_training_contract import _coordinates, collect_exclusions

VERSION = "sircus-intergrid-layouts-v1"
SEED = 2026111501
MIN_SEPARATION_M = 0.005
PERIMETER_STRIP_M = 0.060
NEAR_BASE_MAX_CLEARANCE_M = 0.035
EXCLUDED_CELLS = {"dev-grid-03-00", "dev-grid-03-06"}


def _historical_coordinates(history_root):
    """Include packet-only requests and legacy bare coordinate arrays."""
    history_root = Path(history_root).resolve()
    repository = Path(__file__).resolve().parents[2]
    if not history_root.is_dir():
        raise ValueError("An archived study directory is required for novelty checks")
    inventory = collect_exclusions(history_root, repository)
    points = set(map(tuple, inventory["coordinates"]))
    for label, root in (("output", history_root), ("baselines", repository / "baselines")):
        for pattern in ("**/splits.json", "**/contract.json", "**/requests/*.json", "**/preparations/*.json"):
            for path in sorted(root.glob(pattern)):
                raw = path.read_bytes()
                payload = json.loads(raw)
                relative = f"{label}/{path.relative_to(root).as_posix()}"
                checksum = hashlib.sha256(raw).hexdigest()
                if inventory["files_sha256"].setdefault(relative, checksum) != checksum:
                    raise ValueError(f"Historical coordinate metadata changed while reading {relative}")
                _coordinates(payload, points)
                if not isinstance(payload, dict):
                    continue
                for splits in (payload, payload.get("splits"), payload.get("future_splits")):
                    if not isinstance(splits, dict):
                        continue
                    for role in ("train", "validation", "test"):
                        _coordinates(splits.get(role), points, coordinate_list=True)
    return sorted(points), inventory["files_sha256"]


def _placement_geometry(scene, assets):
    """Bound every compiled fixed arm collider by its XY distance from the arm axis."""
    import mujoco as mj

    from .fixed35_six_geometry import compiled_shape, fixed_ancestry
    from .model import build_scene

    axes = {"Arm10": [0.0, 0.0], "Arm14": [float(scene["arms"]["base_spacing_m"]), 0.0]}
    xml = build_scene(scene, Path(assets), solref=(.004, .15), contact_friction=.35)
    model = mj.MjModel.from_xml_string(xml)
    state = mj.MjData(model)
    mj.mj_forward(model, state)
    bounds = {}
    for index in range(model.ngeom):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, index) or ""
        arm = name.split("/", 1)[0]
        if arm not in axes or not fixed_ancestry(model, int(model.geom_bodyid[index]))[0]:
            continue
        shape = compiled_shape(model, state, index)
        if shape["kind"] != "convex_polyhedron":
            raise ValueError(f"Expected the fixed arm collision mesh for {name}")
        radius = float(np.linalg.norm(shape["points"][:, :2] - axes[arm], axis=1).max())
        bound = bounds.setdefault(arm, dict(center_xy_m=axes[arm], radius_m=0.0, colliders=[]))
        bound["radius_m"] = max(bound["radius_m"], radius)
        bound["colliders"].append(name)
    if set(bounds) != set(axes):
        raise ValueError("Could not verify fixed collider bounds for both arms")

    bucket = scene["bucket"]
    radius = max(float(bucket[name]) for name in (
        "outer_top_diameter_m", "outer_bottom_diameter_m",
        "inner_top_diameter_m", "inner_bottom_diameter_m",
    )) / 2
    table = scene["table"]
    center = np.asarray(table["center_m"][:2], float)
    half_size = np.asarray(table["size_m"][:2], float) / 2
    return dict(
        fixed_arm_bounds=bounds, bucket_radius_m=radius,
        tabletop_xy_bounds_m=[(center - half_size).tolist(), (center + half_size).tolist()],
        scene_sha256=digest(scene), mujoco_version=mj.__version__,
        minimum_base_clearance_m=MIN_SEPARATION_M,
        method="Horizontal distance between conservative bounding circles of compiled fixed arm meshes and bucket.",
    )


def make_layouts(history_root, grid=None, *, scene, assets, seed=SEED):
    """Return 50 fixed TEST layouts and the exact inventory and placement inputs.

    Two samples cover each of the 24 inter-grid rectangles. Two extra samples
    challenge fixed-base clearance. Placement uses only coordinates and geometry.
    """
    canonical = {row["id"]: tuple(row["bucket_xy_m"]) for row in grid_layouts()}
    usable = {key: xy for key, xy in canonical.items() if key not in EXCLUDED_CELLS}
    if grid is not None:
        rows = list(grid)
        supplied = {
            row["id"].removeprefix("fixed35-"): tuple(row["bucket_xy_m"]) for row in rows
        }
        if len(supplied) != len(rows) or supplied not in (canonical, usable):
            raise ValueError("Expected the canonical fixed grid or its 33 usable locations")
    xs, ys = (np.unique(np.asarray(list(usable.values()))[:, i]) for i in range(2))
    lower, upper = np.array([xs[0], ys[0]]), np.array([xs[-1], ys[-1]])
    historical, files = _historical_coordinates(history_root)
    excluded = sorted(set(historical) | set(canonical.values()))
    used = np.asarray(excluded)
    geometry = _placement_geometry(scene, assets)
    fixed = list(geometry["fixed_arm_bounds"].values())
    bases = np.asarray([entry["center_xy_m"] for entry in fixed])
    radii = np.asarray([entry["radius_m"] for entry in fixed]) + geometry["bucket_radius_m"]
    tabletop_low, tabletop_high = np.asarray(geometry["tabletop_xy_bounds_m"])
    rng, layouts = np.random.default_rng(seed), []

    def sample(box_low, box_high, stratum, *, base_index=None):
        nonlocal used
        for _ in range(10000):
            xy = rng.uniform(box_low, box_high)
            clearance = np.linalg.norm(bases - xy, axis=1) - radii
            if clearance.min() < MIN_SEPARATION_M:
                continue
            if base_index is not None and clearance[base_index] > NEAR_BASE_MAX_CLEARANCE_M:
                continue
            margin = float(np.minimum(xy - lower, upper - xy).min())
            if stratum == "grid_perimeter" and margin > PERIMETER_STRIP_M:
                continue
            table_margin = np.minimum(xy - tabletop_low, tabletop_high - xy).min()
            if table_margin < geometry["bucket_radius_m"]:
                continue
            if np.linalg.norm(used - xy, axis=1).min() < MIN_SEPARATION_M:
                continue
            nearest = min(usable, key=lambda key: float(np.linalg.norm(usable[key] - xy)))
            ix, iy = np.searchsorted(xs, xy[0]) - 1, np.searchsorted(ys, xy[1]) - 1
            identity = f"intergrid-test-{len(layouts):02d}"
            layouts.append(dict(
                id=identity, context_id=identity, split="test", stratum=stratum,
                difficulty={"near_fixed_base": "base_proximity", "grid_perimeter": "edge"}.get(stratum, "interior"),
                bucket_xy_m=xy.tolist(), grid_cell=[int(ix), int(iy)],
                nearest_grid_id=f"fixed35-{nearest}",
                nearest_grid_distance_m=float(np.linalg.norm(usable[nearest] - xy)),
                grid_perimeter_distance_m=margin,
                fixed_base_clearance_lower_bound_m=float(clearance.min()),
            ))
            used = np.vstack((used, xy))
            return
        raise ValueError(f"Could not place the {stratum} sample in bounds {box_low} to {box_high}")

    for iy in range(len(ys) - 1):
        for ix in range(len(xs) - 1):
            corner = np.array([xs[ix], ys[iy]])
            span = np.array([xs[ix + 1], ys[iy + 1]]) - corner
            sample(corner + .25 * span, corner + .75 * span, "rectangle_center")
            perimeter = ix in (0, len(xs) - 2) or iy in (0, len(ys) - 2)
            stratum = "grid_perimeter" if perimeter else "rectangle_interior"
            sample(corner + .02 * span, corner + .98 * span, stratum)
    for index in range(len(fixed)):
        sample(lower + 1e-6, upper - 1e-6, "near_fixed_base", base_index=index)

    provenance = dict(
        version=VERSION, seed=seed, minimum_separation_m=MIN_SEPARATION_M,
        grid_perimeter_strip_m=PERIMETER_STRIP_M,
        near_base_max_clearance_m=NEAR_BASE_MAX_CLEARANCE_M,
        usable_grid={key: list(xy) for key, xy in usable.items()},
        grid_bounds_m=[lower.tolist(), upper.tolist()],
        historical_coordinate_count=len(historical), historical_files_sha256=files,
        excluded_coordinates_m=[list(xy) for xy in excluded],
        excluded_coordinates_sha256=digest(excluded),
        layouts_sha256=digest(layouts), geometry=geometry,
        scope="Novelty against the inventoried archives and canonical coordinates. Geometry strata use no IK or flight outcomes.",
    )
    return layouts, provenance
