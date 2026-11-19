"""Outcome-free broad shot proposals and strict endpoint IK, without Isaac/GPU.

The analytic frictionless bounce is a proposal prior, never a capture label.
Every admitted endpoint still needs the native settling and contact evaluator.
"""

import copy
import hashlib
import json
import math
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from expanded_pose_menu import (
    _action,
    canonical_hash,
    endpoint_targets,
    geometry_reason as common_geometry_reason,
    geometry_signature,
    make_manifest,
    outer_bounds,
)
from prepare_assets import chain, fk, matrix

VERSION = "sircus-physics-informed-shot-sweep-v2"
SEED = 20261112
REST_PRIORS = (0.45, 0.55, 0.65, 0.75, 0.85)
RIM_PLANE_PRIORS = (-1.0, 0.0, 1.0)


def ballistic_gap(
    scene, height, distance, tilt_deg, restitution=None, *, rim_offset_r=1.0
):
    """Invert a frictionless board reflection and descending rim-plane crossing.

    Height is board-center Z. Gap is initial ball-bottom Z minus board-center Z.
    Sphere/board thickness move the actual impact-center Z above board center.
    ``None`` means this approximation has no finite positive-speed solution;
    it does not prove that the native shot is impossible.
    """
    e = scene["ball"]["restitution"] if restitution is None else restitution
    values = np.asarray([height, distance, tilt_deg, e, rim_offset_r], dtype=float)
    if not np.isfinite(values).all() or distance <= 0 or not 0 < tilt_deg < 90:
        return None
    if e < 0 or e > 1:
        return None
    theta = math.radians(tilt_deg)
    r = scene["ball"]["radius_m"]
    z_impact = height + (r + scene["board"]["size_m"][2] / 2) / math.cos(theta)
    z_target = (
        scene["bucket"]["assumed_interior_depth_m"]
        + scene["bucket"]["assumed_bottom_thickness_m"]
        + rim_offset_r * r
    )
    a = (1 + e) * math.sin(theta) * math.cos(theta)
    b = (1 + e) * math.cos(theta) ** 2 - 1
    denominator = z_impact + b * distance / a - z_target
    if denominator <= 0:
        return None
    fall = distance**2 / (4 * a**2 * denominator)
    gap = fall + z_impact - height - r
    # Reject the ascending crossing of the rim plane; entry requires descent.
    speed = math.sqrt(2 * 9.81 * fall)
    time_to_target = distance / (a * speed)
    if b * speed - 9.81 * time_to_target >= 0:
        return None
    return float(gap) if math.isfinite(gap) and gap > 0 else None


def analytic_ballistics(scene, action):
    """Return explicitly approximate geometry diagnostics, not success labels."""
    from feasibility_contract import proposed_range

    required = ballistic_gap(
        scene, action["height_m"], action["radius_m"], action["tilt_deg"]
    )
    predicted = proposed_range(
        scene, action["height_m"], action["gap_m"], action["tilt_deg"]
    )
    return dict(
        model="frictionless-static-plane-sphere-v1",
        restitution=scene["ball"]["restitution"],
        required_gap_m=required,
        predicted_range_m=predicted,
        range_error_m=float(predicted - action["radius_m"]),
        scope="Proposal-only approximation; friction, compliant contact and robots omitted.",
    )


def workspace_bounds(scene, workspace):
    if workspace == "common":
        return (0.38, 0.94), (-0.50, 0.30)
    if workspace != "table":
        raise ValueError("Workspace must be common or table")
    size, center = scene["table"]["size_m"], scene["table"]["center_m"]
    return tuple(
        (center[axis] - size[axis] / 2, center[axis] + size[axis] / 2)
        for axis in (0, 1)
    )


def geometry_reason(scene, cell, action, workspace="common"):
    """Preserve common gates or opt into a full tabletop endpoint proposal box.

    The tabletop mode changes only proposal coverage, not geometry or release
    gates. The board center stays inside the original tabletop with15mm margin;
    the arm-held board may overhang its edge. Actual bottom clearance is checked
    after IK. A projected overhang is not itself a contact or invalid shot.
    Neither mode certifies a collision-free arm posture or swept path.
    """
    if workspace == "common":
        return common_geometry_reason(scene, cell, action)
    bounds = workspace_bounds(scene, workspace)
    vector = np.array(
        [action[k] for k in ("board_x_m", "board_y_m", "height_m", "gap_m", "tilt_deg")]
    )
    if not np.isfinite(vector).all():
        return "nonfinite_geometry"
    x, y, h, gap, tilt = vector
    if not (0.14 <= h <= 0.54 and 0.08 <= gap <= 0.56 and 12 <= tilt <= 52):
        return "proposal_bounds"
    radius = float(np.linalg.norm(np.asarray(cell["bucket_xy_m"]) - [x, y]))
    separation = (
        np.linalg.norm(np.array(scene["board"]["size_m"][:2]) / 2)
        + scene["bucket"]["outer_top_diameter_m"] / 2
        + 0.02
    )
    if radius < max(0.28, separation) or radius > 1.2 + 1e-9:
        return "board_bucket_separation"
    for value, (low, high) in zip((x, y), bounds):
        if value < low + 0.015 or value > high - 0.015:
            return "board_center_workspace"
    return None


def board_geometry_diagnostics(scene, arm14_fit, links):
    """Actual FK rectangle bounds and overhang; overhang is diagnostic only."""
    base = matrix([scene["arms"]["base_spacing_m"], 0, 0], [0, 0, math.pi])
    mount = matrix(
        [0, 0, scene["board"]["tool_offset_m"]],
        [0, 0, scene["board"]["mount_yaw_rad"]],
    )
    board = base @ fk(np.asarray(arm14_fit["q"]), links) @ mount
    extents = np.abs(board[:3, :3]) @ (np.asarray(scene["board"]["size_m"]) / 2)
    minimum, maximum = board[:3, 3] - extents, board[:3, 3] + extents
    bounds = workspace_bounds(scene, "table")
    return dict(
        center_m=board[:3, 3].tolist(),
        normal=board[:3, 2].tolist(),
        bounds_min_m=minimum.tolist(),
        bounds_max_m=maximum.tolist(),
        bottom_clearance_m=float(minimum[2]),
        overhang_xy_m=[
            [
                max(0.0, float(low - minimum[axis])),
                max(0.0, float(maximum[axis] - high)),
            ]
            for axis, (low, high) in enumerate(bounds)
        ],
        scope="Actual FK board bounding box; projected table overhang is allowed and not a capture/contact label.",
    )


def fitted_geometry_reason(scene, arm14_fit, links, workspace="table", action=None):
    """Check actual board center/bottom, including unconstrained in-plane spin.

    Arm14 IK constrains the flange normal, not rotation about that normal.
    The nominal target footprint alone therefore cannot certify the fitted
    board corners. Held-board overhang is allowed. This is FK geometry only,
    not dynamic/path validation; measured native checks remain necessary.
    """
    if workspace == "common":
        return None
    bounds = workspace_bounds(scene, workspace)
    diagnostic = board_geometry_diagnostics(scene, arm14_fit, links)
    center = np.asarray(diagnostic["center_m"])
    if (
        action is not None
        and np.linalg.norm(
            center
            - np.array([action["board_x_m"], action["board_y_m"], action["height_m"]])
        )
        >= 1e-4
    ):
        return "actual_board_center_error"
    if diagnostic["bottom_clearance_m"] < 0.015:
        return "actual_board_table_clearance"
    for value, (low, high) in zip(center[:2], bounds):
        if value < low + 0.015 or value > high - 0.015:
            return "actual_board_center_workspace"
    return None


def check_fitted_geometry(scene, cell, action, fits, links, workspace="table"):
    """Revalidate strict IK and the actual rotated board before native tests."""
    from expanded_farm import validate_action
    from expanded_pose_native import verified_fit

    validate_action(cell, action)
    reason = geometry_reason(scene, cell, action, workspace)
    if reason:
        return reason
    targets = endpoint_targets(scene, cell, action)
    for name in ("Arm10", "Arm14"):
        verified_fit(targets[name], fits[name], links[name], name == "Arm14")
    return fitted_geometry_reason(
        scene, fits["Arm14"], links["Arm14"], workspace, action
    )


def filter_manifest(source, assets, *, workspace="table"):
    """Post-filter a frozen CPU bank without rerunning or relabeling its search.

    A new offline manifest binds the exact input digest and records every
    center/clearance/IK rejection. It never manufactures native qualification or fills
    an emptied cell with an old control that failed the new geometric checks.
    """
    if isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
        value = json.loads(raw)
    else:
        value = copy.deepcopy(source)
        raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    unsigned = dict(value)
    fingerprint = unsigned.pop("fingerprint", None)
    if canonical_hash(unsigned) != fingerprint or value.get("qualification") not in (
        "offline_endpoint_ik",
        "native_settled_endpoints",
    ):
        raise ValueError("Frozen search manifest checksum or qualification mismatch")
    spec = value["spec"]
    if (
        len(spec["cells"]) != 35
        or len(value["menus"]) != 35
        or len(value["fits"]) != 35
    ):
        raise ValueError("Geometry filter preserves all 35 original sites")
    if spec["fingerprint"] != canonical_hash(
        {k: v for k, v in spec.items() if k != "fingerprint"}
    ):
        raise ValueError("Frozen task identity changed")
    assets = Path(assets)
    links = {
        name: chain(ET.parse(assets / filename).getroot())
        for name, filename in (("Arm10", "dropper.urdf"), ("Arm14", "board_arm.urdf"))
    }
    menus, fits, checks = ([[] for _ in range(35)] for _ in range(3))
    for i, (cell, actions, solutions) in enumerate(
        zip(spec["cells"], value["menus"], value["fits"])
    ):
        if len(actions) != len(solutions):
            raise ValueError("Frozen menu is missing endpoint fits")
        for action, fit in zip(actions, solutions):
            reason = check_fitted_geometry(
                spec["scene"], cell, action, fit, links, workspace
            )
            checks[i].append(
                dict(action_id=action["id"], reason=reason, ik_checked=True)
            )
            if reason is None:
                menus[i].append(copy.deepcopy(action))
                fits[i].append(copy.deepcopy(fit))
    return make_manifest(
        spec,
        menus,
        fits=fits,
        checks=checks,
        qualification="offline_endpoint_ik",
        search_proposal_version=value.get("search_proposal_version", VERSION),
        proposal_protocol=copy.deepcopy(value.get("proposal_protocol", {})),
        filter_protocol=dict(
            version="sircus-actual-fk-board-center-clearance-v2",
            workspace=workspace,
            margin_m=0.015,
            source_sha256=hashlib.sha256(raw).hexdigest(),
            source_manifest_fingerprint=fingerprint,
            source_qualification=value["qualification"],
            source_counts=[len(x) for x in value["menus"]],
            retained_counts=[len(x) for x in menus],
            held_board_overhang_allowed=True,
            scope="Strict endpoint FK, board center and bottom clearance only; footprint overhang is diagnostic. Not a native/path/capture certificate.",
        ),
        summary=dict(
            accepted_per_cell=[len(x) for x in menus],
            reasons=dict(
                Counter(
                    c["reason"] or "actual_fk_geometry_accepted"
                    for group in checks
                    for c in group
                )
            ),
        ),
        scope="Post-filtered frozen CPU proposals with actual FK footprint checks; native settling/contact validation still required.",
    )


def proposals(scene, cell, limit=2048, *, seed=SEED, workspace="common"):
    """Interleave inverse-ballistic and broad low-discrepancy coverage, 3:1.

    Analytic proposals sample board XY, height, tilt and restitution priors,
    solving the release gap. Broad proposals sample XY, height, gap and tilt
    independently so an inaccurate analytic model cannot exclude a region.
    Radius and bearing are derived exactly from the fixed bucket and board XY.
    """
    if not 1 <= limit <= 32768:
        raise ValueError("Candidate limit must be 1..32768")
    (xl, xh), (yl, yh) = workspace_bounds(scene, workspace)
    cell_seed = seed + 101 * int(cell.get("index", 0))
    engine = qmc.Halton(5, scramble=True, seed=cell_seed)
    broad_engine = qmc.Halton(5, scramble=True, seed=cell_seed + 7919)
    streams = [[], []]
    # Each stream is finite; the independent broad stream can fill the bank even
    # when the analytic prior has no eligible prediction for a distant bucket.
    count = max(1024, 12 * limit)
    for stream, samples in enumerate(
        (engine.random(count), broad_engine.random(count))
    ):
        for u in samples:
            x, y = xl + u[0] * (xh - xl), yl + u[1] * (yh - yl)
            height = 0.14 + u[2] * 0.40
            if stream == 0:
                tilt = 12 + u[3] * 40
                distance = float(np.linalg.norm(np.array(cell["bucket_xy_m"]) - [x, y]))
                prior_index = min(
                    int(u[4] * len(REST_PRIORS) * len(RIM_PLANE_PRIORS)),
                    len(REST_PRIORS) * len(RIM_PLANE_PRIORS) - 1,
                )
                prior = REST_PRIORS[prior_index // len(RIM_PLANE_PRIORS)]
                plane = RIM_PLANE_PRIORS[prior_index % len(RIM_PLANE_PRIORS)]
                gap = ballistic_gap(
                    scene, height, distance, tilt, prior, rim_offset_r=plane
                )
                if gap is None:
                    continue
            else:
                gap, tilt = 0.08 + u[3] * 0.48, 12 + u[4] * 40
            action = _action(cell, x, y, height, gap, tilt, "unassigned")
            if geometry_reason(scene, cell, action, workspace) is None:
                streams[stream].append(action)
                if len(streams[stream]) >= limit:
                    break
    result, positions = [], [0, 0]
    for n in range(2 * limit):
        which = 1 if n % 4 == 3 else 0
        if positions[which] == len(streams[which]):
            which = 1 - which
        if positions[which] == len(streams[which]):
            break
        action = streams[which][positions[which]]
        positions[which] += 1
        action["id"] = f"shot-sweep-{cell.get('index', 0):02d}-{len(result):05d}"
        result.append(action)
        if len(result) >= limit:
            break
    return result


def read_seed_manifest(path, spec):
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    unsigned = dict(value)
    fingerprint = unsigned.pop("fingerprint", None)
    if canonical_hash(unsigned) != fingerprint or value.get("spec") != spec:
        raise ValueError("Seed manifest checksum or original task mismatch")
    if value.get("qualification") not in (
        "native_settled_endpoints",
        "offline_endpoint_ik",
    ):
        raise ValueError("Seed must contain explicit endpoint qualification")
    if len(value["menus"]) != 35 or len(value["fits"]) != 35:
        raise ValueError("Seed must preserve all 35 original sites")
    if any(len(a) != len(b) for a, b in zip(value["menus"], value["fits"])):
        raise ValueError("Seed contains missing endpoint fits")
    return value, hashlib.sha256(raw).hexdigest()


def qualify(
    spec,
    assets,
    *,
    seed_manifest=None,
    cell_indices=None,
    max_checks_per_cell=512,
    accepted_per_cell=128,
    candidates_per_cell=2048,
    controls=2,
    workspace="table",
    progress=None,
    partial=None,
):
    """Broad bounded CPU search; accepted means IK only, not native success.

    Retain only ``controls`` seed options first in requested sites, followed by
    broad new options. This prevents an old 24-option local menu from consuming
    the search budget. Unrequested sites retain the full seed menu unchanged.
    All seeds are independently FK-checked before use. No capture outcomes or
    reward labels enter this function.
    """
    from expanded_farm import validate_action
    from expanded_pose_native import verified_fit
    from robust_ik import solve

    if not 1 <= max_checks_per_cell <= 8192 or not 2 <= accepted_per_cell <= 512:
        raise ValueError("Use 1..8192 checks and 2..512 accepted endpoints")
    if not 0 <= controls < accepted_per_cell:
        raise ValueError("Control count must leave room for broad proposals")
    unsigned = dict(spec)
    fingerprint = unsigned.pop("fingerprint")
    if canonical_hash(unsigned) != fingerprint or len(spec["cells"]) != 35:
        raise ValueError("Expected the sealed original 35-site task")
    workspace_bounds(spec["scene"], workspace)
    active = list(range(35)) if cell_indices is None else list(cell_indices)
    if len(set(active)) != len(active) or any(i not in range(35) for i in active):
        raise ValueError("Choose distinct original cell indices")
    assets = Path(assets)
    links = {
        name: chain(ET.parse(assets / filename).getroot())
        for name, filename in (("Arm10", "dropper.urdf"), ("Arm14", "board_arm.urdf"))
    }
    poses = json.loads((assets / "poses.json").read_text())
    initial = dict(Arm10=poses["dropper_joint_rad"], Arm14=poses["board_joint_rad"])
    menus, fits, checks = ([[] for _ in range(35)] for _ in range(3))
    seed_hash = None
    if seed_manifest:
        source, seed_hash = read_seed_manifest(seed_manifest, spec)
        menus, fits = copy.deepcopy(source["menus"]), copy.deepcopy(source["fits"])
    # Exact target keys preserve the native prepare target convention. Never
    # round keys: cached solutions are independently checked against each target.
    cache = {}
    for cell, group, solutions in zip(spec["cells"], menus, fits):
        for action, by_arm in zip(group, solutions):
            validate_action(cell, action)
            targets = endpoint_targets(spec["scene"], cell, action)
            for name in initial:
                record = verified_fit(
                    targets[name], by_arm[name], links[name], name == "Arm14"
                )
                cache[name, tuple(targets[name].ravel())] = record
    started = time.monotonic()
    protocol = dict(
        version=VERSION,
        workspace=workspace,
        proposal_xy_bounds=workspace_bounds(spec["scene"], workspace),
        control_options_per_requested_site=controls,
        max_checks_per_cell=max_checks_per_cell,
        accepted_per_cell=accepted_per_cell,
        candidates_per_cell=candidates_per_cell,
        inverse_ballistic_to_broad_ratio="3:1 where eligible; broad fallback",
        restitution_priors=list(REST_PRIORS),
        rim_plane_radius_offsets=list(RIM_PLANE_PRIORS),
        held_board_overhang_allowed=True,
        actual_fk_board_center_tolerance_m=1e-4,
        actual_fk_board_bottom_clearance_m=0.015,
        geometry_only=True,
        outcome_scope="Caller may prioritize failed sites; proposals/IK use no capture outcomes.",
    )

    def manifest():
        return make_manifest(
            spec,
            menus,
            fits=fits,
            checks=checks,
            qualification="offline_endpoint_ik",
            seed_manifest_sha256=seed_hash,
            search_proposal_version=VERSION,
            source_original_geometry_fp=spec.get("camera_source_fingerprint"),
            proposal_protocol=protocol,
            summary=dict(
                accepted_per_cell=[len(m) for m in menus],
                requested_cells=active,
                ik_checks_per_cell=[
                    sum(c.get("ik_checked", False) for c in group) for group in checks
                ],
                reasons=dict(
                    Counter(
                        c["reason"] or "endpoint_ik_qualified"
                        for group in checks
                        for c in group
                    )
                ),
                elapsed_s=time.monotonic() - started,
            ),
            scope="Outcome-free broad proposal/strict endpoint IK only; native settling and ball contact validation required. No path certificate or impossibility claim.",
        )

    for index in active:
        cell = spec["cells"][index]
        # Keep seed solutions as nearest-target warm starts even when the old
        # narrow options are removed from the actual trial menu.
        warm_actions = copy.deepcopy(menus[index])
        warm_fits = copy.deepcopy(fits[index])
        warm_targets = {name: [] for name in initial}
        for old in warm_actions:
            targets = endpoint_targets(spec["scene"], cell, old)
            for name in initial:
                warm_targets[name].append(targets[name])
        menus[index], fits[index] = menus[index][:controls], fits[index][:controls]
        seen = {geometry_signature(a) for a in menus[index]}
        candidates = proposals(
            spec["scene"], cell, candidates_per_cell, workspace=workspace
        )
        used = 0
        for action in candidates:
            if len(menus[index]) >= accepted_per_cell or used >= max_checks_per_cell:
                break
            if geometry_signature(action) in seen:
                continue
            targets = endpoint_targets(spec["scene"], cell, action)
            bound = outer_bounds(targets, links)
            if not bound["eligible"]:
                checks[index].append(
                    dict(
                        action_id=action["id"],
                        reason="outer_link_length_bound",
                        ik_checked=False,
                    )
                )
                continue
            used += 1
            result, reason = {}, None
            for name in initial:
                key = name, tuple(targets[name].ravel())
                if key not in cache:
                    if warm_actions:
                        # Nearest requested endpoint in physical position/normal
                        # supplies a warm start, but never changes IK acceptance.
                        old_targets = np.asarray(warm_targets[name])
                        errors = np.linalg.norm(
                            old_targets[:, :3, 3] - targets[name][:3, 3], axis=1
                        ) + 0.25 * np.linalg.norm(
                            old_targets[:, :3, 2] - targets[name][:3, 2], axis=1
                        )
                        seed = warm_fits[int(np.argmin(errors))][name]["q"]
                    else:
                        seed = initial[name]
                    cache[key] = solve(
                        targets[name], links[name], seed, name == "Arm14"
                    )
                result[name] = cache[key]
                if not result[name]["solved"]:
                    reason = "endpoint_ik_" + name
                    break
                verified_fit(targets[name], result[name], links[name], name == "Arm14")
            if reason is None:
                reason = check_fitted_geometry(
                    spec["scene"], cell, action, result, links, workspace
                )
            checks[index].append(
                dict(
                    action_id=action["id"],
                    reason=reason,
                    ik_checked=True,
                    ik=result,
                    analytic_prior=analytic_ballistics(spec["scene"], action),
                )
            )
            if reason is None:
                menus[index].append(action)
                fits[index].append(result)
                warm_actions.append(action)
                warm_fits.append(result)
                for name in initial:
                    warm_targets[name].append(targets[name])
                seen.add(geometry_signature(action))
            if progress:
                progress(
                    dict(
                        cell=index,
                        cell_id=cell["id"],
                        ik_checks=used,
                        accepted=len(menus[index]),
                        elapsed_s=time.monotonic() - started,
                    )
                )
        if partial:
            partial(manifest())
    return manifest()


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-native", type=Path)
    parser.add_argument("--cells", type=int, nargs="+")
    parser.add_argument("--max-checks", type=int, default=512)
    parser.add_argument("--accepted", type=int, default=128)
    parser.add_argument("--candidates", type=int, default=2048)
    parser.add_argument("--controls", type=int, default=2)
    parser.add_argument("--workspace", choices=("common", "table"), default="table")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output; preserve previous search evidence")
    spec = json.loads(args.spec.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint(value):
        path = args.output.with_suffix(".partial.json")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, allow_nan=False))
        temporary.replace(path)

    result = qualify(
        spec,
        args.assets,
        seed_manifest=args.seed_native,
        cell_indices=args.cells,
        max_checks_per_cell=args.max_checks,
        accepted_per_cell=args.accepted,
        candidates_per_cell=args.candidates,
        controls=args.controls,
        workspace=args.workspace,
        progress=lambda row: print(json.dumps(row), flush=True),
        partial=checkpoint,
    )
    args.output.write_text(json.dumps(result, allow_nan=False))
    print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
