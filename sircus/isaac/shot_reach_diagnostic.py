"""CPU-only model/URDF reach audit; never a native feasibility or capture proof.

Compare the current shot bounds with a wider diagnostic box without changing
either production contract. A frictionless model ranks candidates, conservative
link-length spheres remove only obvious reach violations, then strict IK checks
a bounded set of model-matched endpoints. No simulator or hardware is imported.
"""

import argparse
import hashlib
import itertools
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from expanded_pose_menu import _action, canonical_hash, endpoint_targets
from prepare_assets import chain
from shot_sweep import ballistic_gap, board_geometry_diagnostics

VERSION = "sircus-model-reach-energy-diagnostic-v1"
DOMAINS = {
    "current": dict(height_m=[0.14, 0.54], gap_m=[0.08, 0.56], tilt_deg=[12.0, 52.0]),
    "wider_diagnostic": dict(
        height_m=[0.06, 0.85], gap_m=[0.02, 1.20], tilt_deg=[5.0, 65.0]
    ),
}


def reach_sphere(links):
    last_origin, last_axis = links[-1][:2]
    terminal_axis = last_origin[:3, :3] @ last_axis
    offset = float(last_origin[:3, 3] @ terminal_axis)
    perpendicular = float(np.linalg.norm(last_origin[:3, 3] - offset * terminal_axis))
    if not np.allclose(last_axis, [0, 0, 1], atol=1e-12):
        raise ValueError(
            "Normal-aware bound requires a terminal flange-Z revolute axis"
        )
    return dict(
        pivot_local_m=links[0][0][:3, 3].tolist(),
        outer_radius_m=float(sum(np.linalg.norm(link[0][:3, 3]) for link in links[1:])),
        terminal_normal_offset_m=offset,
        terminal_perpendicular_offset_m=perpendicular,
        normal_wrist_outer_radius_m=float(
            sum(np.linalg.norm(link[0][:3, 3]) for link in links[1:-1]) + perpendicular
        ),
        scope="Conservative sum-of-link-origin-length outer bound; not orientation-aware reach or a sufficient IK condition.",
    )


def release_ceiling(xy, sphere, flange_offset=0.095, normal_wrist_bound=True):
    pivot = np.asarray(sphere["pivot_local_m"])
    horizontal2 = np.sum((np.asarray(xy) - pivot[:2]) ** 2)
    radius = (
        sphere["normal_wrist_outer_radius_m"]
        if normal_wrist_bound
        else sphere["outer_radius_m"]
    )
    remaining = radius**2 - horizontal2
    if remaining < 0:
        return None
    terminal = sphere["terminal_normal_offset_m"] if normal_wrist_bound else 0
    return float(pivot[2] + math.sqrt(remaining) - terminal - flange_offset)


def model_range(scene, height, gap, tilt_deg, rim_offset_r=1.0):
    theta = np.radians(tilt_deg)
    r = scene["ball"]["radius_m"]
    impact = height + (r + scene["board"]["size_m"][2] / 2) / np.cos(theta)
    fall = height + r + gap - impact
    speed = np.sqrt(np.maximum(0, 2 * 9.81 * fall))
    e = scene["ball"]["restitution"]
    vx = (1 + e) * speed * np.sin(theta) * np.cos(theta)
    vz = speed * ((1 + e) * np.cos(theta) ** 2 - 1)
    target = (
        scene["bucket"]["assumed_interior_depth_m"]
        + scene["bucket"]["assumed_bottom_thickness_m"]
        + rim_offset_r * r
    )
    discriminant = vz**2 + 2 * 9.81 * (impact - target)
    distance = vx * (vz + np.sqrt(np.maximum(discriminant, 0))) / 9.81
    return np.where(
        (fall > 0) & (discriminant >= 0) & (distance >= 0), distance, np.nan
    )


def grid(lower, upper, step):
    values = np.arange(lower, upper + step * 0.25, step)
    return np.unique(np.round(np.r_[values[values <= upper + 1e-9], upper], 9))


def analyze_cell(
    scene,
    cell,
    spheres,
    *,
    xy_step=0.10,
    height_step=0.02,
    tilt_step=2.0,
    top=24,
    normal_wrist_bound=True,
    rim_offset_r=0.0,
):
    center, size = np.array(scene["table"]["center_m"]), np.array(
        scene["table"]["size_m"]
    )
    xs = grid(center[0] - size[0] / 2 + 0.015, center[0] + size[0] / 2 - 0.015, xy_step)
    ys = grid(center[1] - size[1] / 2 + 0.015, center[1] + size[1] / 2 - 0.015, xy_step)
    bucket = np.asarray(cell["bucket_xy_m"])
    board_pivot = np.array(spheres["Arm14"]["pivot_local_m"]) + [
        scene["arms"]["base_spacing_m"],
        0,
        0,
    ]
    r = scene["ball"]["radius_m"]
    half_diag = float(np.linalg.norm(np.array(scene["board"]["size_m"][:2]) / 2))
    thickness = scene["board"]["size_m"][2]
    separation = max(
        0.28, half_diag + scene["bucket"]["outer_top_diameter_m"] / 2 + 0.020
    )
    result = {}
    for name, bounds in DOMAINS.items():
        heights = grid(*bounds["height_m"], height_step)
        tilts = grid(*bounds["tilt_deg"], tilt_step)
        if name == "wider_diagnostic":
            heights = np.unique(
                np.r_[heights, grid(*DOMAINS["current"]["height_m"], height_step)]
            )
            tilts = np.unique(
                np.r_[tilts, grid(*DOMAINS["current"]["tilt_deg"], tilt_step)]
            )
        h, angle = np.meshgrid(heights, tilts, indexing="ij")
        theta = np.radians(angle)
        sin, cos = np.sin(theta), np.cos(theta)
        impact = h + (r + thickness / 2) / cos
        target_z = (
            scene["bucket"]["assumed_interior_depth_m"]
            + scene["bucket"]["assumed_bottom_thickness_m"]
            + rim_offset_r * r
        )
        e = scene["ball"]["restitution"]
        a, b = (1 + e) * sin * cos, (1 + e) * cos**2 - 1
        max_range = None
        max_margin = None
        best_range_action = None
        best_margin_action = None
        matches = 0
        pool = []
        evaluated = 0
        for x, y in itertools.product(xs, ys):
            distance = float(np.linalg.norm(bucket - [x, y]))
            if not separation <= distance <= 1.2:
                continue
            ceiling = release_ceiling(
                [x, y], spheres["Arm10"], normal_wrist_bound=normal_wrist_bound
            )
            if ceiling is None:
                continue
            gap_cap = np.minimum(bounds["gap_m"][1], ceiling - h - r)
            direction = (bucket - [x, y]) / distance
            mount_offset = scene["board"]["tool_offset_m"] + (
                spheres["Arm14"]["terminal_normal_offset_m"]
                if normal_wrist_bound
                else 0
            )
            board_radius = (
                spheres["Arm14"]["normal_wrist_outer_radius_m"]
                if normal_wrist_bound
                else spheres["Arm14"]["outer_radius_m"]
            )
            drop_radius = (
                spheres["Arm10"]["normal_wrist_outer_radius_m"]
                if normal_wrist_bound
                else spheres["Arm10"]["outer_radius_m"]
            )
            flange_x = x - mount_offset * direction[0] * sin
            flange_y = y - mount_offset * direction[1] * sin
            flange_z = h - mount_offset * cos
            board_distance = np.sqrt(
                (flange_x - board_pivot[0]) ** 2
                + (flange_y - board_pivot[1]) ** 2
                + (flange_z - board_pivot[2]) ** 2
            )
            # Worst in-plane board spin; a deliberately conservative analytic
            # screen. Actual fitted geometry is checked after strict IK below.
            clearance = h - half_diag * sin - thickness / 2 * cos
            eligible = (
                (gap_cap >= bounds["gap_m"][0])
                & (board_distance <= board_radius)
                & (clearance >= 0.015)
            )
            ranges = model_range(scene, h, gap_cap, angle, rim_offset_r)
            ranges = np.where(eligible, ranges, np.nan)
            evaluated += int(eligible.sum())
            if np.isfinite(ranges).any():
                index = np.unravel_index(np.nanargmax(ranges), ranges.shape)
                value = float(ranges[index])
                margin = value - distance
                if max_range is None or value > max_range:
                    max_range = value
                    best_range_action = _action(
                        cell,
                        x,
                        y,
                        h[index],
                        gap_cap[index],
                        angle[index],
                        "analytic-maximum",
                    )
                if max_margin is None or margin > max_margin:
                    max_margin = margin
                    action = _action(
                        cell,
                        x,
                        y,
                        h[index],
                        gap_cap[index],
                        angle[index],
                        "analytic-smallest-deficit",
                    )
                    required = ballistic_gap(
                        scene,
                        action["height_m"],
                        distance,
                        action["tilt_deg"],
                        rim_offset_r=rim_offset_r,
                    )
                    best_margin_action = dict(
                        action=action,
                        predicted_range_m=value,
                        required_distance_m=distance,
                        model_required_gap_m=required,
                        available_gap_m=float(gap_cap[index]),
                        model_required_extra_gap_m=(
                            max(0.0, required - gap_cap[index])
                            if required is not None
                            else None
                        ),
                        available_release_ball_center_ceiling_m=ceiling,
                    )
            denominator = impact + b * distance / a - target_z
            with np.errstate(divide="ignore", invalid="ignore"):
                fall = distance**2 / (4 * a**2 * denominator)
                gap = fall + impact - h - r
                speed = np.sqrt(2 * 9.81 * fall)
                time = distance / (a * speed)
                downward = b * speed - 9.81 * time < 0
            valid = (
                eligible
                & (denominator > 0)
                & downward
                & np.isfinite(gap)
                & (gap >= bounds["gap_m"][0])
                & (gap <= gap_cap)
            )
            matches += int(valid.sum())
            if not valid.any():
                continue
            release_z = h + r + gap
            terminal = (
                spheres["Arm10"]["terminal_normal_offset_m"]
                if normal_wrist_bound
                else 0
            )
            drop_distance = np.sqrt(
                x * x
                + y * y
                + (release_z + 0.095 + terminal - spheres["Arm10"]["pivot_local_m"][2])
                ** 2
            )
            score = (
                np.maximum(drop_distance / drop_radius, board_distance / board_radius)
                + 0.08 * gap
            )
            score = np.where(valid, score, np.inf)
            for flat in np.argsort(score.ravel())[:3]:
                index = np.unravel_index(flat, score.shape)
                if not np.isfinite(score[index]):
                    continue
                action = _action(
                    cell, x, y, h[index], gap[index], angle[index], "analytic-matched"
                )
                pool.append(
                    dict(
                        action=action,
                        rim_plane_offset_radii=rim_offset_r,
                        model_match_error_m=float(
                            model_range(
                                scene, h[index], gap[index], angle[index], rim_offset_r
                            )
                            - distance
                        ),
                        reach_rank=float(score[index]),
                        outside_current_bounds=any(
                            not DOMAINS["current"][key][0]
                            <= action[field]
                            <= DOMAINS["current"][key][1]
                            for key, field in (
                                ("height_m", "height_m"),
                                ("gap_m", "gap_m"),
                                ("tilt_deg", "tilt_deg"),
                            )
                        ),
                    )
                )
        pool.sort(key=lambda v: v["reach_rank"])
        # Preserve several XY/height/tilt regions rather than just adjacent
        # entries from one analytic optimum.
        selected = []
        for candidate in pool:
            action = candidate["action"]
            if any(
                abs(action["board_x_m"] - old["action"]["board_x_m"]) < 0.08
                and abs(action["board_y_m"] - old["action"]["board_y_m"]) < 0.08
                and abs(action["height_m"] - old["action"]["height_m"]) < 0.06
                and abs(action["tilt_deg"] - old["action"]["tilt_deg"]) < 6
                for old in selected
            ):
                continue
            selected.append(candidate)
            if len(selected) >= top:
                break
        result[name] = dict(
            evaluated_outer_bound_strata=evaluated,
            model_matched_strata=matches,
            max_model_range_m=max_range,
            max_model_range_minus_required_distance_m=max_margin,
            best_range_action=best_range_action,
            best_margin_action=best_margin_action,
            promising_model_candidates=selected,
        )
    return dict(cell_id=cell["id"], bucket_xy_m=cell["bucket_xy_m"], **result)


def strict_ik_candidates(scene, cell, candidates, links, poses, checks=6):
    from expanded_pose_native import verified_fit
    from robust_ik import solve

    records = []
    seeds = dict(Arm10=poses["dropper_joint_rad"], Arm14=poses["board_joint_rad"])
    for candidate in candidates[:checks]:
        action = candidate["action"]
        targets = endpoint_targets(scene, cell, action)
        fits = {}
        for name in ("Arm10", "Arm14"):
            fitted = solve(targets[name], links[name], seeds[name], name == "Arm14")
            fits[name] = fitted
            if not fitted["solved"]:
                break
            verified_fit(targets[name], fitted, links[name], name == "Arm14")
        solved = len(fits) == 2 and all(v["solved"] for v in fits.values())
        geometry = (
            board_geometry_diagnostics(scene, fits["Arm14"], links["Arm14"])
            if solved
            else None
        )
        center_error = (
            float(
                np.linalg.norm(
                    np.asarray(geometry["center_m"])
                    - np.array(
                        [action["board_x_m"], action["board_y_m"], action["height_m"]]
                    )
                )
            )
            if geometry
            else None
        )
        passed = bool(
            solved and geometry["bottom_clearance_m"] >= 0.015 and center_error < 1e-4
        )
        records.append(
            dict(
                **candidate,
                model_center_match=abs(candidate["model_match_error_m"]) < 1e-6,
                strict_ik_solved=solved,
                actual_fk_bottom_clearance_m=(
                    geometry["bottom_clearance_m"] if geometry else None
                ),
                actual_fk_board_center_error_m=center_error,
                endpoint_geometry_accepted=passed,
                fits=fits,
            )
        )
        if passed:
            break
    return records


def probe_manifest(spec, assets, native_seed, audit_path):
    """Current-bounds, offline-only C6/C7 probes plus two original controls/site."""
    from expanded_farm import HIGH, LOW, validate_action
    from expanded_pose_menu import geometry_signature, make_manifest
    from expanded_pose_native import verified_fit
    from robust_ik import solve
    from shot_sweep import check_fitted_geometry, read_seed_manifest

    source, seed_sha = read_seed_manifest(native_seed, spec)
    audit_raw = Path(audit_path).read_bytes()
    audit = json.loads(audit_raw)
    if audit["source_fingerprint"] != spec["fingerprint"]:
        raise ValueError("Reach diagnostic belongs to another task")
    links = {
        name: chain(ET.parse(Path(assets) / file).getroot())
        for name, file in (("Arm10", "dropper.urdf"), ("Arm14", "board_arm.urdf"))
    }
    spheres = {name: reach_sphere(value) for name, value in links.items()}
    menus = [[dict(a) for a in group[:2]] for group in source["menus"]]
    fits = [[dict(a) for a in group[:2]] for group in source["fits"]]
    if any(len(group) != 2 for group in menus):
        raise ValueError("Every original site needs two preserved controls")
    rows = {r["cell_id"]: r for r in audit["cells"]}
    diagnostics, checks = {}, [[] for _ in menus]
    parameters = ("azimuth_offset_deg", "radius_m", "height_m", "gap_m", "tilt_deg")
    for i, cell in enumerate(spec["cells"]):
        for a, f in zip(menus[i], fits[i]):
            reason = check_fitted_geometry(spec["scene"], cell, a, f, links, "table")
            if reason:
                raise ValueError(
                    f"Preserved control fails actual FK geometry: {reason}"
                )
        if i % 7 not in (5, 6):
            continue
        row = rows[cell["id"]]
        accepted = next(
            r
            for r in row["current"]["strict_ik_checks"]
            if r["endpoint_geometry_accepted"]
        )
        nominal = dict(accepted["action"])
        options = [
            dict(
                action=nominal,
                plane=accepted.get("rim_plane_offset_radii", 0),
                kind="model_match" if accepted["model_center_match"] else "under_range",
            )
        ]
        limit = 12 if row["current"]["model_matched_strata"] else 3
        if limit == 12:
            for dx, dy in ((0, 0), (-0.025, 0), (0.025, 0), (0, -0.025), (0, 0.025)):
                x, y = nominal["board_x_m"] + dx, nominal["board_y_m"] + dy
                distance = float(np.linalg.norm(np.array(cell["bucket_xy_m"]) - [x, y]))
                ceiling = release_ceiling([x, y], spheres["Arm10"])
                for h, tilt, plane in itertools.product(
                    (0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30),
                    (18.0, 20.0, 22.0, 24.0, 26.0, 28.0),
                    (0.0, -1.0, 1.0),
                ):
                    gap = ballistic_gap(
                        spec["scene"], h, distance, tilt, rim_offset_r=plane
                    )
                    if (
                        gap is not None
                        and ceiling is not None
                        and 0.08 <= gap <= 0.56
                        and h + gap + spec["scene"]["ball"]["radius_m"] <= ceiling
                    ):
                        options.append(
                            dict(
                                action=_action(cell, x, y, h, gap, tilt, "unassigned"),
                                plane=plane,
                                kind="model_match",
                            )
                        )
        else:
            base = row["current"]["best_margin_action"]["action"]
            options.extend(
                dict(
                    action=dict(base, gap_m=base["gap_m"] - reduction),
                    plane=0.0,
                    kind="under_range",
                )
                for reduction in (0.0, 0.02, 0.04, 0.08)
            )
        chosen = []
        seen = {geometry_signature(a) for a in menus[i]}
        while options and len(chosen) < limit:
            if chosen:

                def diversity(option):
                    v = np.array([option["action"][k] for k in parameters])
                    old = np.array([[a[k] for k in parameters] for a in chosen])
                    return float(
                        np.min(np.linalg.norm((old - v) / (HIGH - LOW), axis=1))
                    )

                at = max(range(len(options)), key=lambda n: diversity(options[n]))
            else:
                at = 0
            proposal = options.pop(at)
            a = dict(proposal["action"])
            if geometry_signature(a) in seen:
                continue
            try:
                validate_action(cell, a)
            except ValueError:
                continue
            targets = endpoint_targets(spec["scene"], cell, a)
            fitted = {}
            for name in ("Arm10", "Arm14"):
                old = [
                    endpoint_targets(spec["scene"], cell, old_action)[name]
                    for old_action in menus[i]
                ]
                nearest = min(
                    range(len(old)),
                    key=lambda n: np.linalg.norm(old[n][:3, 3] - targets[name][:3, 3])
                    + 0.25 * np.linalg.norm(old[n][:3, 2] - targets[name][:3, 2]),
                )
                fitted[name] = solve(
                    targets[name],
                    links[name],
                    fits[i][nearest][name]["q"],
                    name == "Arm14",
                )
                if not fitted[name]["solved"]:
                    break
                verified_fit(targets[name], fitted[name], links[name], name == "Arm14")
            if (
                len(fitted) != 2
                or not all(v["solved"] for v in fitted.values())
                or check_fitted_geometry(spec["scene"], cell, a, fitted, links, "table")
                is not None
            ):
                continue
            a["id"] = f"reach-probe-{i:02d}-{len(chosen):03d}"
            diagnostics[a["id"]] = dict(
                kind=proposal["kind"],
                model_restitution=spec["scene"]["ball"]["restitution"],
                target_plane_radius_offset=proposal["plane"],
                predicted_range_m=float(
                    model_range(
                        spec["scene"],
                        a["height_m"],
                        a["gap_m"],
                        a["tilt_deg"],
                        proposal["plane"],
                    )
                ),
                required_distance_m=a["radius_m"],
                scope="Analytic geometry and strict CPU IK only; native outcome unknown.",
            )
            menus[i].append(a)
            fits[i].append(fitted)
            chosen.append(a)
            seen.add(geometry_signature(a))
            checks[i].append(dict(action_id=a["id"], reason=None, ik_checked=True))
    return make_manifest(
        spec,
        menus,
        fits=fits,
        checks=checks,
        qualification="offline_endpoint_ik",
        source_native_sha256=seed_sha,
        reach_diagnostic_sha256=hashlib.sha256(audit_raw).hexdigest(),
        diagnostic_source_sha256=hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        proposal_protocol=dict(
            version="sircus-current-bounds-reach-probe-v1",
            workspace="table",
            original_controls_per_site=2,
            geometry_only=True,
            target_planes_radii=[-1, 0, 1],
            scope="Extra probes only in C6/C7; current five-action bounds, all35bucketpositions and firsttwo controls unchanged.",
        ),
        diagnostic_actions=diagnostics,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xy-step", type=float, default=0.10)
    parser.add_argument("--ik-checks", type=int, default=6)
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--probe-audit", type=Path)
    parser.add_argument("--seed-native", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing diagnostic output")
    if not 0.04 <= args.xy_step <= 0.25 or not 0 <= args.ik_checks <= 24:
        parser.error("Use bounded XY step/check count")
    spec = json.loads(args.spec.read_text())
    unsigned = dict(spec)
    fingerprint = unsigned.pop("fingerprint", None)
    if (
        canonical_hash(unsigned) != fingerprint
        or len(spec["cells"]) != 35
        or len({tuple(c["bucket_xy_m"]) for c in spec["cells"]}) != 35
    ):
        parser.error("Expected the sealed original 35 distinct bucket sites")
    if args.probe_audit is not None or args.seed_native is not None:
        if args.probe_audit is None or args.seed_native is None:
            parser.error("Probe construction requires --probe-audit and --seed-native")
        output = probe_manifest(spec, args.assets, args.seed_native, args.probe_audit)
        args.output.write_text(json.dumps(output, allow_nan=False) + "\n")
        print(
            json.dumps(
                dict(
                    probe_bank=str(args.output),
                    options_per_cell=[len(m) for m in output["menus"]],
                    added=len(output["diagnostic_actions"]),
                )
            )
        )
        return
    links = {
        name: chain(ET.parse(args.assets / filename).getroot())
        for name, filename in (("Arm10", "dropper.urdf"), ("Arm14", "board_arm.urdf"))
    }
    spheres = {name: reach_sphere(link) for name, link in links.items()}
    poses = json.loads((args.assets / "poses.json").read_text())
    indices = args.indices if args.indices is not None else list(range(35))
    if len(set(indices)) != len(indices) or any(i not in range(35) for i in indices):
        parser.error("Choose distinct original indices in0..34")
    cells = []
    for i in indices:
        cell = spec["cells"][i]
        record = analyze_cell(spec["scene"], cell, spheres, xy_step=args.xy_step)
        loose = analyze_cell(
            spec["scene"], cell, spheres, xy_step=args.xy_step, normal_wrist_bound=False
        )
        record["loose_full_link_sphere_comparison"] = {
            name: {
                key: loose[name][key]
                for key in (
                    "model_matched_strata",
                    "max_model_range_m",
                    "max_model_range_minus_required_distance_m",
                )
            }
            for name in DOMAINS
        }
        alternatives = {
            offset: analyze_cell(
                spec["scene"], cell, spheres, xy_step=args.xy_step, rim_offset_r=offset
            )
            for offset in (-1.0, 1.0)
        }
        record["other_rim_planes"] = {
            str(offset): {
                name: {
                    key: data[name][key]
                    for key in (
                        "model_matched_strata",
                        "max_model_range_m",
                        "max_model_range_minus_required_distance_m",
                        "best_margin_action",
                    )
                }
                for name in DOMAINS
            }
            for offset, data in alternatives.items()
        }
        for domain in DOMAINS:
            candidates = record[domain]["promising_model_candidates"]
            if not candidates:
                candidates = alternatives[-1.0][domain]["promising_model_candidates"]
            if not candidates and record[domain]["best_margin_action"] is not None:
                best = record[domain]["best_margin_action"]
                candidates = []
                for reduction in (0, 0.02, 0.04, 0.08):
                    candidate = dict(best, action=dict(best["action"]))
                    candidate["action"]["gap_m"] -= reduction
                    candidate["model_match_error_m"] = float(
                        model_range(
                            spec["scene"],
                            candidate["action"]["height_m"],
                            candidate["action"]["gap_m"],
                            candidate["action"]["tilt_deg"],
                            0,
                        )
                        - best["required_distance_m"]
                    )
                    candidate["rim_plane_offset_radii"] = 0
                    candidates.append(candidate)
            record[domain]["strict_ik_checks"] = (
                strict_ik_candidates(
                    spec["scene"], cell, candidates, links, poses, args.ik_checks
                )
                if i % 7 in (5, 6)
                else []
            )
        cells.append(record)
        print(
            json.dumps(
                dict(
                    cell_id=cell["id"],
                    domains={
                        k: dict(
                            matches=record[k]["model_matched_strata"],
                            max_range=record[k]["max_model_range_m"],
                            margin=record[k][
                                "max_model_range_minus_required_distance_m"
                            ],
                            strict_success=any(
                                r["endpoint_geometry_accepted"]
                                for r in record[k]["strict_ik_checks"]
                            ),
                        )
                        for k in DOMAINS
                    },
                )
            ),
            flush=True,
        )
    result = dict(
        version=VERSION,
        source_spec_sha256=hashlib.sha256(args.spec.read_bytes()).hexdigest(),
        source_fingerprint=spec["fingerprint"],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        asset_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                args.assets / "dropper.urdf",
                args.assets / "board_arm.urdf",
                args.assets / "poses.json",
            )
        },
        domains=DOMAINS,
        grid=dict(
            xy_step_m=args.xy_step,
            height_step_m=0.02,
            tilt_step_deg=2,
            nested_current_in_wider=True,
        ),
        rim_plane_ball_center_offset_radii=0.0,
        alternative_plane_offsets_radii=[-1.0, 1.0],
        restitution=spec["scene"]["ball"]["restitution"],
        reach_spheres=spheres,
        normal_aware_rule="Remove terminal offset along flange normal before applying proximal link sphere. Dropper normal is downward; board flange normal equals desired board normal. Terminal perpendicular rounding residual expands the proximal sphere.",
        opening_scope="Primary target is ball center at rim height; alternatives cover approach(+radius) and whole-ball-entry(-radius) planes. A plane-center range match does not establish clearance through the full mouth corridor or retention.",
        cells=cells,
        scope="Frictionless model plus conservative URDF reach spheres and a bounded strict endpoint IK diagnostic. No native ball releases, capture labels, path certification, changed production bounds or impossibility claim.",
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
