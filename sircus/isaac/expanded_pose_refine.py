"""Refine geometry around measured native holds, never ball-release outcomes."""

import copy
import hashlib
import itertools
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from expanded_pose_menu import (
    _action,
    canonical_hash,
    endpoint_targets,
    geometry_reason,
    geometry_signature,
    make_manifest,
    outer_bounds,
)
from prepare_assets import chain
from settling_contract import accepted

VERSION = "sircus-native-hold-refinement-v1"
LOCAL_VERSION = "sircus-branch-local-hold-refinement-v1"


def read_manifest(path, qualification):
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    unsigned = dict(value)
    fingerprint = unsigned.pop("fingerprint", None)
    if (
        canonical_hash(unsigned) != fingerprint
        or value.get("qualification") != qualification
    ):
        raise ValueError("Hold refinement input checksum or qualification mismatch")
    return value, hashlib.sha256(raw).hexdigest()


def held_seed(native, cell_index):
    """Use only a requested pose whose original native hold actually passed."""
    options, fits = native["menus"][cell_index], native["fits"][cell_index]
    if len(options) != len(fits) or not options:
        raise ValueError("Requested cell has no complete native-held seed")
    proofs = {
        row["action_id"]: row
        for row in native["native_checks"][cell_index]
        if row.get("accepted")
    }
    action, fit = options[0], fits[0]
    proof = proofs.get(action["id"])
    if not proof or proof.get("action") != action or proof.get("invalid") is not None:
        raise ValueError("Native-held seed has no matching measured proof")
    measured = proof["settled"]
    keys = (
        "position_error_mm",
        "normal_error_deg",
        "joint_speed_rad_s",
        "position_drift_mm",
        "normal_drift_deg",
        "dropper_position_error_mm",
    )
    values = [measured[key] for key in keys]
    if (
        not np.isfinite(values).all()
        or min(values) < 0
        or not accepted(measured)
        or measured["dropper_position_error_mm"] > 2
    ):
        raise ValueError("Native-held seed failed the unchanged release guard")
    return copy.deepcopy(action), copy.deepcopy(fit), copy.deepcopy(proof)


def refinement_actions(
    scene, cell, seed, limit=256, *, local_only=False, board_first=False
):
    """Small geometry variations, with optional board-first local ordering."""
    if not 1 <= limit <= 1024:
        raise ValueError("Refinement limit must be 1..1024")
    gap_changes = [
        dict(gap_m=seed["gap_m"] + d)
        for d in (
            (-0.01, 0.01, -0.02, 0.02)
            if local_only
            else (-0.01, 0.01, -0.02, 0.02, -0.04, 0.04)
        )
    ]
    board_changes = [
        dict(height_m=seed["height_m"] + d)
        for d in (
            (-0.01, 0.01, -0.02, 0.02) if local_only else (-0.02, 0.02, -0.04, 0.04)
        )
    ]
    if local_only:
        board_changes += [
            dict(tilt_deg=seed["tilt_deg"] + d) for d in (-1.0, 1.0, -2.0, 2.0)
        ]
    board_changes += [
        dict(**{key: seed[key] + d})
        for d in (
            (-0.005, 0.005, -0.01, 0.01) if local_only else (-0.005, 0.005, -0.02, 0.02)
        )
        for key in ("board_x_m", "board_y_m")
    ]
    if not local_only:
        board_changes += [
            dict(tilt_deg=t) for t in (50.0, 48.0, 52.0) if t != seed["tilt_deg"]
        ]
    if board_first:
        changes = []
        for start in range(0, len(board_changes), 2):
            changes += board_changes[start : start + 2]
            if start // 2 < len(gap_changes):
                changes.append(gap_changes[start // 2])
    else:
        changes = gap_changes + board_changes
    changes += [
        dict(
            board_x_m=seed["board_x_m"] + dx,
            board_y_m=seed["board_y_m"] + dy,
            height_m=seed["height_m"] + dh,
            gap_m=seed["gap_m"] + dg,
            tilt_deg=tilt,
        )
        for dx, dy, dh, dg, tilt in itertools.product(
            (-0.005, 0, 0.005),
            (-0.005, 0, 0.005),
            ((-0.01, 0, 0.01) if local_only else (-0.02, 0, 0.02)),
            ((-0.01, 0, 0.01) if local_only else (-0.02, 0, 0.02)),
            (
                (seed["tilt_deg"] - 1, seed["tilt_deg"], seed["tilt_deg"] + 1)
                if local_only
                else (48.0, 50.0, 52.0)
            ),
        )
    ]
    result, seen = [], {geometry_signature(seed)}
    for change in changes:
        values = dict(seed, **change)
        action = _action(
            cell,
            values["board_x_m"],
            values["board_y_m"],
            values["height_m"],
            values["gap_m"],
            values["tilt_deg"],
            f"{'branch-local' if local_only else 'hold-refine'}-{cell['index']:02d}-{len(result):04d}",
        )
        signature = geometry_signature(action)
        if signature in seen or geometry_reason(scene, cell, action):
            continue
        seen.add(signature)
        result.append(action)
        if len(result) >= limit:
            break
    return result


def refine(
    offline_path,
    native_path,
    assets,
    cells=(5,),
    append=12,
    max_checks=128,
    progress=None,
    *,
    local_only=False,
    max_joint_delta=None,
    board_first=False,
):
    """Append strict CPU fits while retaining every other native-held workcell."""
    from expanded_pose_native import verified_fit
    from robust_ik import solve

    if not 1 <= append <= 32 or not 1 <= max_checks <= 512:
        raise ValueError("Use 1..32 new fits and 1..512 finite checks")
    if local_only and max_joint_delta is None:
        max_joint_delta = 0.35
    if max_joint_delta is not None and (
        not np.isfinite(max_joint_delta) or not 0 < max_joint_delta <= np.pi
    ):
        raise ValueError("Maximum joint delta must be finite and between zero and pi")
    offline, offline_hash = read_manifest(offline_path, "offline_endpoint_ik")
    native, native_hash = read_manifest(native_path, "native_settled_endpoints")
    if (
        canonical_hash(offline["spec"]) != canonical_hash(native["spec"])
        or offline["source_fingerprint"] != native["source_fingerprint"]
    ):
        raise ValueError("Offline and native inputs describe different fixed scenes")
    spec = offline["spec"]
    if (
        len(spec["cells"]) != 35
        or len(native["menus"]) != 35
        or len(native["fits"]) != 35
    ):
        raise ValueError("Refinement preserves all 35 original workcells")
    if len(set(cells)) != len(cells) or any(i not in range(35) for i in cells):
        raise ValueError("Choose distinct existing cell indices")
    assets = Path(assets)
    links = {
        name: chain(ET.parse(assets / filename).getroot())
        for name, filename in (("Arm10", "dropper.urdf"), ("Arm14", "board_arm.urdf"))
    }
    menus, fits = copy.deepcopy(native["menus"]), copy.deepcopy(native["fits"])
    checks = copy.deepcopy(offline["checks"])
    details = []
    for index in cells:
        seed, seed_fit, proof = held_seed(native, index)
        original_count = len(menus[index])
        if local_only:
            menus[index], fits[index] = [copy.deepcopy(seed)], [copy.deepcopy(seed_fit)]
        targets = endpoint_targets(spec["scene"], spec["cells"][index], seed)
        for name in ("Arm10", "Arm14"):
            verified_fit(targets[name], seed_fit[name], links[name], name == "Arm14")
        cache = {
            (name, tuple(targets[name].ravel())): seed_fit[name]
            for name in ("Arm10", "Arm14")
        }
        signatures = {geometry_signature(a) for a in menus[index]}
        ids = {a["id"] for a in menus[index]}
        attempted, added = 0, 0
        for action in refinement_actions(
            spec["scene"],
            spec["cells"][index],
            seed,
            limit=512,
            local_only=local_only,
            board_first=board_first,
        ):
            if added >= append or attempted >= max_checks:
                break
            if geometry_signature(action) in signatures:
                continue
            if action["id"] in ids:
                raise ValueError("Refinement action ID collides with existing evidence")
            other = endpoint_targets(spec["scene"], spec["cells"][index], action)
            if not outer_bounds(other, links)["eligible"]:
                checks[index].append(
                    dict(
                        action_id=action["id"],
                        reason="outer_link_length_bound",
                        ik_checked=False,
                        seed_action_id=seed["id"],
                    )
                )
                continue
            attempted += 1
            result = {}
            reason = None
            joint_deltas = {}
            for name in ("Arm10", "Arm14"):
                key = (name, tuple(other[name].ravel()))
                if key not in cache:
                    # Every solve uses the original native-good branch. A bad
                    # or merely last-solved candidate never becomes the seed.
                    cache[key] = solve(
                        other[name], links[name], seed_fit[name]["q"], name == "Arm14"
                    )
                result[name] = cache[key]
                if not result[name]["solved"]:
                    reason = "endpoint_ik_" + name
                    break
                verified_fit(other[name], result[name], links[name], name == "Arm14")
                joint_deltas[name] = float(
                    np.max(
                        np.abs(
                            np.asarray(result[name]["q"])
                            - np.asarray(seed_fit[name]["q"])
                        )
                    )
                )
                if (
                    max_joint_delta is not None
                    and joint_deltas[name] > max_joint_delta + 1e-12
                ):
                    reason = "joint_branch_delta_" + name
                    break
            checks[index].append(
                dict(
                    action_id=action["id"],
                    reason=reason,
                    ik_checked=True,
                    ik=result,
                    seed_action_id=seed["id"],
                    joint_delta_rad=joint_deltas,
                )
            )
            if reason is None:
                menus[index].append(action)
                fits[index].append(result)
                ids.add(action["id"])
                signatures.add(geometry_signature(action))
                added += 1
            if progress:
                progress(
                    dict(
                        cell=index,
                        seed_action_id=seed["id"],
                        checks=attempted,
                        appended=added,
                        action_id=action["id"],
                        reason=reason,
                    )
                )
        details.append(
            dict(
                cell=index,
                seed_action=seed,
                seed_native_proof=proof,
                original_good_fit=seed_fit,
                checks=attempted,
                appended=added,
                prior_native_options=original_count,
                local_only=bool(local_only),
                max_joint_delta_rad=max_joint_delta,
            )
        )
    return make_manifest(
        spec,
        menus,
        fits=fits,
        checks=checks,
        qualification="offline_endpoint_ik",
        source_original_geometry_fp=spec.get("camera_source_fingerprint"),
        refinement_protocol=LOCAL_VERSION if local_only else VERSION,
        local_only=bool(local_only),
        board_first=bool(board_first),
        max_joint_delta_rad=max_joint_delta,
        prior_offline_sha256=offline_hash,
        prior_native_sha256=native_hash,
        refinement=details,
        summary=dict(
            accepted_per_cell=[len(m) for m in menus],
            new_endpoint_fits=sum(d["appended"] for d in details),
        ),
        scope=(
            "Branch-local geometry around one native-held anchor per requested cell; not global pose coverage or collision-free transition certification. "
            if local_only
            else "Geometry/native-hold seed refinement only. "
        )
        + "No ball releases, reward or capture labels. Every output endpoint awaits fresh native qualification.",
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("offline", "native", "assets", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cells", type=int, nargs="+", default=[5])
    parser.add_argument("--append", type=int, default=12)
    parser.add_argument("--max-checks", type=int, default=128)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--max-joint-delta", type=float)
    parser.add_argument("--board-first", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new refinement evidence file")
    result = refine(
        args.offline,
        args.native,
        args.assets,
        args.cells,
        args.append,
        args.max_checks,
        lambda row: print(json.dumps(row), flush=True),
        local_only=args.local_only,
        max_joint_delta=args.max_joint_delta,
        board_first=args.board_first,
    )
    args.output.write_text(json.dumps(result, allow_nan=False))
    print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
