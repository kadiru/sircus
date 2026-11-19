"""Confirm expanded geometry candidates with unchanged native settling gates."""

import hashlib
import json
from pathlib import Path

import numpy as np
from prepare_assets import fk
from scipy.spatial.transform import Rotation
from settling_contract import accepted


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def verified_fit(target, record, links, normal_only):
    """Independently check an offline fit before it enters the native IK cache."""
    q = np.asarray(record["q"], dtype=float)
    lower, upper = np.array([x[2] for x in links]), np.array([x[3] for x in links])
    if (
        q.shape != (6,)
        or not np.isfinite(q).all()
        or np.any(q < lower)
        or np.any(q > upper)
    ):
        raise ValueError("Offline IK joints are nonfinite or outside limits")
    current = fk(q, links)
    angular = (
        current[:3, 2] - target[:3, 2]
        if normal_only
        else Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()
    )
    error = np.r_[current[:3, 3] - target[:3, 3], 0.25 * angular]
    if not record["solved"] or np.linalg.norm(error) >= 1e-4:
        raise ValueError("Offline IK fit does not satisfy the original residual gate")
    return dict(record)


def qualify_native_menu(env, spec, path, desired_per_cell=4):
    """No ball releases or rewards; accept only measured settled endpoints."""
    from expanded_farm import validate_action
    from expanded_pose_menu import endpoint_targets, load_native_menu, make_manifest
    from parallel_runtime import atomic

    path = Path(path)
    raw = path.read_bytes()
    source = json.loads(raw)
    unsigned = dict(source)
    digest = unsigned.pop("fingerprint")
    if (
        canonical_hash(unsigned) != digest
        or source["source_fingerprint"] != spec["fingerprint"]
        or canonical_hash(source["spec"]) != canonical_hash(spec)
    ):
        raise ValueError("Expanded candidate source fingerprint mismatch")
    menus, fits = source["menus"], source["fits"]
    if len(menus) != env.n or len(fits) != env.n or env.n != 35:
        raise ValueError("Expanded qualification requires all 35 fixed cells")
    if source["qualification"] == "native_settled_endpoints":
        load_native_menu(path, spec)
    elif source["qualification"] != "offline_endpoint_ik":
        raise ValueError("Unsupported expanded geometry qualification")
    for i, (options, solutions) in enumerate(zip(menus, fits)):
        if len(options) != len(solutions):
            raise ValueError("Expanded menu has missing endpoint solutions")
        for action, solutions_by_arm in zip(options, solutions):
            validate_action(spec["cells"][i], action)
            targets = endpoint_targets(spec["scene"], spec["cells"][i], action)
            for name in ("Arm10", "Arm14"):
                target = targets[name]
                record = verified_fit(
                    target, solutions_by_arm[name], env.links, name == "Arm14"
                )
                env.ik_cache[(name, tuple(target.ravel()))] = record
    accepted_menus, accepted_fits, checks = ([[] for _ in menus] for _ in range(3))
    maximum = max(map(len, menus), default=0)
    if not maximum or any(not options for options in menus):
        atomic(
            env.dest / "expanded-menu-coverage.json",
            dict(phase="offline_incomplete", options_per_cell=[len(m) for m in menus]),
        )
        raise RuntimeError(
            "Offline geometry search has not found candidates in all 35 cells"
        )
    env.training_active = False
    for j in range(maximum):
        enabled = np.array(
            [
                j < len(m) and len(a) < desired_per_cell
                for m, a in zip(menus, accepted_menus)
            ]
        )
        if not enabled.any():
            break
        actions = [m[j] if j < len(m) else m[0] for m in menus]
        # For qualification, preserve already finished stations in every target
        # write. The regular training mixin applies this rule during transit.
        starts = {name: arm.data.joint_pos.clone() for name, arm in env.arms.items()}
        env.preparation_enabled = enabled
        original_approach = env.approach_endpoints

        def hold_then_initialize(targets):
            for name in targets:
                targets[name][~enabled] = starts[name][~enabled]
            return original_approach(targets)

        env.approach_endpoints = hold_then_initialize
        try:
            _, _, records, invalid = env.prepare(actions)
        finally:
            env.approach_endpoints = original_approach
        for i in np.flatnonzero(enabled):
            metric = records[i][-1]
            passed = (
                invalid[i] is None
                and accepted(metric)
                and metric["dropper_position_error_mm"] <= 2
            )
            checks[i].append(
                dict(
                    action_id=actions[i]["id"],
                    action=actions[i],
                    accepted=bool(passed),
                    invalid=invalid[i],
                    settled=metric,
                )
            )
            if passed:
                accepted_menus[i].append(actions[i])
                accepted_fits[i].append(fits[i][j])
        atomic(
            env.dest / "expanded-menu-progress.json",
            dict(
                phase="native_qualification",
                batches=j + 1,
                options_per_cell=[len(m) for m in accepted_menus],
            ),
        )
    env.preparation_enabled = np.ones(env.n, dtype=bool)
    qualified = make_manifest(
        spec,
        accepted_menus,
        fits=accepted_fits,
        native_checks=checks,
        qualification="native_settled_endpoints",
        offline_source_sha256=hashlib.sha256(raw).hexdigest(),
        source_original_geometry_fp=spec.get("camera_source_fingerprint"),
    )
    output = env.dest / "expanded-menu-qualified.json"
    atomic(output, qualified)
    atomic(
        env.dest / "expanded-menu-coverage.json",
        dict(
            phase="native_completed",
            options_per_cell=[len(m) for m in accepted_menus],
            cells_with_two=sum(len(m) >= 2 for m in accepted_menus),
        ),
    )
    load_native_menu(output, spec)
    return output
