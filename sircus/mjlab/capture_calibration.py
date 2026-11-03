"""Independent current-ball contact-envelope calibration and verification.

Fixtures are isolated compiled bucket/ball components, not completed robot shots.
The fit and verification inputs are fixed before collecting their outcomes. The
benchmark's separate native pilot validates actual board-bounce capture routes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np


VERSION = "sircus-independent-contact-controls-v2"
CALIBRATION_VERSION = "sircus-contact-capture-calibration-v1"
FIT_RULE = "1.25 * maximum fit contact/inside-plane overlap + 8um; capped below wall thickness and half base thickness"
GEOMETRY_EPS_M = 2e-6
INPUT_SPEED_LIMIT_M_S = 6.0
VERIFIED_SPEED_LIMIT_M_S = 6.25


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def isolated_spec(scene, assets):
    """Retain exact authored bucket/ball materials and compiled mesh geometry."""
    import mujoco
    from .model import build_scene

    root = ET.fromstring(build_scene(scene, Path(assets), solref=(.004, .15), contact_friction=.35))
    world = root.find("worldbody")
    for child in list(world):
        if child.tag != "body" or child.get("name") not in ("bucket", "ball"):
            world.remove(child)
    for tag in ("actuator", "contact", "sensor", "equality", "keyframe"):
        child = root.find(tag)
        if child is not None:
            root.remove(child)
    return mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))


def option_record(model):
    keys = ("timestep", "solver", "iterations", "tolerance", "cone", "integrator",
            "ccd_iterations", "ccd_tolerance", "disableflags", "enableflags")
    result = {}
    for key in keys:
        value = getattr(model.opt, key)
        result[key] = float(value) if isinstance(value, (float, np.floating)) else int(value)
    result["gravity"] = model.opt.gravity.tolist()
    return result


def compiled_geometry(model, scene):
    """Extract cavity planes from compiler-transformed collision hulls."""
    import mujoco
    from scipy.spatial import ConvexHull

    data = mujoco.MjData(model)
    data.mocap_pos[model.body_mocapid[model.body("bucket").id]] = 0
    mujoco.mj_forward(model, data)
    walls, planes, vertices, facet_errors = [], [], [], []
    for index in range(scene["bucket"]["segments"]):
        geom = model.geom(f"bucket/wall{index:02}").id
        mesh = int(model.geom_dataid[geom])
        begin, size = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        world = model.mesh_vert[begin:begin + size] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        equations = ConvexHull(world).equations
        angle = (index + .5) * 2 * math.pi / scene["bucket"]["segments"]
        inward = np.array([-math.cos(angle), -math.sin(angle), 0.0])
        candidates = equations[equations[:, :3] @ inward > .98]
        if not len(candidates):
            raise ValueError("No compiler-derived inner wall face")
        # Compiler float vertices need not be exactly coplanar. Compare their
        # distance in metres, not dimensionless normal coefficients.
        plane = candidates.mean(axis=0)
        plane = plane / np.linalg.norm(plane[:3])
        extent = max(scene["bucket"]["outer_top_diameter_m"]/2, .11)
        probes = np.asarray([[x, y, z, 1.] for x in (-extent, extent)
                             for y in (-extent, extent)
                             for z in (0., scene["bucket"]["assumed_interior_depth_m"] + scene["bucket"]["assumed_bottom_thickness_m"])])
        error_m = float(np.max(np.abs(probes @ (candidates-plane).T)))
        if error_m > GEOMETRY_EPS_M:
            raise ValueError("Compiled wall facets exceed the geometric precision envelope")
        facet_errors.append(error_m)
        if plane[3] <= 0 or abs(plane[2]) > .2:
            raise ValueError("Invalid cavity-facing wall plane")
        planes.append(plane.tolist())
        vertices.append(world)
        walls.append(geom)
    points = np.concatenate(vertices)
    base = model.geom("bucket/base").id
    floor = float(data.geom_xpos[base, 2] + model.geom_size[base, 1])
    rim = float(points[:, 2].max())
    b = scene["bucket"]
    if abs(floor - b["assumed_bottom_thickness_m"]) > GEOMETRY_EPS_M or abs(rim - floor - b["assumed_interior_depth_m"]) > GEOMETRY_EPS_M:
        raise ValueError("Compiled vertical bucket geometry differs from scene")
    thickness = min((b["outer_bottom_diameter_m"] - b["inner_bottom_diameter_m"]) / 2,
                    (b["outer_top_diameter_m"] - b["inner_top_diameter_m"]) / 2)
    plane_array = np.asarray(planes)
    angle = (np.arange(len(planes)) + .5) * 2 * np.pi / len(planes)
    k = (b["inner_top_diameter_m"] - b["inner_bottom_diameter_m"]) / (2 * b["assumed_interior_depth_m"])
    c = math.cos(math.pi / len(planes))
    theoretical = np.column_stack([-np.cos(angle), -np.sin(angle), np.full(len(planes), k*c), np.full(len(planes), c*(b["inner_bottom_diameter_m"]/2-k*floor))]) / math.sqrt(1+(k*c)**2)
    delta = np.abs(plane_array-theoretical)
    plane_error = float(np.max(delta[:, 3] + .11*delta[:, 0] + .11*delta[:, 1] + rim*delta[:, 2]))
    if plane_error > GEOMETRY_EPS_M:
        raise ValueError("Authored/compiled cavity planes disagree")
    ball = model.geom("ball/geom").id
    return dict(
        compiled_planes=planes, compiled_floor_m=floor, compiled_rim_m=rim,
        authored_compiled_plane_max_error=plane_error,
        compiled_facet_plane_distance_bound_m=max(facet_errors),
        minimum_radial_wall_thickness_m=thickness,
        wall_tolerance_cap_m=.95*thickness,
        floor_tolerance_cap_m=.5*b["assumed_bottom_thickness_m"],
        ball_radius_m=float(model.geom_size[ball, 0]),
        contact_geom_ids=dict(ball=ball, walls=walls, base=base),
        effective_options=option_record(model),
        ball_contact=dict(solref=model.geom_solref[ball].tolist(), solimp=model.geom_solimp[ball].tolist(),
                          friction=model.geom_friction[ball].tolist(), margin=float(model.geom_margin[ball]), gap=float(model.geom_gap[ball])),
        geometry_sha256=digest(dict(planes=planes, floor=floor, rim=rim, radius=float(model.geom_size[ball, 0]))),
    )


def fixtures(scene, role):
    """Independent deterministic fit/verification input grids, not shot search."""
    r = scene["ball"]["radius_m"]
    floor = scene["bucket"]["assumed_bottom_thickness_m"]
    rim = floor + scene["bucket"]["assumed_interior_depth_m"]
    result = []

    def add(label, p, v, bucket=(0., 0.)):
        result.append(dict(id=f"{role}-{label}", position_local_m=list(p), velocity_m_s=list(v), bucket_xy_m=list(bucket)))

    if role == "fit":
        for i, height in enumerate((floor+r, .25, .75, 1.2875)):
            add(f"vertical-{i}", (0, 0, height), (0, 0, 0))
        speeds, heights, phases = (1., 3., 6.), (floor+r+.01, rim-r-.01), (0., math.pi/64)
    elif role == "verification":
        for i, height in enumerate((floor+r+.001, .4, 1.05)):
            add(f"vertical-{i}", (.006, -.004, height), (0, 0, 0), (.31, -.27))
        speeds, heights, phases = (.5, 2., 4.5, 5.75), (floor+r+.007, rim-r-.006), (math.pi/128, 3*math.pi/128)
    else:
        raise ValueError("Unknown contact-control role")
    for direction in range(4):
        for p, phase in enumerate(phases):
            angle = direction*math.pi/2+phase
            for h, height in enumerate(heights):
                for speed in speeds:
                    add(f"wall-{direction}-{p}-{h}-{speed:g}", (0, 0, height), (speed*math.cos(angle), speed*math.sin(angle), 0),
                        (0., 0.) if role == "fit" else (.31, -.27))
    for i, offset in enumerate((-.004, -.001, .001, .004)):
        x = scene["bucket"]["inner_top_diameter_m"]/2-r+offset
        add(f"rim-{i}", (x, 0, rim+r+.04), (-.08, 0, -.5), (0., 0.) if role == "fit" else (-.22, .18))
    add("outside", (.20, 0, rim+.12), (0, 0, 0))
    return result


def run_component_controls(scene, assets, directory, *, substeps=64, role="fit", reverse_order=False):
    import torch
    import warp as wp
    from mjlab.sim import Simulation, SimulationCfg, MujocoCfg
    from mjlab.sensor import ContactSensorCfg, ContactMatch
    from .calibrated_native_log import DenseNativeLog

    if torch.cuda.device_count() != 1 or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Select only the RTX4090 by UUID")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    cases = fixtures(scene, role)
    if reverse_order:
        cases.reverse()
    write_json(directory/"fixture-manifest.json", dict(version=VERSION, role=role, substeps=substeps,
        fixture_order="reversed" if reverse_order else "canonical",
        input_speed_limit_m_s=INPUT_SPEED_LIMIT_M_S, verified_speed_limit_m_s=VERIFIED_SPEED_LIMIT_M_S,
        fit_rule=FIT_RULE, duration_s=4, cases=cases, fingerprint=digest(cases)))
    spec = isolated_spec(scene, assets)
    sensor = ContactSensorCfg(name="bucket", primary=ContactMatch(mode="body", pattern="bucket"),
                              secondary=ContactMatch(mode="geom", pattern="ball/geom"), fields=("force",), reduce="netforce").build()
    sensor.edit_spec(spec, {})
    model = spec.compile()
    cfg = SimulationCfg(mujoco=MujocoCfg(timestep=1/(240*substeps)), nconmax=512, njmax=2048)
    sim = Simulation(num_envs=len(cases), cfg=cfg, model=model, device="cuda:0")
    sensor.initialize(sim.mj_model, sim.model, sim.data, "cuda:0")
    geometry = compiled_geometry(sim.mj_model, scene)
    write_json(directory/"compiled-geometry.json", geometry)
    sim.reset()
    ball_q = int(sim.mj_model.jnt_qposadr[sim.mj_model.joint("ball/free").id])
    ball_v = int(sim.mj_model.jnt_dofadr[sim.mj_model.joint("ball/free").id])
    mocap = int(sim.mj_model.body_mocapid[sim.mj_model.body("bucket").id])
    tags = np.zeros(sim.mj_model.ngeom, dtype=np.int32)
    for i in range(scene["bucket"]["segments"]):
        tags[sim.mj_model.geom(f"bucket/wall{i:02}").id] = 1
    tags[sim.mj_model.geom("bucket/base").id] = 2
    ball_geom = sim.mj_model.geom("ball/geom").id
    ticks = 240*substeps*4
    stream = torch.cuda.ExternalStream(wp.get_stream(sim.wp_device).cuda_stream)
    start = time.monotonic()
    with torch.cuda.stream(stream):
        sim.data.mocap_pos[:, mocap, :] = torch.tensor([[*c["bucket_xy_m"], 0] for c in cases], device="cuda:0")
        sim.data.mocap_quat[:, mocap, :] = torch.tensor([1., 0., 0., 0.], device="cuda:0")
        sim.data.qpos[:] = torch.tensor(np.tile(sim.mj_model.qpos0, (len(cases), 1)), device="cuda:0", dtype=torch.float32)
        sim.data.qpos[:, ball_q:ball_q+3] = torch.tensor([[c["position_local_m"][0]+c["bucket_xy_m"][0], c["position_local_m"][1]+c["bucket_xy_m"][1], c["position_local_m"][2]] for c in cases], device="cuda:0")
        sim.data.qvel[:] = 0
        sim.data.qvel[:, ball_v:ball_v+3] = torch.tensor([c["velocity_m_s"] for c in cases], device="cuda:0")
        sim.forward()
        sensor.reset()
        log = DenseNativeLog(sim, len(cases), ticks, ball_q, ball_v, ball_geom, tags)
        force = torch.zeros((len(cases), 6), device="cuda:0")
        for tick in range(ticks):
            sim.step()
            sensor.update(1/(240*substeps))
            force[:, 1] = torch.linalg.vector_norm(sensor.data.force, dim=-1).sum(1)
            log.step(tick, force)
        torch.cuda.synchronize()
        arrays = log.arrays()
    np.savez_compressed(directory/"native-components.npz", **arrays, record_ids=np.asarray([c["id"] for c in cases]))
    summary = summarize_controls(scene, geometry, cases, arrays)
    write_json(directory/"component-summary.json", dict(version=VERSION, role=role, seconds=time.monotonic()-start,
        substeps=substeps, geometry_sha256=geometry["geometry_sha256"], fixture_sha256=sha(directory/"fixture-manifest.json"),
        raw_sha256=sha(directory/"native-components.npz"), rows=summary,
        scope="Isolated compiled bucket/ball contact controls; not completed board-bounce shots"))
    return geometry, summary


def summarize_controls(scene, geometry, cases, arrays):
    planes = np.asarray(geometry["compiled_planes"])
    radius = scene["ball"]["radius_m"]
    rows = []
    for i, case in enumerate(cases):
        p = arrays["position_m"][:, i].astype(float).copy()
        p[:, :2] -= case["bucket_xy_m"]
        vel = arrays["velocity_m_s"][:, i]
        gap = (p @ planes[:, :3].T + planes[:, 3]).min(1)
        cavity = (gap >= 0) & (p[:, 2]+radius <= geometry["compiled_rim_m"]+GEOMETRY_EPS_M) & (p[:, 2] >= geometry["compiled_floor_m"])
        distance = arrays["contact_distance_m"][:, i]
        contact = np.isfinite(distance).any(1)
        domain = cavity[1:] | contact
        floor_overlap = geometry["compiled_floor_m"]+radius-p[:, 2]
        rows.append(dict(id=case["id"], contact_wall_overlap_m=max(0., -float(np.min(distance[:, 0]))),
            contact_floor_overlap_m=max(0., -float(np.min(distance[:, 1]))),
            cavity_wall_overlap_m=max(0., float(np.max(radius-gap[cavity]))) if cavity.any() else 0.,
            cavity_floor_overlap_m=max(0., float(np.max(floor_overlap[cavity]))) if cavity.any() else 0.,
            max_bucket_domain_speed_m_s=float(np.linalg.norm(vel[1:][domain], axis=1).max()) if domain.any() else 0.,
            final_position_local_m=p[-1].tolist(),
            native_clock_drift_s=float(np.max(np.abs(arrays["native_time_s"][:, i]-arrays["time_s"])))))
    return rows


def fit_config(geometry, rows):
    wall = max(max(r["contact_wall_overlap_m"], r["cavity_wall_overlap_m"]) for r in rows)
    floor = max(max(r["contact_floor_overlap_m"], r["cavity_floor_overlap_m"]) for r in rows)
    config = dict(version=CALIBRATION_VERSION, compiled_planes=geometry["compiled_planes"],
        compiled_floor_m=geometry["compiled_floor_m"], compiled_rim_m=geometry["compiled_rim_m"],
        wall_tolerance_m=1.25*wall+8e-6, floor_tolerance_m=1.25*floor+8e-6,
        wall_tolerance_cap_m=geometry["wall_tolerance_cap_m"], floor_tolerance_cap_m=geometry["floor_tolerance_cap_m"],
        geometry_epsilon_m=GEOMETRY_EPS_M, max_speed_m_s=VERIFIED_SPEED_LIMIT_M_S, retention_s=1.)
    if config["wall_tolerance_m"] >= config["wall_tolerance_cap_m"] or config["floor_tolerance_m"] >= config["floor_tolerance_cap_m"]:
        raise ValueError("Measured penetration exceeds the predeclared thickness cap")
    config["fingerprint"] = digest(config)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "assets", "output", "gpu-lock"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--role", choices=("fit", "verification"), required=True)
    parser.add_argument("--substeps", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--reverse-order", action="store_true")
    args = parser.parse_args()
    import fcntl
    from .sweep_worker import load_reference
    ref = load_reference(args.reference, args.assets)
    # Existing shared lock is intentionally read-only to this account. flock's
    # exclusive advisory lock does not require changing its contents or mode.
    with args.gpu_lock.open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        geometry, rows = run_component_controls(ref["scene"], args.assets, args.output, substeps=args.substeps, role=args.role,
                                               reverse_order=args.reverse_order)
    if args.role == "fit":
        config = fit_config(geometry, rows)
        write_json(args.output/"candidate-score-config.json", config)
        print(json.dumps(dict(status="fit-complete", cases=len(rows), score_config=config)), flush=True)
    else:
        print(json.dumps(dict(status="verification-components-complete", cases=len(rows))), flush=True)


if __name__ == "__main__":
    main()
